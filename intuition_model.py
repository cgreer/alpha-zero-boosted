from collections import defaultdict
from dataclasses import dataclass
import json
import typing
import random
from multiprocessing import Pool

import numpy
from treelite_runtime import (
    Batch as TreeliteBatch,
)

from training_samples import split_train_test, SampleData
from gbdt_model import GBDTModel
from paths import generate_tmp_path


class UnopinionatedValue:
    def predict(self, features):
        # :features ~ [(0, 1, ...), ...]
        return (0.0,) * len(features)


class UniformPolicy:
    def predict(self, features, allowable_actions):
        # Has to handle terminal state as well?
        if not allowable_actions:
            return {}
        uniform_probability = 1.0 / len(allowable_actions)
        return [uniform_probability] * len(allowable_actions)


def calculate_weights(samples, highest_generation, strategy):
    # Reweight based on generation
    meta_info = samples.meta_info
    weights = numpy.full(meta_info.shape[0], 1.0, numpy.float32)
    for row_index in range(meta_info.shape[0]):
        this_generation = meta_info[row_index][1] # 1 is generation
        generations_from_highest = highest_generation - this_generation
        weight = .2 + (.8 * (.9**generations_from_highest))
        weights[row_index] = weight

    # Renormalize weights
    # - Want weights to sum up to num samples
    if strategy == "rg2":
        weights_normed = weights / (weights.sum() / weights.shape[0])
        return weights_normed
    else:
        return weights


@dataclass
class GBDTValue(GBDTModel):
    weighting_strat: str = None
    highest_generation: int = None

    def extract_training_observations(
        self,
        samples,
        test_fraction,
    ) -> (SampleData, SampleData):
        train_samples, test_samples = split_train_test(samples, test_fraction)

        if self.weighting_strat:
            train_samples.weights = calculate_weights(train_samples, self.highest_generation, self.weighting_strat)
            test_samples.weights = calculate_weights(test_samples, self.highest_generation, self.weighting_strat)

        return train_samples, test_samples

    def train(
        self,
        samples: SampleData,
        test_fraction=.2,
    ):
        super().train(
            objective="mean_squared_error",
            eval_metrics=["mean_squared_error", "mae"],
            samples=samples,
            test_fraction=test_fraction,
        )

    def predict(self, features) -> numpy.array:
        # :features ~ [features_1, features_2, ...]
        # :features ~ [(1, 0, ...), (0, 1, ...), ...]
        # return self.treelite_predictor.predict(batch).item(0)
        return self.treelite_predictor.predict(TreeliteBatch.from_npy2d(features)).tolist()


def extract_policy_observations(features, labels):
    '''
    :features ~ [[0.0, 1.0, ...], ...], one feature set for each game position.
    :labels ~ [[.01, .92, .001, ...], ...], one label set for each game position

    When the actions per state of an environment are low, you can make a policy
    observation (state features, action probability) for every action (move)
    for every state (game position).

    However, if the environment (aka game) has a large branching factor,
    then it will...
        - Use a lot of memory per state (position)
        - Make a lot of training observations for actions that have 0.0
          probabilities
            - Without a lot of training data/time, it'll make it difficult to
              learn the top N actions for a state.

    To deal with these issues, we sample a subset of the actions in a given
    state for environments with high actions per state.

    First we sample N samples (without replacement) proportional to the
    probability of the move. Recall that the move probability was determined
    by the results of the mcts considerations for that state. This
    policy-proportionate sample ensures we likely sample the most
    favorable actions for that state. It will allow the model to understand
    the top actions to take from a given state.

    We then uniformly sample N more samples from the remaining samples that
    weren't chosen in the first sampling step to ensure we represent some
    "negative" samples.  If we didn't do this negative sampling, the global
    bias of the model would be high because the model will only have seen
    favorable moves for every state. It would assume that moves are, in
    general, high likelihood.  So these negative samples (often times 0.0
    probability in high branching-factor games) will correct the global bias
    and make our action policy probabilities more accurate.
    '''
    pdf_sample_count = 5

    print("\nExtracting policy observations")
    observation_features = []
    observation_labels = []
    for row_index in range(features.shape[0]):
        if row_index % 10_000 == 0:
            print(f"...Position {row_index}")

        position_features = features[row_index]
        move_probabilities = labels[row_index]
        action_ids = list(range(len(move_probabilities)))

        # What's the cap on the number of moves we can sample from this pdf?
        num_above_zero = 0
        for mp in move_probabilities:
            if mp > 0.0:
                num_above_zero += 1
        num_to_sample = min(num_above_zero, pdf_sample_count)

        # Sample N labels proportional to policy pdf
        pdf_samples = numpy.random.choice(
            action_ids,
            size=num_to_sample,
            replace=False,
            p=move_probabilities
        )

        # Sample N "negative" labels that didn't get picked
        remaining_ids = [x for x in action_ids if x not in pdf_samples]
        negative_samples = numpy.random.choice(
            remaining_ids,
            size=min(num_to_sample, len(remaining_ids)),
            replace=False,
        )

        # Make a policy training observation by prepending the position features
        # with the action id.
        # XXX: This will be SLOOOW. Do better. Use hstack.
        for samples in (pdf_samples, negative_samples):
            for action_id in samples:
                policy_features = numpy.concatenate(([action_id], position_features))
                observation_features.append(policy_features)
                observation_labels.append(move_probabilities[action_id])

    return (
        numpy.array(observation_features, dtype=numpy.float32),
        numpy.array(observation_labels, dtype=numpy.float32)
    )


def partition_data_to_disk(key, data, num_pieces):
    part_paths = []
    for data_part in numpy.array_split(data, num_pieces):
        part_path = generate_tmp_path(key, "npy")
        numpy.save(part_path, data_part)
        part_paths.append(part_path)
    return part_paths


def policy_extraction_worker(args):
    features_part_path, labels_part_path = args

    features = numpy.load(features_part_path)
    labels = numpy.load(labels_part_path)
    assert features.shape[0] == labels.shape[0]

    observation_features, observation_labels = extract_policy_observations(features, labels)

    of_path = generate_tmp_path("observation_features", "npy")
    numpy.save(of_path, observation_features)

    ol_path = generate_tmp_path("observation_labels", "npy")
    numpy.save(ol_path, observation_labels)

    return of_path, ol_path


@dataclass
class GBDTPolicy(GBDTModel):
    num_workers: int = 1

    def extract_policy_observations(self, samples: SampleData):
        # Split up the data into :num_workers parts
        features_part_paths = partition_data_to_disk("policy_features", samples.features, self.num_workers)
        labels_part_paths = partition_data_to_disk("policy_labels", samples.labels, self.num_workers)

        # Ship off to workers
        worker_args = []
        for features_part_path, labels_part_path in zip(features_part_paths, labels_part_paths):
            worker_args.append((features_part_path, labels_part_path))
        with Pool(len(worker_args)) as p:
            results = p.map(policy_extraction_worker, worker_args)

        # Concatenate all the worker data
        observation_features = []
        observation_labels = []
        for of_path, ol_path in results:
            observation_features.append(numpy.load(of_path))
            observation_labels.append(numpy.load(ol_path))
        observation_features = numpy.concatenate(observation_features)
        observation_labels = numpy.concatenate(observation_labels)

        return observation_features, observation_labels

    def extract_training_observations(
        self,
        game_samples: SampleData,
        test_fraction,
    ):
        train_samples, test_samples = split_train_test(game_samples, test_fraction)

        # Make policy samples for each label in (features, labels) pairs
        # - Note that this scrubbed the meta info
        print("\nBuilding policy training observations. Sit tight.")
        train_features, train_labels = self.extract_policy_observations(train_samples)
        train_samples = SampleData(
            features=train_features,
            labels=train_labels,
        )

        test_features, test_labels = self.extract_policy_observations(test_samples)
        test_samples = SampleData(
            features=test_features,
            labels=test_labels,
        )

        return train_samples, test_samples

    def train(
        self,
        samples: SampleData,
        test_fraction=.2,
    ):
        super().train(
            objective="cross_entropy",
            eval_metrics=["cross_entropy", "mae"],
            samples=samples,
            test_fraction=test_fraction,
        )

    def predict(self, agent_features, allowable_actions):
        # :agent_features ~ array[0, 1, 0, 7, ....]
        #   - This is just ONE agent's features.  Unlike for the Value Model, every node only needs
        #     the policy of the state's *moving* agent
        # :allowable_actions ~ array[0, 1, 0, 7, ....]
        if len(allowable_actions) == 1:
            return [1.0]

        # Build ndarray with policy features
        # - tile the state features with a leading placeholder feature(s) for each action
        # - overwrite the placeholder feature(s) with action values
        # XXX: Do something besides using index as feature for model
        # XXX: Will this be slower with more allowable_actions actions than just tiling?
        num_agent_features = len(agent_features)
        to_predict = numpy.empty((len(allowable_actions), num_agent_features + 1), dtype=numpy.float32)
        for i, action in enumerate(allowable_actions):
            to_predict[i][0] = action
            to_predict[i][1:num_agent_features + 1] = agent_features[0:num_agent_features]

        # Predict move probabilities
        move_probabilities = self.treelite_predictor.predict(TreeliteBatch.from_npy2d(to_predict))

        # Normalize scores to sum to 1.0
        # - The scores returned are strong attempts at probabilities that sum up to 1.0.  In fact,
        #   they already sum up to close to 1.0 without normalization.  But because of the way the
        #   training is setup (not ovr multiclass), we need to normalize to ensure they sum to 1.0.
        move_probabilities = move_probabilities / move_probabilities.sum()
        return move_probabilities.tolist()


@dataclass
class NaiveValue:
    state_visits: typing.Any = None # features: int
    state_wins: typing.Any = None # features: int

    def save(self, output_path):
        data = {
            "state_visits": list(self.state_visits.items()),
            "state_wins": list(self.state_wins.items()),
        }

        # pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            f.write(json.dumps(data))

    def load(self, model_path):
        data = open(model_path, 'r').read()
        data = json.loads(data)
        self.state_visits = {tuple(key): int(value) for (key, value) in data["state_visits"]}
        self.state_wins = {tuple(key): int(value) for (key, value) in data["state_wins"]}

    def train(self, samples, test_fraction=.2):
        raise RuntimeError("Broken")
        train_set, test_set = split_train_test(samples, test_fraction, "value")

        # "Train"
        self.state_visits = defaultdict(int)
        self.state_wins = defaultdict(int)
        for features, label in train_set:
            self.state_visits[tuple(features)] += 1
            self.state_wins[tuple(features)] += label

        # Convert them to dicts to maintain consistency with load
        self.state_visits = dict(self.state_visits)
        self.state_wins = dict(self.state_wins)

        # delete any keys that are too infrequent
        to_delete = []
        for k, v in self.state_visits.items():
            if v <= 5:
                to_delete.append(k)
        for k in to_delete:
            del self.state_visits[k]
            del self.state_wins[k]

        # "Test"
        absolute_error = 0
        absolute_error_random = 0
        for features, label in test_set:
            value = self.predict(features)
            random_value = -1.0 + (2.0 * random.random())
            absolute_error += abs(label - value)
            absolute_error_random += abs(label - random_value)
        mean_absolute_error = absolute_error / len(test_set)
        mean_absolute_error_random = absolute_error_random / len(test_set)

        print("MAE:", mean_absolute_error)
        print("MAE (random):", mean_absolute_error_random)

    def predict(self, features):
        # :features ~ [(0, 1, ...), ...]
        values = []
        for board_features in features:
            try:
                values.append(self.state_wins[tuple(features)] / self.state_visits[tuple(features)])
            except (KeyError, ZeroDivisionError):
                # XXX: How is there a ZeroDivisionError but not a key error
                values.append(0)
        return tuple(values)


@dataclass
class NaivePolicy:
    state_action_mass: typing.Any = None # tuple: float
    state_action_weight: typing.Any = None # tuple: float

    def save(self, output_path):
        data = {
            "state_action_mass": list(self.state_action_mass.items()),
            "state_action_weight": list(self.state_action_weight.items()),
        }
        with open(output_path, 'w') as f:
            f.write(json.dumps(data))

    def load(self, model_path):
        data = open(model_path, 'r').read()
        data = json.loads(data)
        self.state_action_mass = {tuple(key): float(value) for (key, value) in data["state_action_mass"]}
        self.state_action_weight = {tuple(key): float(value) for (key, value) in data["state_action_weight"]}

    def train(self, samples):
        # Don't use defaultdicts so that you can distinguish the keyerror
        self.state_action_mass = {}
        self.state_action_weight = {}
        for sample_type, features, labels in samples:
            if sample_type == "value":
                continue
            # Order is determined/fixed by environment
            for i, label in enumerate(labels):
                state_action = tuple(features + [i])
                self.state_action_mass[state_action] = self.state_action_mass.get(state_action, 0.0) + label
                self.state_action_weight[state_action] = self.state_action_weight.get(state_action, 0.0) + 1.0

        # delete any keys that are too infrequent
        to_delete = []
        for k, v in self.state_action_weight.items():
            if v <= 5:
                to_delete.append(k)
        for k in to_delete:
            del self.state_action_mass[k]
            del self.state_action_weight[k]

    def predict(self, features, allowable_actions):
        try:
            move_probabilities = []
            for i, action in enumerate(allowable_actions):
                state_action = tuple(features + [i])
                move_probabilities.append(self.state_action_mass[state_action] / self.state_action_weight[state_action])
            return move_probabilities
        except KeyError:
            # Never seen this state before; therefore, use uniform policy
            # XXX: Change this to be a list like it's other predict friends.
            uniform_probability = 1.0 / len(allowable_actions)
            return [uniform_probability] * len(allowable_actions)


if __name__ == "__main__":
    pass
