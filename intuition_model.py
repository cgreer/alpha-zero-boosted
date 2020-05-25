import time
from collections import defaultdict
from dataclasses import dataclass
import json
import sys
import typing
import random
from rich import print as rprint
import lightgbm
import numpy
import pprint
from treelite.runtime import (
    Predictor as TreelitePredictor,
    Batch as TreeliteBatch,
)
from treelite import (
    Model as TreeliteModel,
    Annotator as TreeliteAnnotator,
    DMatrix as TreeliteDMatrix,
)
from uuid import uuid4
import settings
import shutil
import itertools


def extract_policy_labels(move, environment):
    '''
    The mcts policy labels are a vector of (move visit_count / total visits) for the node that the
    mcts ran for.
    '''
    mcts_info = {}
    total_visits = 0
    for move, _, visit_count, _ in move["policy_info"]:
        # move, prior_probability, visit_count, reward_total
        mcts_info[move] = visit_count
        total_visits += visit_count

    policy_labels = []
    for env_move in environment.all_possible_actions():
        label = mcts_info.get(env_move, 0.0) / total_visits
        policy_labels.append(label)

    return policy_labels


def split_train_test(samples, test_fraction, only_of_type):
    '''
    Partition the games into training and test games.

    If the positions from the a game are in both the training and test sets, then it'll be easy for
    a model to overfit by memorizing non-generalizable aspects of a particular game.
    '''
    # Convert test fraction to percentage from 0-100
    rounded_test_percentage = int(test_fraction) # off by a bit cuz rounding...
    train_set = []
    test_set = []
    for sample_type, game_bucket, features, label in samples:
        if sample_type != only_of_type:
            continue
        which_set = test_set if (game_bucket % 100) <= rounded_test_percentage else train_set
        which_set.append((features, label))
    return train_set, test_set

    train_set = []
    test_set = []
    split_point = int(len(samples) * (1 - test_fraction))
    i = 0
    for sample_type, features, label in samples:
        i += 1
        if sample_type != only_of_type:
            continue
        if i <= split_point:
            train_set.append((features, label))
        else:
            test_set.append((features, label))
    return train_set, test_set

    i = 0
    every_n_test = int(round(1 / test_fraction))
    for sample_type, features, label in samples:
        if sample_type != only_of_type:
            continue
        i += 1

        if i % every_n_test == 0:
            test_set.append((features, label))
        else:
            train_set.append((features, label))

    return train_set, test_set


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
class GBDTValue:
    treelite_model_path: typing.Any = None
    treelite_predictor: typing.Any = None

    def save(self, output_path):
        if self.treelite_model_path is None:
            raise RuntimeError("Model hasn't been created/loaded, can't save elsewhere.")

        # Copy the current dylib file (which might be temporary path) to another location
        shutil.copyfile(self.treelite_model_path, output_path)

    def load(self, model_path, nthread=1):
        self.treelite_model_path = model_path
        self.treelite_predictor = TreelitePredictor(
            model_path,
            nthread=nthread,
        )

    def build_annotation_data(
        self,
        model,
        annotation_samples,
        output_path,
    ):
        # sample_data_path in libsvm format
        samples_matrix = TreeliteDMatrix(annotation_samples)
        ann = TreeliteAnnotator()
        ann.annotate_branch(
            model=model,
            dmat=samples_matrix,
            verbose=False,
        )
        ann.save(path=output_path)
        print("Saved branch annotations in", output_path)
        return output_path

    def build_treelite_model(
        self,
        gbdt_model_path,
        model_format="lightgbm",
        annotation_samples=None,
    ):
        treelite_model = TreeliteModel.load(
            gbdt_model_path,
            model_format=model_format,
        )

        # Build up branch expectations from data samples
        annotation_results_path = None
        if annotation_samples is not None:
            tmp_annotation_info = f"{settings.TMP_DIRECTORY}/treelite_ann-{str(uuid4())}.info"
            annotation_results_path = self.build_annotation_data(
                treelite_model,
                annotation_samples,
                tmp_annotation_info,
            )

        # Compile model to C++
        params = dict(
            parallel_comp=0,
            # quantize=1, # Supposed to speed up predictions. Didn't when I tried it.
        )
        if annotation_results_path is not None:
            params["annotate_in"] = annotation_results_path

        # Save DLL
        self.treelite_model_path = f"{settings.TMP_DIRECTORY}/model-{str(uuid4())}.dylib"
        treelite_model.export_lib(
            toolchain="clang", # clang for MacOS, gcc for unix?
            libpath=self.treelite_model_path,
            verbose=False,
            params=params,
        )
        return self.treelite_model_path

    def stash_training_data(
        self,
        train_features,
        train_labels,
        test_features,
        test_labels,
    ):
        base_path = f"{settings.TMP_DIRECTORY}/value_model-{str(uuid4())}_"
        train_path = f"{base_path}train_features.npy"
        train_labels_path = f"{base_path}train_labels.npy"
        test_path = f"{base_path}test_features.npy"
        test_labels_path = f"{base_path}test_labels.npy"
        numpy.save(train_path, train_features)
        numpy.save(train_labels_path, train_labels)
        numpy.save(test_path, test_features)
        numpy.save(test_labels_path, test_labels)
        print("\nSTASHING TRAINING DATA")
        for p in (train_path, train_labels_path, test_path, test_labels_path):
            print(p)

    @classmethod
    def load_stashed_training_data(self, base_path):
        train_path = f"{base_path}train_features.npy"
        train_labels_path = f"{base_path}train_labels.npy"
        test_path = f"{base_path}test_features.npy"
        test_labels_path = f"{base_path}test_labels.npy"

        train_features = numpy.load(train_path)
        train_labels = numpy.load(train_labels_path)
        test_features = numpy.load(test_path)
        test_labels = numpy.load(test_labels_path)

        return train_features, train_labels, test_features, test_labels

    def train(self, samples, test_fraction=.2):
        # Each set ~ [(features, label), ...]
        #  - features ~ (0, 1, .7, ...)
        #  - label is {-1, 0, 1}
        train_set, test_set = split_train_test(samples, test_fraction, "value")

        print("Building Train Matrix")
        train_features = numpy.array(tuple(numpy.array(x[0]) for x in train_set))
        train_labels = numpy.array(tuple(x[1] for x in train_set))
        # train_data = lightgbm.Dataset(data, label=label)

        print("Building Test Matrix")
        test_features = numpy.array(tuple(numpy.array(x[0]) for x in test_set))
        test_labels = numpy.array(tuple(x[1] for x in test_set))
        # test_data = lightgbm.Dataset(data, label=label)

        # Stash the data so we can reload it later to easily tweak with stuff
        self.stash_training_data(train_features, train_labels, test_features, test_labels)

        # Train lgbm model/treelite model
        self.train_from_training_data(
            train_features,
            train_labels,
            test_features,
            test_labels,
        )

    def train_from_training_data(
        self,
        train_features,
        train_labels,
        test_features,
        test_labels,
    ):
        train_data = lightgbm.Dataset(train_features, label=train_labels)
        test_data = lightgbm.Dataset(test_features, label=test_labels)

        num_round = 15000
        early_stopping_rounds = 10
        # learning_rate_fxn=lambda x: (lr - lrs) + (lrs * (lrsh ** x)),  # Start with a higher learning rate and adjust lower over time
        learning_rate = 0.15
        learning_rate_fxn = lambda x: learning_rate # noqa

        # bagging_fractions = [.05, .1, .2, .3, .4]
        # bagging_freqs = [5, 10, 20, 30]
        # num_leaves_choices = [2**7, 2**8, 2**9, 2**10, 2**11]
        bagging_fractions = [.3]
        bagging_freqs = [20]
        num_leaves_choices = [2**11]
        for (
            bagging_fraction,
            bagging_freq,
            num_leaves,
        ) in itertools.product(
            bagging_fractions,
            bagging_freqs,
            num_leaves_choices
        ):
            params = {
                'objective': 'mean_squared_error', # aka L2, rmse.  XXX Why same as root?
                'boosting': "gbdt",  # gbdt is slower but better accuracy, goss is faster (but only slightly)
                'bagging_fraction': bagging_fraction,
                'bagging_freq': bagging_freq,
                'learning_rate': learning_rate,  # This is overriden in the case where dynamic learning_rates are specified below
                'num_leaves': num_leaves,
                # 'max_depth': 3,
                'max_bin': 128,
                'metric': ["mean_squared_error", "mae"],
                'min_data_in_leaf': 10,
                'num_threads': 16,  # 0 is as many as CPUs for server
                # 'min_gain_to_split': 0.01,
                'verbose': 1,
            }

            print("\nTraining")
            lightgbm_booster = lightgbm.train(
                params,
                train_data,
                num_round,
                valid_sets=[train_data, test_data],
                learning_rates=learning_rate_fxn,
                early_stopping_rounds=early_stopping_rounds, # Stops if ANY metric in metrics doesn't improve in N rounds
            )

            print("\nTrained with following params:")
            pprint.pprint(params)

        # Save lightgbm model to disk so treelite can load it
        lightgbm_model_path = f"{settings.TMP_DIRECTORY}/lightgbm-{str(uuid4())}.model"
        lightgbm_booster.save_model(lightgbm_model_path)

        model_dict = lightgbm_booster.dump_model()
        lightgbm_model_dump_path = f"{settings.TMP_DIRECTORY}/lightgbm-{str(uuid4())}.json"
        with open(lightgbm_model_dump_path, 'w') as f:
            f.write(json.dumps(model_dict))
        print("Dumped model here:", lightgbm_model_dump_path)

        # Build treelite model
        #  - stash path in self.treelite_model_path
        self.build_treelite_model(
            lightgbm_model_path,
            annotation_samples=test_features, # XXX: Change
        )

        # Load up the just-made treelite model for use
        self.load(self.treelite_model_path)

    def predict(self, features) -> numpy.array:
        # :features ~ [features_1, features_2, ...]
        # :features ~ [(1, 0, ...), (0, 1, ...), ...]
        # XXX: change agent code to be numpy arrays?
        batch = TreeliteBatch.from_npy2d(numpy.array(features, dtype=numpy.float32))
        return self.treelite_predictor.predict(batch)
        # return self.treelite_predictor.predict(batch).item(0)

    def predict_instance(self, features):
        # This goes much slower than just passing a batch of 1 in above...
        return self.treelite_predictor.predict_instance(features)


@dataclass
class NaivePolicy:
    possible_actions: typing.Any
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
            move_probabilities = {}
            for i, action in enumerate(self.possible_actions):
                state_action = tuple(features + [i])
                move_probabilities[action] = self.state_action_mass[state_action] / self.state_action_weight[state_action]
            return move_probabilities
        except KeyError:
            # Never seen this state before; therefore, use uniform policy
            uniform_probability = 1.0 / len(allowable_actions)
            return {move: uniform_probability for move in allowable_actions}


def train_naive_models():
    from tictactoe import Environment, State, generate_features # noqa
    import pprint # noqa
    from train import generate_training_samples

    env = Environment()
    replay_directory = sys.argv[1]
    samples = list(
        generate_training_samples(
            replay_directory,
            State,
            generate_features,
            env,
        )
    )

    value_model = NaiveValue()
    value_model.train(samples)
    value_model.save("./ttt_naive_value.model")
    value_model.load("./ttt_naive_value.model")

    policy_model = NaivePolicy(env.all_possible_actions())
    policy_model.train(samples)
    policy_model.save("./ttt_naive_policy.model")
    policy_model.load("./ttt_naive_policy.model")

    # Show value/policy for first N possible moves of game
    for pos in range(9):
        s = State(tuple(1 if x == pos else 0 for x in range(9)), 1)
        features = generate_features(s, 1)
        allowable_actions = env.enumerate_actions(s)

        value = value_model.predict(features)
        move_probabilities = policy_model.predict(features, allowable_actions)
        print()
        print(env.text_display(s))
        print(value)
        pprint.pprint(move_probabilities)


def train_connect_4():
    import connect_four # noqa
    import pprint # noqa
    from train import generate_training_samples

    env_class = connect_four
    env = env_class.Environment()
    replay_directory = sys.argv[1]
    samples = list(
        generate_training_samples(
            replay_directory,
            env_class.State,
            env_class.generate_features,
            env,
        )
    )

    print("Samples:", len(samples))

    value_model = NaiveValue()
    value_model.train(samples)
    value_model.save("./c4_naive_value.model")
    value_model.load("./c4_naive_value.model")

    policy_model = NaivePolicy(env.all_possible_actions())
    policy_model.train(samples)
    policy_model.save("./c4_naive_policy.model")
    policy_model.load("./c4_naive_policy.model")

    s_initial = env.initial_state()
    allowable_actions = env.enumerate_actions(s_initial)
    for action in allowable_actions:
        s_prime = env.transition_state(s_initial, action)
        features = env_class.generate_features(s_prime, 1)
        allowable_actions = env.enumerate_actions(s_prime)

        value = value_model.predict(features)
        move_probabilities = policy_model.predict(features, allowable_actions)
        print()
        rprint(env.text_display(s_prime))
        print(value)
        pprint.pprint(move_probabilities)


def test_gbdt():
    import connect_four # noqa
    from train import generate_training_samples
    env_class = connect_four
    env = env_class.Environment()

    replay_directory = sys.argv[1]
    samples = list(
        generate_training_samples(
            replay_directory,
            env_class.State,
            env_class.generate_features,
            env,
        )
    )

    print("Samples:", len(samples))

    value_model = GBDTValue()
    value_model.train(samples)
    value_model.save('./treelite-model.dylib') # must have dylib suffix
    value_model.load('./treelite-model.dylib')


def test_treelite_predictions():
    import connect_four # noqa
    from train import generate_training_samples
    env_class = connect_four
    env = env_class.Environment()

    replay_directory = sys.argv[1]
    samples = list(
        generate_training_samples(
            replay_directory,
            env_class.State,
            env_class.generate_features,
            env,
        )
    )

    value_model = GBDTValue()
    value_model.load('./treelite-model.dylib')

    state_features = [numpy.array(x[1], dtype=numpy.float32) for x in samples if x[0] == "value"]
    state_features = numpy.array(state_features, dtype=numpy.float32)
    # state_features = state_features[:10]

    st_time = time.time()
    value_model.predict(state_features, batch_mode=True)
    elapsed = time.time() - st_time
    print("took", elapsed, "seconds")
    print("p/s", len(state_features) / elapsed)

    st_time = time.time()
    for features in state_features:
        value_model.predict(features)
    elapsed = time.time() - st_time
    print("took", elapsed, "seconds")
    print("p/s", len(state_features) / elapsed)


if __name__ == "__main__":
    # test_gbdt()
    test_treelite_predictions()
    # train_naive_models()
    # train_connect_4()
