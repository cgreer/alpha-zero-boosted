from dataclasses import dataclass
import os
import json
from hashlib import md5
from uuid import uuid4

import numpy

import settings


@dataclass
class SampleData:
    features: numpy.array
    labels: numpy.array
    meta_info: numpy.array = None
    weights: numpy.array = None

    def stash_data(self):
        base_path = f"{settings.TMP_DIRECTORY}/model-{str(uuid4())}_"

        stashables = [
            ("features", self.features),
            ("labels", self.labels),
            ("meta_info", self.meta_info),
            ("weights", self.weights),
        ]
        for dtype, data in stashables:
            if not data:
                continue
            stash_path = f"{base_path}{dtype}.npy"
            numpy.save(stash_path, data)
            print(f"stashed: {stash_path}")

    @classmethod
    def from_stashed_data(cls, base_path):
        stash_path = f"{base_path}features.npy"
        features = numpy.load(stash_path)

        stash_path = f"{base_path}labels.npy"
        labels = numpy.load(stash_path)

        stash_path = f"{base_path}meta_info.npy"
        meta_info = None
        if os.path.exists(stash_path):
            meta_info = numpy.load(stash_path)

        stash_path = f"{base_path}weights.npy"
        weights = None
        if os.path.exists(stash_path):
            weights = numpy.load(stash_path)

        return cls(features, labels, meta_info, weights)


def split_train_test(game_samples, test_fraction):
    '''
    Partition the games into training and test games. Positions from a game that are
    in the training set should not be in the test set.

    If the positions from the same game are in both the training and test sets, then
    it'll be easy for a model to overfit without early stopping catching it by memorizing
    non-generalizable aspects of a particular game.

    :test_fraction in range [0.0, 1.0]
    '''
    meta_info = game_samples["meta"]
    features = game_samples["features"]
    labels = game_samples["labels"]
    assert meta_info.shape[0] == features.shape[0] == labels.shape[0]

    # Convert test fraction to percentage from 0-100
    rounded_test_percentage = int(test_fraction * 100) # off by a bit cuz rounding... should be fine

    # Decide which indices belong train (0) or test (1)
    test_set_indices = []
    for i in range(meta_info.shape[0]):
        game_bucket = meta_info[i][0]
        is_in_test = 1 if (int(game_bucket) % 100) <= rounded_test_percentage else 0
        test_set_indices.append(is_in_test)

    # Filter into respective sets
    # - Convert test_set_indices to numpy array so we can do boolean operations.
    test_set_indices = numpy.array(test_set_indices)

    training_sample_data = SampleData(
        features=features[test_set_indices == 0],
        labels=labels[test_set_indices == 0],
        meta_info=meta_info[test_set_indices == 0],
    )

    test_sample_data = SampleData(
        features=features[test_set_indices == 1],
        labels=labels[test_set_indices == 1],
        meta_info=meta_info[test_set_indices == 1],
    )

    return training_sample_data, test_sample_data


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


def calculate_game_bucket(game_id_guid):
    '''
    :game_id_guid ~ c0620bd2-30ba-463f-b1e1-093ecd1f2f3e
    :game_id_guid ~ XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX
    :game_id_guid ~ A-B-C-D-E

    Take the last 7 hex digits and convert to int.  That'll create a space of 268,435,455 possible
    buckets.  Should be enough...
    '''
    return int(game_id_guid[-7:], 16)


def samples_from_replay(
    agent_replay,
    feature_extractor,
    state_class,
    environment,
):
    game_bucket = calculate_game_bucket(agent_replay["id"])
    outcome = agent_replay["outcome"]
    this_agent_num = agent_replay["agent"]["agent_num"]
    agent_generation = agent_replay["agent"]["generation"]
    game_agent_nums = agent_replay["agent_nums"]
    for move in agent_replay["replay"]:
        state = state_class.unmarshall(move["state"])

        # Only use the moves that this agent played.
        # - We only have policies for these moves
        #   - Only these moves were deeply considered
        # XXX: Ok to not do this? Better way?
        if state.whose_move != this_agent_num:
            continue

        # Generate samples
        # - Generate N value samples, one from each agent's POV
        # - Generate 1 policy sample from this agent's pov
        # - Other agents' replays will cover other (s, a) policy samples not covered from
        #   this agent's moves.
        # XXX How to handle terminal states for value?
        #   - Technically you should learn the pattern of terminal states and understand their
        #     value...  Might be good for them to bend parameters to detect them.
        features = feature_extractor(state, game_agent_nums)
        for i, agent_num in enumerate(game_agent_nums):
            agent_features = features[i]
            value_sample_label = outcome[agent_num]
            meta_info = [game_bucket, agent_generation]
            yield "value", meta_info, agent_features, value_sample_label

            # No terminal moves...
            if (agent_num == this_agent_num) and move["move"] is not None:
                policy_labels = extract_policy_labels(move, environment)
                yield "policy", meta_info, agent_features, policy_labels


# XXX: Move these to hashring.py or something
def fast_deterministic_hash(string):
    return int(md5(string.encode()).hexdigest(), 16)


def is_my_task(key, worker_num, num_workers):
    return (fast_deterministic_hash(key) % num_workers) == worker_num


def iter_replay_data(replay_directory, worker_num=0, num_workers=1):
    for file_name in os.listdir(replay_directory):
        if not file_name.endswith(".json"):
            continue
        file_path = os.path.join(replay_directory, file_name)

        if not is_my_task(file_path, worker_num, num_workers):
            continue

        try:
            agent_replay = json.loads(open(file_path, 'r').read())
        except Exception as e:
            print(f"Exception JSON decoding Replay: {file_name}", e)
            continue
        yield agent_replay


def generate_training_samples(
    replay_directory,
    state_class,
    feature_extractor,
    environment,
    worker_num=0,
    num_workers=1,
):
    replays_parsed = -1
    for agent_replay in iter_replay_data(replay_directory, worker_num, num_workers):
        replays_parsed += 1
        if (replays_parsed % 100) == 0:
            print("Replays Parsed:", replays_parsed)
        for sample_type, game_bucket, features, labels in samples_from_replay(
            agent_replay,
            feature_extractor,
            state_class,
            environment,
        ):
            yield sample_type, game_bucket, features, labels
