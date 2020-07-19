from dataclasses import dataclass
import os
from hashlib import md5
from uuid import uuid4

import numpy

from agent_replay import AgentReplay
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
            if data is None:
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


def split_train_test(samples: SampleData, test_fraction):
    '''
    Partition the games into training and test games. Positions from a game that
    are in the training set should not be in the test set.

    If the positions from the same game are in both the training and test sets,
    then it'll be easy for a model to overfit without early stopping catching it
    by memorizing non-generalizable aspects of a particular game.

    :test_fraction in range [0.0, 1.0]
    '''
    meta_info = samples.meta_info
    features = samples.features
    labels = samples.labels
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


def extract_policy_labels(position, environment):
    '''
    The mcts policy labels are a vector of (move visit_count / total visits) for
    the node that the mcts ran for.
    '''
    total_visits = position.edge_visits()
    actions_considered = position.actions_considered
    policy_labels = []
    for action_id in environment.all_possible_actions():
        ac = actions_considered.get(action_id)
        action_edge_visits = ac.visit_count if ac else 0.0
        label = action_edge_visits / total_visits
        policy_labels.append(label)
    return policy_labels


def calculate_game_bucket(game_id_guid):
    '''
    :game_id_guid ~ c0620bd2-30ba-463f-b1e1-093ecd1f2f3e
    :game_id_guid ~ XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX
    :game_id_guid ~ A-B-C-D-E

    Take the last 7 hex digits and convert to int.  That'll create a space of
    268,435,455 possible buckets.  Should be enough...
    '''
    return int(game_id_guid[-7:], 16)


def is_trainable_value(
    is_terminal_position,
    num_visits,
    full_search_steps,
    partial_search_steps,
    require_full_steps=True,
):
    # Terminal positions are valuable even though there are no mcts
    # considerations
    if is_terminal_position:
        return True

    # Don't take positions that have no considerations. An example of these are
    # the positions that were reconstructed before playing more games from a
    # position.
    # XXX: Test. It might be advantageous to take these too.
    if num_visits < partial_search_steps:
        return False

    # Only take positions if a full consideration was performed.
    # - Unlike alpha zero, this implementation has decoupled value/policy
    # models. I think the KataGo implementation had to only take the samples
    # that had full mcts considerations because the value sample was coupled to
    # the policy sample, but it might be advantageous to take all the value
    # samples when not under that constraint.
    # XXX: Test. It might be advantageous to take these too.
    if require_full_steps:
        if num_visits < full_search_steps:
            return False

    return True


def is_trainable_policy(
    is_terminal_position,
    num_visits,
    full_search_steps,
):
    # Terminal positions have no actions...
    if is_terminal_position:
        return False

    # Policy target is useless without a full consideration.
    if num_visits < full_search_steps:
        return False

    return True


def samples_from_replay(
    agent_replay,
    feature_extractor,
    environment,
):
    game_bucket = calculate_game_bucket(agent_replay.game_id)
    outcomes = agent_replay.outcomes
    this_agent_num = agent_replay.agent_settings.agent_num
    agent_generation = agent_replay.agent_settings.generation
    game_agent_nums = agent_replay.agent_nums
    full_search_steps = agent_replay.agent_settings.full_search_steps
    partial_search_steps = agent_replay.agent_settings.partial_search_steps
    require_full_steps = agent_replay.agent_settings.require_full_steps
    for position in agent_replay.positions:

        # Only use the moves that this agent played.
        # - We only have policies for these moves
        #   - Only these moves were deeply considered
        # XXX: Ok to not do this? Better way?
        if position.state.whose_move != this_agent_num:
            continue

        # Position attributes
        # - Early stopped positions do *not* count as terminal.  The last
        #   (state, action) pair for them will have a valid action.
        num_visits = position.edge_visits()
        is_terminal_position = position.is_terminal()

        # Check if sample is trainable
        value_trainable = is_trainable_value(
            is_terminal_position,
            num_visits,
            full_search_steps,
            partial_search_steps,
            require_full_steps=require_full_steps,
        )
        policy_trainable = is_trainable_policy(
            is_terminal_position,
            num_visits,
            full_search_steps,
        )

        # Short circuit: Don't even extract features if we know there's no
        # samples to be had.
        if (not value_trainable) and (not policy_trainable):
            continue

        # Generate samples
        # - Generate N value samples, one from each agent's POV
        # - Generate 1 policy sample from this agent's pov
        # - Other agents' replays will cover other (s, a) policy samples not covered from
        #   this agent's moves.
        #
        # How should we handle terminal states for value?
        #   - Technically you should learn the pattern of terminal states and
        #     understand their value...  Might be good for them to bend parameters
        #     to detect them.
        #   - XXX: Double check that this is the right thing to do.
        features = feature_extractor(position.state, game_agent_nums)
        for i, agent_num in enumerate(game_agent_nums):
            agent_features = features[i]
            value_sample_label = outcomes[agent_num]
            meta_info = [game_bucket, agent_generation]

            # Value sample
            if value_trainable:
                yield "value", meta_info, agent_features, value_sample_label

            # Policy sample
            if agent_num != this_agent_num:
                continue
            if not policy_trainable:
                continue
            policy_labels = extract_policy_labels(position, environment)
            yield "policy", meta_info, agent_features, policy_labels


# XXX: Move these to hashring.py or something
def fast_deterministic_hash(string):
    return int(md5(string.encode()).hexdigest(), 16)


def is_my_task(key, worker_num, num_workers):
    return (fast_deterministic_hash(key) % num_workers) == worker_num


def iter_replay_data(
    replay_directory,
    StateClass,
    worker_num=0,
    num_workers=1,
):
    for file_name in os.listdir(replay_directory):
        if not file_name.endswith(".json"):
            continue
        file_path = os.path.join(replay_directory, file_name)

        if not is_my_task(file_path, worker_num, num_workers):
            continue

        try:
            agent_replay = AgentReplay.from_path(file_path, StateClass)
        except Exception as e:
            print(f"Exception JSON decoding Replay: {file_name}", e)
            continue
        yield agent_replay


def generate_training_samples(
    replay_directory,
    StateClass,
    feature_extractor,
    environment,
    worker_num=0,
    num_workers=1,
):
    replays_parsed = 0
    for agent_replay in iter_replay_data(
        replay_directory,
        StateClass,
        worker_num,
        num_workers,
    ):
        if (replays_parsed % 100) == 0:
            print("Replays Parsed:", replays_parsed)
        for sample_type, game_bucket, features, labels in samples_from_replay(
            agent_replay,
            feature_extractor,
            environment,
        ):
            yield sample_type, game_bucket, features, labels
        replays_parsed += 1
