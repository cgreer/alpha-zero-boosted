import os
import json


def split_train_test(samples, test_fraction, only_of_type):
    '''
    Partition the games into training and test games. Positions from a game that are
    in the training set should not be in the test set.

    If the positions from the same game are in both the training and test sets, then
    it'll be easy for a model to overfit without early stopping catching it by memorizing
    non-generalizable aspects of a particular game.

    :test_fraction ~ [0.0, 1.0]
    '''
    # Convert test fraction to percentage from 0-100
    rounded_test_percentage = int(test_fraction * 100) # off by a bit cuz rounding... should be fine
    train_set = []
    test_set = []
    for sample_type, game_bucket, features, label in samples:
        if sample_type != only_of_type:
            continue
        which_set = test_set if (game_bucket % 100) <= rounded_test_percentage else train_set
        which_set.append((features, label))
    return train_set, test_set


def iter_samples(env_class, replay_directory):
    env = env_class.Environment() # XXX: Get rid of this

    for sample in generate_training_samples(
        replay_directory,
        env_class.State,
        env_class.generate_features,
        env,
    ):
        yield sample


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
            yield "value", game_bucket, agent_features, value_sample_label

            # No terminal moves...
            if (agent_num == this_agent_num) and move["move"] is not None:
                policy_labels = extract_policy_labels(move, environment)
                yield "policy", game_bucket, agent_features, policy_labels


def iter_replay_data(replay_directory):
    for file_name in os.listdir(replay_directory):
        file_path = os.path.join(replay_directory, file_name)
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
):
    games_parsed = -1
    for agent_replay in iter_replay_data(replay_directory):
        games_parsed += 1
        if (games_parsed % 1000) == 0:
            print("Replays Parsed:", games_parsed)
        for sample_type, game_bucket, features, label in samples_from_replay(
            agent_replay,
            feature_extractor,
            state_class,
            environment,
        ):
            yield sample_type, game_bucket, features, label
