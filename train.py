import os
import json
from environment_registry import get_env_module
from paths import build_model_paths, find_batch_directory
from intuition_model import NaiveValue, NaivePolicy, GBDTValue


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
        for agent_num in game_agent_nums:
            features = feature_extractor(state, agent_num)
            value_sample_label = outcome[agent_num]
            yield "value", game_bucket, features, value_sample_label

            # No terminal moves...
            if (agent_num == this_agent_num) and move["move"] is not None:
                policy_labels = extract_policy_labels(move, environment)
                yield "policy", game_bucket, features, policy_labels


def generate_training_samples(
    replay_directory,
    state_class,
    feature_extractor,
    environment,
):
    games_parsed = -1
    for file_name in os.listdir(replay_directory):
        games_parsed += 1
        if (games_parsed % 2000) == 0:
            print("Replays Parsed:", games_parsed)

        file_path = os.path.join(replay_directory, file_name)
        agent_replay = json.loads(open(file_path, 'r').read())
        for sample_type, game_bucket, features, label in samples_from_replay(
            agent_replay,
            feature_extractor,
            state_class,
            environment,
        ):
            yield sample_type, game_bucket, features, label


def run(
    environment,
    bot_species,
    bot_generation,
    head_batch_num,
    max_games=500_000,
    max_generational_lookback=10,
):

    env_class = get_env_module(environment)
    env = env_class.Environment() # XXX: Refactor to get rid of this.

    ####################
    # Collect Samples
    ####################
    samples = []
    for batch_num in range(head_batch_num, 0, -1):
        replay_directory = find_batch_directory(environment, bot_species, batch_num)
        print("Collecting Samples", replay_directory)

        # XXX: Implement game capping
        # num_games = parse_batch_directory # XXX: Implement game capping
        # XXX: Implement generation lookback capping
        for sample in generate_training_samples(
            replay_directory,
            env_class.State,
            env_class.generate_features,
            env,
        ):
            samples.append(sample)

    ####################
    # Train Model
    ####################

    value_model_path, policy_model_path = build_model_paths(
        environment,
        bot_species,
        bot_generation,
    )

    # XXX: Abstract
    if bot_species == "mcts_naive":
        value_model = NaiveValue()
        # value_model = GBDTValue()
        value_model.train(samples)
        value_model.save(value_model_path)
        print("Saved value model to", value_model_path)

        policy_model = NaivePolicy(env.all_possible_actions())
        policy_model.train(samples)
        policy_model.save(policy_model_path)
        print("Saved policy model to", policy_model_path)
    elif bot_species == "mcts_gbdt":
        value_model = GBDTValue()
        value_model.train(samples)
        value_model.save(value_model_path)
        print("Saved value model to", value_model_path)

        # policy_model = NaivePolicy(env.all_possible_actions())
        # policy_model.train(samples)
        # policy_model.save(policy_model_path)
        # print("Saved policy model to", policy_model_path)
    else:
        raise KeyError(f"Unknown species: {bot_species}")


if __name__ == "__main__":
    ENVIRONMENT = "connect_four"
    BOT_SPECIES = "mcts_gbdt"
    BOT_GENERATION = 7 + 1 # XXX Highest + 1
    HEAD_BATCH_NUM = 8 # Highest
    MAX_GAMES = 500_000
    MAX_GENERATIONAL_LOOKBACK = 10

    run(
        ENVIRONMENT,
        BOT_SPECIES,
        BOT_GENERATION,
        HEAD_BATCH_NUM,
        MAX_GAMES,
        MAX_GENERATIONAL_LOOKBACK,
    )
