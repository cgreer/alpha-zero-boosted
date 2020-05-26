from environment_registry import get_env_module
from paths import build_model_paths, find_batch_directory
from intuition_model import NaiveValue, NaivePolicy, GBDTValue, GBDTPolicy
from training_samples import generate_training_samples


def run(
    environment,
    bot_species,
    bot_generation,
    head_batch_num,
    max_games=500_000,
    max_generational_lookback=10,
    positions_per_batch=1_000_000_000,
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
        for i, sample in enumerate(
            generate_training_samples(
                replay_directory,
                env_class.State,
                env_class.generate_features,
                env,
            )
        ):
            if i >= (positions_per_batch - 1):
                break
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
    POSITIONS_PER_BATCH = 100 * 64_000 * 10 # moves/game * max games/batch * safety multiplier
    run(
        ENVIRONMENT,
        BOT_SPECIES,
        BOT_GENERATION,
        HEAD_BATCH_NUM,
        MAX_GAMES,
        MAX_GENERATIONAL_LOOKBACK,
        POSITIONS_PER_BATCH,
    )
