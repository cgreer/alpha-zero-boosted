import time
from agents import MCTSAgent
import intuition_model
from environment_registry import get_env_module
from paths import build_model_paths, build_replay_directory
from multiprocessing import Pool
from quoridor import BootstrapValue


def configure_bot(environment_name, species, generation):
    env_module = get_env_module(environment_name)

    if generation == "HIGHEST":
        # XXX: Lookup highest current generation
        pass

    if generation == 1:
        # XXX: Setup per-environment bootstrapping models here
        value_model = intuition_model.UnopinionatedValue()
        if environment_name == "quoridor":
            value_model = BootstrapValue()
        policy_model = intuition_model.UniformPolicy()
    else:
        if species == "mcts_naive":
            value_model_path, policy_model_path = build_model_paths(environment_name, species, generation)

            value_model = intuition_model.NaiveValue()
            value_model.load(value_model_path)

            # XXX: pass in env, fix all_possible_actions nonsense
            policy_model = intuition_model.NaivePolicy()
            policy_model.load(policy_model_path)
        elif species == "mcts_gbdt":
            value_model_path, policy_model_path = build_model_paths(
                environment_name,
                species,
                generation,
            )
            value_model = intuition_model.GBDTValue()
            value_model.load(value_model_path)

            policy_model = intuition_model.GBDTPolicy()
            policy_model.load(policy_model_path)
        else:
            raise KeyError(f"Unknown species: {species}")

    return dict(
        game_tree=None,
        current_node=None,
        feature_extractor=env_module.generate_features,
        value_model=value_model,
        policy_model=policy_model,
        move_consideration_steps=1,
        move_consideration_time=0.4,
        puct_explore_factor=1.0,
        # puct_noise_alpha=0.2, XXX: Configure per game (like for connect_four)
        puct_noise_alpha=0.1,
        puct_noise_influence=0.25,
    )


def play_game(
    environment_name,
    agent_settings,
    replay_directory=None,
):
    env_module = get_env_module(environment_name)
    environment = env_module.Environment()

    mcts_agent_1 = MCTSAgent(environment=environment, **agent_settings)
    mcts_agent_2 = MCTSAgent(environment=environment, **agent_settings)

    # Play
    environment.add_agent(mcts_agent_1)
    environment.add_agent(mcts_agent_2)
    _, was_early_stopped = environment.run()

    mcts_agent_1.record_replay(replay_directory, was_early_stopped)
    mcts_agent_2.record_replay(replay_directory, was_early_stopped)


def run_worker(args):
    environment, bot_species, bot_generation, num_games, batch = args

    agent_settings = configure_bot(environment, bot_species, bot_generation)
    replay_directory = build_replay_directory(environment, bot_species, bot_generation, batch)
    print(f"Self playing, bot: {bot_species}-{bot_generation}, batch: {replay_directory}")

    total_elapsed = 0.0
    for i in range(num_games):
        st_time = time.time()
        try:
            play_game(environment, agent_settings, replay_directory)
        except Exception as e:
            print("GAME FAILED:", e)
        elapsed = time.time() - st_time
        total_elapsed += elapsed
        if i % 20 == 0:
            print(f"GAME {i:05d}: {round(elapsed, 2)} seconds, AVERAGE: {round(total_elapsed / (i + 1), 2)} seconds")

    return batch, num_games


def run(
    environment,
    bot_species,
    bot_generation,
    num_games,
    batch,
    num_workers,
):
    num_worker_games = num_games // num_workers # 16 * 625 = 10K
    results = []
    with Pool(num_workers) as p:
        worker_args = [(environment, bot_species, bot_generation, num_worker_games, batch) for _ in range(num_workers)]
        results = p.map(run_worker, worker_args)

    total_games = 0
    for i, result in enumerate(results):
        print(f"Worker {i}, batch: {result[0]}, games: {result[1]}")
        total_games += result[1]

    return total_games


if __name__ == "__main__":
    # ENVIRONMENT = "connect_four"
    ENVIRONMENT = "quoridor"
    BOT_SPECIES = "mcts_naive"
    BOT_GENERATION = 1 # {"HIGHEST", int}
    BATCH = 1 # Highest + 1 (1 if first batch)
    NUM_WORKERS = 10
    NUM_GAMES = 1000
    # NUM_GAMES = 5 * NUM_WORKERS

    run(ENVIRONMENT, BOT_SPECIES, BOT_GENERATION, NUM_GAMES, BATCH, NUM_WORKERS)
