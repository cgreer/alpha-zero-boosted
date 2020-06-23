import time
from species import get_species
from environment_registry import get_env_module
from paths import build_replay_directory


def play_game(
    environment_name,
    Agent,
    agent_settings,
    replay_directory=None,
):
    env_module = get_env_module(environment_name)
    environment = env_module.Environment()

    mcts_agent_1 = Agent(environment=environment, **agent_settings)
    mcts_agent_2 = Agent(environment=environment, **agent_settings)

    # Play
    environment.add_agent(mcts_agent_1)
    environment.add_agent(mcts_agent_2)
    _, was_early_stopped = environment.run()

    mcts_agent_1.record_replay(replay_directory, was_early_stopped)
    mcts_agent_2.record_replay(replay_directory, was_early_stopped)


def run_worker(args):
    environment, species, generation, num_games, batch = args

    sp = get_species(species)
    Agent = sp.AgentClass
    agent_settings = sp.agent_settings(environment, generation, play_setting="self_play")

    replay_directory = build_replay_directory(environment, species, generation, batch)
    print(f"Self playing, bot: {species}-{generation}, batch: {replay_directory}")

    total_elapsed = 0.0
    for i in range(num_games):
        st_time = time.time()
        try:
            play_game(environment, Agent, agent_settings, replay_directory)
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
