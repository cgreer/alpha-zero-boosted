from multiprocessing import Pool
import time
import traceback

from species import get_species
from environment_registry import get_env_module
from paths import build_replay_directory
from surprise import find_surprises


def play_game(
    env_module,
    Agent,
    agent_settings,
    replay_directory=None,
    reconstruction_info=None,
):
    # Play a game
    environment = env_module.Environment()

    agent_1 = Agent(environment=environment, **agent_settings)
    agent_2 = Agent(environment=environment, **agent_settings)

    environment.add_agent(agent_1)
    environment.add_agent(agent_2)

    environment.setup()
    if reconstruction_info:
        environment.reconstruct_position(*reconstruction_info)
    environment.run()

    # Record game replay
    _, agent_1_replay = agent_1.record_replay(replay_directory)
    _, agent_2_replay = agent_2.record_replay(replay_directory)

    return (agent_1_replay, agent_2_replay)


def self_play_cycle(
    environment_name,
    Agent,
    agent_settings,
    replay_directory,
):
    env_module = get_env_module(environment_name)

    # Play a full game
    agent_replays = play_game(
        env_module,
        Agent,
        agent_settings,
        replay_directory=replay_directory,
    )

    # Replay more games from certain positions (if enabled)
    if not agent_settings.get("revisit_violated_expectations", False):
        return

    # Setup revisit settings
    # XXX: Tune
    # agent_settings["full_search_proportion"] = 1.0
    # agent_settings["temperature"] = 1.0
    num_revisits = 10
    raw_error_range = [-2.0, -0.50]
    upstream_turns = 1

    # Run revisits
    for agent_replay in agent_replays:
        # Get the position with a highest expectation violation, above a certain
        # threshold.
        surprises = find_surprises(
            agent_replay=agent_replay,
            raw_error_range=raw_error_range,
        )
        if not surprises:
            continue

        # Play :num_revisits games from a few turns upstream of that position.
        initial_index = max(surprises[0].initial_position_index - upstream_turns, 0)
        reconstruction_info = (agent_replay, agent_replay.positions[initial_index])
        for _ in range(num_revisits):
            play_game(
                env_module,
                Agent,
                agent_settings,
                replay_directory=replay_directory,
                reconstruction_info=reconstruction_info,
            )


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
            self_play_cycle(environment, Agent, agent_settings, replay_directory)
            # play_game(environment, Agent, agent_settings, replay_directory)
        except Exception as e:
            print("GAME FAILED:", e)
            traceback.print_exc()
        elapsed = time.time() - st_time
        total_elapsed += elapsed
        if i % 10 == 0:
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
