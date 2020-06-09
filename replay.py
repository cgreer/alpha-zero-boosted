import sys
import json
import random
import time
import os

from rich import print as rprint

from environment_registry import get_env_module
from paths import find_batch_directory


def replay_video(replay_path, speed, first_n_moves=100_000_000):
    replay = json.loads(open(replay_path, 'r').read())
    environment_name = replay["name"]
    moves = []
    for move_info in replay["replay"]:
        moves.append(move_info["move"])

    env_module = get_env_module(environment_name)
    environment = env_module.Environment()
    current_state = environment.initial_state()
    for i, move in enumerate(moves):
        if i >= first_n_moves:
            break
        # Show state
        os.system("clear")
        rprint(environment.text_display(current_state))
        time.sleep(speed)

        if move is None:
            break

        current_state = environment.transition_state(current_state, move)


def sample_batch_replay_files(
    environment,
    species,
    batch,
):
    replay_directory = find_batch_directory(environment, species, batch)

    # Grabbing replays
    print("Grabbing replays")
    all_replays = []
    for file_name in os.listdir(replay_directory):
        if not file_name.endswith(".json"):
            continue
        file_path = os.path.join(replay_directory, file_name)
        all_replays.append(file_path)

    print("Shuffling")
    random.shuffle(all_replays)

    return all_replays


def sample_batch_replays(
    environment,
    species,
    batch,
    speed=.3,
    first_n_moves=1_000,
):
    replay_directory = find_batch_directory(environment, species, batch)

    # Grabbing replays
    print("Grabbing replays")
    all_replays = []
    for file_name in os.listdir(replay_directory):
        if not file_name.endswith(".json"):
            continue
        file_path = os.path.join(replay_directory, file_name)
        all_replays.append(file_path)

    print("Shuffling")
    random.shuffle(all_replays)

    for replay_path in all_replays:
        replay_video(replay_path, speed, first_n_moves)


if __name__ == "__main__":
    # replay_video(sys.argv[1], float(sys.argv[2]))
    sample_batch_replays(
        sys.argv[1],
        sys.argv[2],
        int(sys.argv[3]),
        float(sys.argv[4]),
    )
