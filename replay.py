import sys
import json
import time
import os

from rich import print as rprint

from environment_registry import get_env_module


def replay_video(replay_path, speed):
    replay = json.loads(open(replay_path, 'r').read())
    environment_name = replay["name"]
    moves = []
    for move_info in replay["replay"]:
        moves.append(move_info["move"])

    while True:
        env_module = get_env_module(environment_name)
        environment = env_module.Environment()
        current_state = environment.initial_state()
        for move in moves:
            # Show state
            os.system("clear")
            rprint(environment.text_display(current_state))
            time.sleep(speed)

            if move is None:
                break

            current_state = environment.transition_state(current_state, move)


if __name__ == "__main__":
    replay_video(sys.argv[1], float(sys.argv[2]))
