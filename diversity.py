import sys
from collections import defaultdict

from environment_registry import get_env_module
from train import find_batch_directory
from training_samples import iter_replay_data


def retrieve_move_lists(environment, species, batch_num):
    env_module = get_env_module(environment)
    replay_directory = find_batch_directory(environment, species, batch_num)

    print("\nGetting move lists")
    games = []
    i = 1
    seen_games = set()
    for replay in iter_replay_data(replay_directory, env_module.State):
        if i % 500 == 0:
            print(f"{i} replays parsed")
        i += 1

        if replay.game_id in seen_games:
            continue
        seen_games.add(replay.game_id)

        moves = []
        for position in replay.positions:
            moves.append(position.chosen_action_id)
        games.append(moves)
    return games


def display_diversity(games, max_moves=100):
    print(f"\n{'NTH MOVE':<30}{'UNIQ PCT':<30}{'TOTAL GAMES':<30}")
    for nth_move in list(range(1, 6)) + list(range(10, max_moves + 1, 10)):
        unique_game_counts = defaultdict(int)
        total_games = 0
        for moves in games:
            total_games += 1
            game_up_to_n = tuple(move for move in moves[:nth_move])
            unique_game_counts[game_up_to_n] += 1
        unique_fraction = len(unique_game_counts) / total_games
        unique_percent = round(unique_fraction * 100, 2)
        print(f"{nth_move:<30}{unique_percent:<30}{total_games:<30}")


def display_batch_diversity(environment, species, batch_num):
    move_lists = retrieve_move_lists(environment, species, batch_num)
    display_diversity(move_lists)


if __name__ == "__main__":
    # <environment> <species> <batch_num>
    display_batch_diversity(sys.argv[1], sys.argv[2], int(sys.argv[3]))
