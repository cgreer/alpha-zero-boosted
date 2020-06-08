import sys
import math
import time

from self_play import run as run_self_play
from train import run as run_model_training
from assess import run_faceoff
from training_info import TrainingInfo, setup_filesystem


def run(
    environment,
    species,
    num_batches,
    num_workers=12,
    adjusted_win_rate_threshold=0.55,
    games_per_batch=5000,
):

    training_info = TrainingInfo.load(
        environment,
        species,
    )
    last_batch = training_info.current_batch + num_batches - 1

    num_faceoff_rounds = math.ceil(150 / num_workers) # Will play at least num_workers per round

    for _ in range(num_batches):
        print(f"\n\nBatch {training_info.current_batch} / {last_batch}")
        print(f"environment: {environment}, species: {species}")
        training_generation = training_info.self_play_bot_generation + 1

        # Ensure directories are made/etc.
        # - Not sure this actually depends on generation, but maybe it will later.
        setup_filesystem(
            environment,
            species,
            training_info.self_play_bot_generation,
        )

        # Self play another batch
        print("\n\nSelf Play")
        st_time = time.time()
        run_self_play(
            environment,
            species,
            training_info.self_play_bot_generation,
            games_per_batch,
            training_info.current_batch,
            num_workers,
        )
        elapsed = round(time.time() - st_time, 1)
        games_per_sec = round(games_per_batch / elapsed, 1)
        print(f"\nSelf play finished in {elapsed} seconds")
        print(f"Games played: {games_per_batch}, games per sec: {games_per_sec}")

        # Train new model
        print("\n\nTraining")
        st_time = time.time()
        run_model_training(
            environment,
            species,
            training_generation,
            training_info.current_batch,
            num_workers,
        )
        elapsed = round(time.time() - st_time, 1)
        print(f"\nTrained new models in {elapsed} seconds")

        # Assess new model
        print("\n\nAssessing")
        st_time = time.time()
        contender_matchup_info = run_faceoff(
            environment,
            species,
            training_generation,
            num_rounds=num_faceoff_rounds,
            num_workers=num_workers,
        )
        elapsed = round(time.time() - st_time, 1)
        print(f"\nAssessed new model in {elapsed} seconds")

        adjusted_win_rate = contender_matchup_info.win_rate(draw_weight=0.5)
        print("Adjusted Win Rate:", adjusted_win_rate)
        if adjusted_win_rate >= adjusted_win_rate_threshold:
            training_info.self_play_bot_generation += 1
            print("FOUND NEW BOT:", training_info.self_play_bot_generation)
        training_info.current_batch += 1
        training_info.save()


if __name__ == "__main__":
    environment, species, num_batches = sys.argv[1:]
    run(environment, species, int(num_batches))
