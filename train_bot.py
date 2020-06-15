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
    adjusted_win_rate_threshold=0.51,
    games_per_batch=3000,
    num_assessment_games=200,
):
    num_faceoff_rounds = math.ceil(num_assessment_games / num_workers) # Will play at least num_workers per round

    training_info = TrainingInfo.load(environment, species)
    final_training_batch = len(training_info.batches) + num_batches
    for _ in range(num_batches):
        current_batch = len(training_info.batches) + 1
        generation_self_play = training_info.current_self_play_generation()
        generation_training = generation_self_play + 1

        print(f"\n\nBatch {current_batch} / {final_training_batch}")
        print(f"environment: {environment}, species: {species}")
        print(f"self-play generation: {generation_self_play}")

        # Ensure directories are made/etc.
        # - Not sure this actually depends on generation, but maybe it will later.
        setup_filesystem(
            environment,
            species,
            generation_self_play,
        )

        # Self play another batch
        print("\n\nSelf Play")
        self_play_start_time = time.time()
        run_self_play(
            environment,
            species,
            generation_self_play,
            games_per_batch,
            current_batch,
            num_workers,
        )
        self_play_end_time = time.time()
        elapsed = round(self_play_end_time - self_play_start_time, 1)
        games_per_sec = round(games_per_batch / elapsed, 1)
        print(f"\nSelf play finished in {elapsed} seconds")
        print(f"Games played: {games_per_batch}, games per sec: {games_per_sec}")

        # Train new model
        print("\n\nTraining")
        training_start_time = time.time()
        run_model_training(
            environment,
            species,
            generation_training,
            current_batch,
            num_workers,
        )
        training_end_time = time.time()
        elapsed = round(training_end_time - training_start_time, 1)
        print(f"\nTrained new models in {elapsed} seconds")

        # Assess new model
        print("\n\nAssessing")
        assessment_start_time = time.time()
        contender_matchup_info = run_faceoff(
            environment,
            species,
            generation_training,
            num_rounds=num_faceoff_rounds,
            num_workers=num_workers,
        )
        assessment_end_time = time.time()
        elapsed = round(assessment_end_time - assessment_start_time, 1)
        print(f"\nAssessed new model in {elapsed} seconds")

        adjusted_win_rate = contender_matchup_info.win_rate(draw_weight=0.5)
        print("Adjusted Win Rate:", round(adjusted_win_rate, 3))
        generation_trained = None
        if adjusted_win_rate >= adjusted_win_rate_threshold:
            generation_trained = generation_training
            print("FOUND NEW BOT:", generation_trained)

        training_info.finalize_batch(
            self_play_start_time=self_play_start_time,
            self_play_end_time=self_play_end_time,
            training_start_time=training_start_time,
            training_end_time=training_end_time,
            assessment_start_time=assessment_start_time,
            assessment_end_time=assessment_end_time,
            generation_self_play=generation_self_play,
            generation_trained=generation_trained,
            assessed_awr=adjusted_win_rate,
        )


if __name__ == "__main__":
    environment, species, num_batches = sys.argv[1:]
    run(environment, species, int(num_batches))
    '''
    environment = "connect_four"
    for _ in range(10):
        run(environment, "gbdt_pcr", 1)
        run(environment, "gbdt_pcr_v", 1)
    '''
