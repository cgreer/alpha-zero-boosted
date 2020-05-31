import math
import time

from self_play import run as run_self_play
from train import run as run_model_training
from assess import run_faceoff
from training_info import TrainingInfo, setup_filesystem

environment = "quoridor"
bot_species = "mcts_gbdt"
batches_to_train = 1

training_info = TrainingInfo.load(
    environment,
    bot_species,
)

num_workers = 12

games_per_batch = 5_000
# games_per_batch = num_workers * 10

num_faceoff_rounds = math.ceil(150 / num_workers) # Will play at least num_workers per round
adjusted_win_rate_threshold = .55

for i in range(batches_to_train):
    print(f"\n\nBatch {training_info.current_batch + i} / {training_info.current_batch + batches_to_train - 1}")
    print(f"environment: {environment}, species: {bot_species}")
    training_generation = training_info.self_play_bot_generation + 1

    # Ensure directories are made/etc.
    # - XXX: Not sure this actually depends on generation, but maybe it will later.
    setup_filesystem(
        environment,
        bot_species,
        training_info.self_play_bot_generation,
    )

    # Self play another batch
    print("\n\nSelf Play")
    st_time = time.time()
    run_self_play(
        environment,
        bot_species,
        training_info.self_play_bot_generation,
        games_per_batch,
        training_info.current_batch,
        num_workers,
    )
    elapsed = round(time.time() - st_time, 1)
    games_per_sec = round(games_per_batch / elapsed, 1)
    print(f"\nGames played: {games_per_batch}, elapsed: {elapsed}, games per sec: {games_per_sec}")

    # Train new model
    print("\n\nTraining")
    st_time = time.time()
    run_model_training(
        environment,
        bot_species,
        training_generation,
        training_info.current_batch,
        num_workers,
    )
    elapsed = round(time.time() - st_time, 1)
    print(f"\nTrained new model in {elapsed} seconds")

    # Assess new model
    print("\n\nAssessing")
    st_time = time.time()
    contender_matchup_info = run_faceoff(
        environment,
        bot_species,
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
