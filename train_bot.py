import time
import pathlib
import os
import json
from dataclasses import dataclass
from self_play import run as run_self_play
from train import run as run_model_training
from assess import run_faceoff
from paths import build_model_directory, build_training_info_path


@dataclass
class TrainingInfo:
    environment: str
    species: str
    current_batch: int
    self_play_bot_generation: int

    @classmethod
    def load(cls, environment, species):
        training_info_path = build_training_info_path(environment, species)
        if not os.path.exists(training_info_path):
            data = dict(
                environment=environment,
                species=species,
                current_batch=1,
                self_play_bot_generation=1,
            )
        else:
            data = json.loads(open(training_info_path, 'r').read())
        return cls(**data)

    def save(self):
        training_info_path = build_training_info_path(self.environment, self.species)
        data = dict(
            environment=self.environment,
            species=self.species,
            current_batch=self.current_batch,
            self_play_bot_generation=self.self_play_bot_generation,
        )
        with open(training_info_path, 'w') as f:
            f.write(json.dumps(data))
        print("Saved training info to", training_info_path)


def setup_filesystem(environment, species, generation):
    model_dir = build_model_directory(environment, species, generation)
    pathlib.Path(model_dir).mkdir(parents=True, exist_ok=True)


environment = "connect_four"
# bot_species = "mcts_naive"
bot_species = "mcts_gbdt"
batches_to_train = 2

training_info = TrainingInfo.load(
    environment,
    bot_species,
)
# training_info.save() # XXX: remove

num_workers = 16
games_per_batch = 10_000
# games_per_batch = num_workers * 10

num_faceoff_rounds = 25 # 25 rounds * 2 bots * 3 games per bot per round = 150 games
adjusted_win_rate_threshold = .55


for i in range(batches_to_train):
    print(f"\n\nBatch {training_info.current_batch + i} / {training_info.current_batch + batches_to_train - 1}")
    print(f"environment: {environment}, species: {bot_species}")

    # Ensure directories are made/etc.
    # - XXX: Not sure this actually depends on generation, but maybe it will later.
    setup_filesystem(
        environment,
        bot_species,
        training_info.self_play_bot_generation,
    )

    training_generation = training_info.self_play_bot_generation + 1

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
