import os
from dataclasses import dataclass
import json
import pathlib

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
