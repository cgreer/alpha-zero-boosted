import os
from dataclasses import dataclass, asdict
import json
import pathlib
from typing import (
    List,
    Dict,
)

from paths import build_model_directory, build_training_info_path


@dataclass
class BatchInfo:
    batch_num: int
    self_play_start_time: float
    self_play_end_time: float
    training_start_time: float
    training_end_time: float
    assessment_start_time: float
    assessment_end_time: float
    generation_self_play: int
    generation_trained: int
    assessed_awr: float

    def start_time(self):
        return self.self_play_start_time

    def end_time(self):
        return self.assessment_end_time

    def marshall(self):
        return asdict(self)

    @classmethod
    def unmarshall(cls, data):
        return cls(**data)


@dataclass
class TrainingInfo:
    environment: str
    species: str
    batches: List[BatchInfo]

    def current_self_play_generation(self):
        '''
        Find the highest generation that passed gating.
        '''
        for batch in self.batches[::-1]:
            if batch.generation_trained is None:
                continue
            return batch.generation_trained
        return 1

    @classmethod
    def load(cls, environment, species):
        training_info_path = build_training_info_path(environment, species)
        if not os.path.exists(training_info_path):
            return cls(
                environment=environment,
                species=species,
                batches=[],
            )
        data = json.loads(open(training_info_path, 'r').read())
        return cls.unmarshall(data)

    def finalize_batch(
        self,
        self_play_start_time,
        self_play_end_time,
        training_start_time,
        training_end_time,
        assessment_start_time,
        assessment_end_time,
        generation_self_play,
        generation_trained,
        assessed_awr,
    ):
        self.batches.append(
            BatchInfo(
                len(self.batches) + 1,
                self_play_start_time,
                self_play_end_time,
                training_start_time,
                training_end_time,
                assessment_start_time,
                assessment_end_time,
                generation_self_play,
                generation_trained,
                assessed_awr,
            )
        )
        self.save()

    def marshall(self):
        return asdict(self)

    @classmethod
    def unmarshall(cls, data):
        data["batches"] = [BatchInfo.unmarshall(x) for x in data["batches"]]
        return cls(**data)

    def save(self):
        data = self.marshall()
        training_info_path = build_training_info_path(self.environment, self.species)
        with open(training_info_path, 'w') as f:
            f.write(json.dumps(data))
        print("Saved training info to", training_info_path)


def setup_filesystem(environment, species, generation):
    model_dir = build_model_directory(environment, species, generation)
    pathlib.Path(model_dir).mkdir(parents=True, exist_ok=True)
