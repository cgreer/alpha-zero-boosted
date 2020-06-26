from dataclasses import dataclass, asdict

from training_info import TrainingInfo


@dataclass
class GenerationInfo:
    environment: str
    species: str
    generation: int
    num_batches_to_train: int = None
    wall_clock_time_to_train: float = None
    cpu_seconds_to_train: float = None
    mcts_considerations: int = None

    def marshall(self):
        return asdict(self)

    @classmethod
    def unmarshall(cls, data):
        return cls(**data)

    @classmethod
    def from_generation_info(
        cls,
        environment: str,
        species: str,
        gen: int,
    ):
        data = dict(
            environment=environment,
            species=species,
            generation=gen,
        )
        data.update(cls.collect_training_stats(environment, species, gen))
        return cls(**data)

    @staticmethod
    def collect_training_stats(environment, species, generation):
        training_info = TrainingInfo.load(environment, species)

        # Find batch
        num_batches_to_train = 0
        wall_clock_time_to_train = 0.0
        cpu_seconds_to_train = 0.0
        mcts_considerations = 0
        if generation > 1:
            for tbatch in training_info.batches:
                num_batches_to_train += 1
                wall_clock_time_to_train += tbatch.self_play_end_time - tbatch.self_play_start_time

                # util is reported from 0.0 to 100.0 for each CPU for some reason...
                cpu_seconds_to_train += (tbatch.self_play_cpu_time / 100.0)
                mcts_considerations += tbatch.total_mcts_considerations

                # This is the batch that trained this generation
                if tbatch.generation_trained == generation:
                    break

        return dict(
            num_batches_to_train=num_batches_to_train,
            wall_clock_time_to_train=wall_clock_time_to_train,
            cpu_seconds_to_train=cpu_seconds_to_train,
            mcts_considerations=mcts_considerations,
        )
