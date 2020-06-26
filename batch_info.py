from dataclasses import dataclass, asdict


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
    num_games: int = None
    num_positions: int = None
    total_mcts_considerations: int = None
    self_play_cpu_time: int = None

    def marshall(self):
        return asdict(self)

    @classmethod
    def unmarshall(cls, data):
        return cls(**data)

    def start_time(self):
        return self.self_play_start_time

    def end_time(self):
        return self.assessment_end_time
