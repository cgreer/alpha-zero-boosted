import os
from dataclasses import dataclass, asdict, astuple
import json
from multiprocessing import Pool, set_start_method
import pathlib
from typing import (
    List,
)

from paths import build_model_directory, build_training_info_path
from batch_info import BatchInfo

from environment_registry import get_env_module

from paths import find_batch_directory
from system_monitoring import SystemMonitor
from training_samples import iter_replay_data

# - https://pythonspeed.com/articles/python-multiprocessing/
set_start_method("spawn")


@dataclass
class WorkerStats:
    total_mcts_considerations: int = 0
    num_games: int = 0
    num_positions: int = 0


def batch_info_worker_task(args):
    (
        environment_name,
        species_name,
        batch_num,
        worker_num,
        num_workers,
    ) = args

    # Go through every replay and sum up stats
    env_class = get_env_module(environment_name)
    replay_directory = find_batch_directory(environment_name, species_name, batch_num)
    ws = WorkerStats()
    for agent_replay in iter_replay_data(
        replay_directory,
        env_class.State,
        worker_num,
        num_workers,
    ):
        ws.total_mcts_considerations += agent_replay.total_mcts_considerations()
        ws.num_games += 1
        ws.num_positions += len(agent_replay.positions)
    return astuple(ws)


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

    def update_batch_stats(self, num_workers=14):
        '''
        Update post-batch stats like num_games, mcts_considerations, etc.
        '''
        sys_mon = SystemMonitor()
        for batch in self.batches:
            # If batch stats exist, don't redo...
            if batch.self_play_cpu_time is not None:
                print("Batch stats already updated for", batch.batch_num)
                continue

            # Update self_play_cpu_time
            # - Sanity check that there was a continuous sampling of cpu utlization
            #   by checking that it got at least a sample every 3 seconds on
            #   average.
            self_play_cpu_time, num_samples = sys_mon.query_utilization(
                batch.self_play_start_time,
                batch.self_play_end_time,
            )
            min_allowable_samples = ((batch.self_play_end_time - batch.self_play_start_time) / 3)
            if num_samples < min_allowable_samples:
                raise RuntimeError(f"Monitoring didn't take enough samples: {num_samples} < {min_allowable_samples}")
            batch.self_play_cpu_time = self_play_cpu_time

            # Grab replay-dependent batch stats
            worker_args = []
            for worker_num in range(num_workers):
                worker_args.append(
                    (
                        self.environment,
                        self.species,
                        batch.batch_num,
                        worker_num,
                        num_workers
                    )
                )
            with Pool(num_workers) as p:
                results = p.map(batch_info_worker_task, worker_args)

            batch.num_games = 0
            batch.num_positions = 0
            batch.total_mcts_considerations = 0
            for worker_num, result in enumerate(results):
                print("Finished", worker_num, self.species, batch.batch_num)
                stats = WorkerStats(*result)
                batch.num_games += stats.num_games
                batch.num_positions += stats.num_positions
                batch.total_mcts_considerations += stats.total_mcts_considerations

            # These are double-counted for both agents, total_mcts_considerations is not.
            batch.num_games = batch.num_games // 2
            batch.num_positions = batch.num_positions // 2

        # Record the batch info
        self.save()

    def save(self):
        data = self.marshall()
        training_info_path = build_training_info_path(self.environment, self.species)
        with open(training_info_path, 'w') as f:
            f.write(json.dumps(data))
        print("Saved training info to", training_info_path)


def setup_filesystem(environment, species, generation):
    model_dir = build_model_directory(environment, species, generation)
    pathlib.Path(model_dir).mkdir(parents=True, exist_ok=True)
