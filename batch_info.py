from dataclasses import dataclass, astuple
from multiprocessing import Pool

from training_info import TrainingInfo
from environment_registry import get_env_module
from paths import find_batch_directory
from system_monitoring import SystemMonitor
from training_samples import iter_replay_data


@dataclass
class WorkerStats:
    total_mcts_considerations: int = 0
    num_games: int = 0
    num_positions: int = 0


def run_worker(args):
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


def update_batch_stats(environment_name, species_name, num_workers=12):
    sys_mon = SystemMonitor()
    tinfo = TrainingInfo.load(environment_name, species_name)
    for batch in tinfo.batches:
        # If batch stats exist, don't redo...
        if batch.self_play_cpu_time is not None:
            print("Batch stats already updated for", batch.batch_num)
            continue

        batch.self_play_cpu_time = sys_mon.query_utilization(
            batch.self_play_start_time,
            batch.self_play_end_time,
        )

        # Sanity check that there was a continuous sampling of cpu utlization.
        # - Check that it got at least a sample every 3 seconds
        self_play_cpu_time, num_samples = sys_mon.query_utilization(
            batch.self_play_start_time,
            batch.self_play_end_time,
        )
        min_allowable_samples = ((batch.self_play_end_time - batch.self_play_start_time) / 3)
        if num_samples < min_allowable_samples:
            raise RuntimeError(f"Monitoring didn't take enough samples: {num_samples} < {min_allowable_samples}")
        batch.self_play_cpu_time = self_play_cpu_time

        # Grab batch stats
        worker_args = []
        for worker_num in range(num_workers):
            worker_args.append(
                (
                    environment_name,
                    species_name,
                    batch.batch_num,
                    worker_num,
                    num_workers
                )
            )
        with Pool(num_workers) as p:
            results = p.map(run_worker, worker_args)

        batch.num_games = 0
        batch.num_positions = 0
        batch.total_mcts_considerations = 0
        for worker_num, result in enumerate(results):
            print("Finished", worker_num, species_name, batch.batch_num)
            stats = WorkerStats(*result)
            batch.num_games += stats.num_games
            batch.num_positions += stats.num_positions
            batch.total_mcts_considerations += stats.total_mcts_considerations

        # These are tabulated for both agents, total_mcts_considerations is not.
        batch.num_games = batch.num_games // 2
        batch.num_positions = batch.num_positions // 2

    # Record the batch info
    tinfo.save()


if __name__ == "__main__":
    update_batch_stats("connect_four", "gbdt_pcrR1")
