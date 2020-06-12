from dataclasses import dataclass
import pathlib
import time
from typing import (
    List,
)
import json

import psutil

import settings

HOUR_SECONDS = 60 * 60
DAY_SECONDS = 24 * HOUR_SECONDS


def ts_buckets(collection_time):
    ts_day = int(time.time() / DAY_SECONDS) * DAY_SECONDS
    ts_hour = int(time.time() / HOUR_SECONDS) * HOUR_SECONDS
    return ts_day, ts_hour


def resolve_path_info(ts_day, ts_hour):
    # statdir/<ts_day>/<ts_hour>.json
    directory = f"{settings.SYSTEM_STATS_DIRECTORY}/{ts_day}"
    base_name = f"{ts_hour}.json"
    full_path = f"{directory}/{base_name}"
    return directory, base_name, full_path


def iter_stat_files(start_ts, end_ts):
    raise NotImplementedError()


def iter_events(file_path, start_ts, end_ts):
    with open(file_path, 'r') as f:
        for line in f:
            stats = StatEvent.unmarshall(json.loads(line.strip()))
            if not (start_ts <= stats.collection_time <= end_ts):
                continue
            yield stats


def iter_stat_events(start_ts, end_ts):
    # open initial file
    for i, file_path in iter_stat_files(start_ts, end_ts):
        for stat_event in iter_events(file_path, start_ts, end_ts):
            yield stat_event


def span_cpu_info(start_ts, end_ts):
    time_total = 0.0
    cpu_seconds_total = 0.0
    previous_time = 0.0
    for i, stats in enumerate(iter_stat_events(start_ts, end_ts)):
        if i == 0:
            previous_time = stats.collection_time
            continue
        elapsed = stats.collection_time - previous_time
        cpus_used = sum(stats.cpu_utilizations)

        time_total += elapsed
        cpu_seconds_total += (elapsed * cpus_used)

        previous_time = stats.collection_time

    return cpu_seconds_total, time_total


@dataclass
class StatEvent:
    collection_time: float
    cpu_utilizations: List[float]

    def ts_buckets(self):
        return ts_buckets(self.collection_time)

    def path_info(self):
        return resolve_path_info(*self.ts_buckets())

    def marshall(self):
        return dict(
            collection_time=self.collection_time,
            cpu_utilizations=self.cpu_utilizations,
        )

    @classmethod
    def unmarshall(cls, data):
        return StatEvent(
            collection_time=data["collection_time"],
            cpu_utilizations=data["cpu_utilizations"],
        )


@dataclass
class Monitor:

    def _gather_stats(self):
        collection_time = time.time()
        cpu_utilizations = psutil.cpu_percent(interval=0.2, percpu=True)
        return StatEvent(
            collection_time=collection_time,
            cpu_utilizations=cpu_utilizations,
        )

    def log_stat(self):
        stats = self._gather_stats()
        stat_dir, _, stat_path = stats.path_info()

        # Make statdir directory if it doesn't exist
        pathlib.Path(stat_dir).mkdir(parents=True, exist_ok=True)

        # Log stat
        data = json.dumps(stats.marshall())
        with open(stat_path, 'a') as f:
            f.write(f"{data}\n")
        print(f"Logged stats to: {stat_path}")

    def run(self):
        while True:
            self.log_stat()
            time.sleep(1.0)


if __name__ == "__main__":
    m = Monitor()
    m.run()
