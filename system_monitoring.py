from dataclasses import dataclass
import time
from typing import (
    Any,
)

import psutil
import sqlite3

from paths import full_path_mkdir_p
import settings


@dataclass
class SystemStats:
    event_time: float
    cpu_utilization_total: float

    @classmethod
    def build_table(cls, conn):
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS system_stats (
                event_time real,
                cpu_utilization_total real
            );
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS system_stats_idx on system_stats (event_time);
            """
        )
        conn.commit()

    def log_stat(self, conn):
        cursor = conn.cursor()
        cursor.execute(
            f"""
            INSERT INTO system_stats VALUES (
                {self.event_time},
                {self.cpu_utilization_total}
            )
            """
        )
        conn.commit()
        print("logged:", self.event_time, self.cpu_utilization_total)

    @classmethod
    def gather_stats(cls):
        event_time = time.time()
        cpu_utilization_total = sum(psutil.cpu_percent(interval=0.2, percpu=True))
        return cls(
            event_time=event_time,
            cpu_utilization_total=cpu_utilization_total,
        )


@dataclass
class SystemMonitor:
    db_connection: Any = None
    sample_every: float = 1.0 # seconds to sample

    def __post_init__(self):
        # Create/stash connection
        db_path = settings.MONITORING_DB_PATH
        full_path_mkdir_p(db_path)
        self.db_connection = sqlite3.connect(db_path)

        # Ensure tables are created
        # - All table/index creates are re-runnable
        SystemStats.build_table(self.db_connection)

    def query_utilization(
        self,
        start_time,
        end_time,
    ):
        cursor = self.db_connection.cursor()
        query = f"""
            SELECT
                event_time
                , cpu_utilization_total
            from system_stats
            where
                event_time >= {start_time}
                AND event_time <= {end_time}
            """
        num_events = 0
        total_utilization = 0
        last_time = start_time
        for row in cursor.execute(query):
            event_time, sum_cpu_util = row
            event_time = float(event_time)
            num_events += 1
            elapsed = float(event_time) - last_time
            total_utilization += (elapsed * sum_cpu_util)
            last_time = event_time
        return (total_utilization, num_events)

    def run(self):
        while True:
            stat_event = SystemStats.gather_stats()
            stat_event.log_stat(self.db_connection)
            time.sleep(self.sample_every)


if __name__ == "__main__":
    m = SystemMonitor()
    m.run()
