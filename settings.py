import os
import platform
import psutil
from pathlib import Path

HOME = str(Path.home())
OS_PLATFORM = platform.system()

'''
0
  - No output
1
  - Display game Board
  - MCTS considerations per second
2
  - Considered move stats per move
  - Replay path
3
  - ...
4
  - ...
5
    - PUCT edge stats
'''
VERBOSITY = int(os.getenv("RL_VERBOSE")) if os.getenv("RL_VERBOSE") else 3

ROOT_DATA_DIRECTORY = f"{HOME}/rl_data"
TMP_DIRECTORY = "/tmp"

SYSTEM_STATS_DIRECTORY = f"{HOME}/system_monitoring"
MONITORING_DB_PATH = f"{HOME}/system_monitoring/monitoring.db"

NUM_CORES = psutil.cpu_count()

SELF_PLAY_THREADS = NUM_CORES
GBDT_TRAINING_THREADS = NUM_CORES
LIGHTGBM_THREADS = min(NUM_CORES, 8) # AWS EC2 AMI's can't handle too many?
TREELITE_THREADS = NUM_CORES
ASSESSMENT_THREADS = NUM_CORES


'''
TOOL_CHAIN used by treelite to compile tree
mac/os: clang
linux: gcc
windows: ?
'''
if OS_PLATFORM == "Darwin":
    TOOL_CHAIN = "clang"
elif OS_PLATFORM == "Linux":
    TOOL_CHAIN = "gcc"
else:
    raise KeyError(f"Unhandled platform: {OS_PLATFORM}")
