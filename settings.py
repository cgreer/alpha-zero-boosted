import os
from pathlib import Path

HOME = str(Path.home())

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
SYSTEM_STATS_DIRECTORY = f"{HOME}/system_monitoring"
TMP_DIRECTORY = "/tmp"

NUM_THREADS = 14
NUM_CORES = 14

TOOL_CHAIN = "clang" # Will need to change this for linux/windows
