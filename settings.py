import os

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
VERBOSITY = int(os.getenv("RL_VERBOSE")) if os.getenv("RL_VERBOSE") else 1

ROOT_DATA_DIRECTORY = "/Users/chrisgreer/rl_data"

TMP_DIRECTORY = "/tmp"

NUM_THREADS = 14

TOOL_CHAIN = "clang" # Will need to change this for linux/windows
