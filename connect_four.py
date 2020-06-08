import json
from dataclasses import dataclass, asdict
from typing import (
    List,
    Any,
    Dict,
)
import environment
import numpy


##############
# Board
##############
'''
5 . . . . . . . f
4 . . . . . . . e
3 . . . . . . . d
2 . . . . . . . c
1 . . . . . . . b
0 . . . . . . . a
  0 1 2 3 4 5 6
'''


@dataclass
class State:
    board: List[List[int]]
    whose_move: int # {0, 1}

    def marshall(self, format="dict"):
        # better to do as_tuple?
        if format == "dict":
            return asdict(self)
        elif format == "json":
            return json.dumps(asdict(self))
        else:
            raise KeyError(f"Unknown format: {format}")

    @classmethod
    def unmarshall(cls, data, format="dict"):
        if format == "dict":
            return cls(**data)


BOARD_POSITIONS = []
for x in (0, 1, 2, 3, 4, 5, 6):
    for y in (0, 1, 2, 3, 4, 5):
        BOARD_POSITIONS.append((x, y))


def generate_features(state, agents) -> numpy.array:
    # :agents ~ [0, 1]
    # :agents ~ [0, 1, 2]
    # :agents ~ [0, 1, 5]
    # Which player's pov, which player is moving, is POV's player moving?
    features = numpy.zeros((2, 87), dtype=numpy.float32)

    agent_0_features = features[0]
    agent_1_features = features[1]

    agent_0_features[0] = 0.0
    agent_0_features[1] = state.whose_move
    agent_0_features[2] = 1.0 if 0 == state.whose_move else 0.0

    for i, (x, y) in enumerate(BOARD_POSITIONS):
        p = state.board[x][y]
        if p == 1:
            agent_0_features[i + 3] = 1.0
        elif p == 2:
            agent_0_features[i + 45] = 1.0

    # Copy agent 0's features and modify the only ones that changed for agent 1
    agent_1_features[0:87] = agent_0_features[0:87]
    agent_1_features[0] = 1.0
    agent_1_features[2] = 1.0 if 1 == state.whose_move else 0.0
    return features


@dataclass
class Environment(environment.Environment):
    checks_by_position: Any = None
    action_name_by_id: Dict[int, str] = None

    def __post_init__(self):
        super().__post_init__()
        self.checks_by_position = self.calculate_adjacent_checks()
        _, self.action_name_by_id = self.build_action_maps()

    def add_agent(self, agent):
        super().add_agent(agent)

    def get_name(self):
        return "connect_four"

    def initial_state(self):
        return State(
            board=[[0] * 6 for _ in range(7)],
            whose_move=0,
        )

    def transition_state(self, state, move):
        # Copy previous board
        # - [:] is 2x faster than copy/deepcopy
        board = [x[:] for x in state.board]

        # Find first column that is empty for row chosen (move)
        for y in (0, 1, 2, 3, 4, 5):
            if board[move][y] == 0:
                board[move][y] = state.whose_move + 1 # +1 because whose_move is 0-based
                break
        whose_move = 1 if state.whose_move == 0 else 0
        return State(board=board, whose_move=whose_move)

    def calculate_adjacent_checks(self):
        '''
        for every (x, y), should be 4 checks:
            [
                (left, right), # left ~ ((x, y), ...)
                (down, up),
                (updiag_left, updiag_right),
                (downdiag_left, downdiag_right)
            ]

        XXX: You can get rid of a lot of these checks:
          - The corners can't have diaganol 4.
          - Only need to check 3 bottom rows for vertical and diaganol.
          - Only need to check 4 left columns for horizontal.
        '''
        checks = [[0] * 6 for _ in range(7)]
        for y in (0, 1, 2, 3, 4, 5):
            for x in (0, 1, 2, 3, 4, 5, 6):
                checks[x][y] = []

                '''
                if x == 3 and y == 1:
                    import pdb; pdb.set_trace()
                '''

                # left / right
                first = []
                second = []
                for dx in (1, 2, 3):
                    xp = x - dx
                    if 0 <= xp <= 6:
                        first.append((xp, y))
                    xp = x + dx
                    if 0 <= xp <= 6:
                        second.append((xp, y))
                checks[x][y].append((tuple(first), tuple(second)))

                # down / up
                first = []
                second = []
                for dy in (1, 2, 3):
                    yp = y - dy
                    if 0 <= yp <= 5:
                        first.append((x, yp))
                    yp = y + dy
                    if 0 <= yp <= 5:
                        second.append((x, yp))
                checks[x][y].append((tuple(first), tuple(second)))

                # updiag left / right
                first = []
                second = []
                for d in (1, 2, 3):
                    xp = x - d
                    yp = y - d
                    if (0 <= xp <= 6) and (0 <= yp <= 5):
                        first.append((xp, yp))
                    xp = x + d
                    yp = y + d
                    if (0 <= xp <= 6) and (0 <= yp <= 5):
                        second.append((xp, yp))
                checks[x][y].append((tuple(first), tuple(second)))

                # downdiag left / right
                first = []
                second = []
                for d in (1, 2, 3):
                    xp = x - d
                    yp = y + d
                    if (0 <= xp <= 6) and (0 <= yp <= 5):
                        first.append((xp, yp))
                    xp = x + d
                    yp = y - d
                    if (0 <= xp <= 6) and (0 <= yp <= 5):
                        second.append((xp, yp))
                checks[x][y].append((tuple(first), tuple(second)))

        return checks

    def find_winner(self, state):
        '''
        For every occupied position on the board:
          - Expanding from that position, check positions surrounding it for 4-in-a-row.
        '''
        b = state.board
        for y in (0, 1, 2, 3, 4, 5):
            for x in (0, 1, 2, 3, 4, 5, 6):
                if b[x][y] == 0:
                    continue
                '''
                direction_checks: (
                    ((x, y), ...), # e.g., horizontal left coords
                    ((x, y), ...), # e.g., horizontal right coords
                )
                '''
                player = b[x][y]
                for direction_checks in self.checks_by_position[x][y]:
                    num_connected = 1
                    for check_coords in direction_checks:
                        for check_x, check_y in check_coords:
                            if b[check_x][check_y] == player:
                                num_connected += 1
                            else:
                                break

                    # After checking first and second direct, check if enough are connected.
                    if num_connected >= 4:
                        return player
        return 0

    def is_terminal(self, state):
        # 4-in-a-row
        if self.find_winner(state) > 0:
            return True

        # Draw
        # - Check if all pieces have been placed
        # - Start at top row, most of the time it'll be empty
        # - Using literals is faster than range.
        # XXX: Might be easier to check move history length?
        for y in (5, 4, 3, 2, 1, 0):
            for x in (0, 1, 2, 3, 4, 5, 6):
                if state.board[x][y] == 0:
                    return False
        return True

    def build_action_maps(self):
        action_name_by_id = {action_id: str(action_id) for action_id in self.all_possible_actions()}
        action_id_by_name = {v: k for k, v in action_name_by_id.items()}
        return action_id_by_name, action_name_by_id

    def translate_human_input(self, human_input):
        return int(human_input)

    def all_possible_actions(self):
        return tuple(range(7))

    def enumerate_actions(self, state):
        actions = []
        for x in (0, 1, 2, 3, 4, 5, 6):
            for y in (0, 1, 2, 3, 4, 5):
                if state.board[x][y] == 0:
                    actions.append(x)
                    break
        return actions

    def rewards(self, state):
        winner = self.find_winner(state)
        if winner == 1:
            return (1, -1)
        elif winner == 2:
            return (-1, 1)
        else:
            return (0, 0)

    def early_stopped_rewards(self, state):
        return None

    def early_stopping_round(self):
        return None

    def text_display(self, state, rich=True):
        '''
        5 . . . . . . .
        4 . . . . . . .
        3 . . . . . . .
        2 . . . . . . .
        1 . . . . . . .
        0 . . . . . . .
          0 1 2 3 4 5 6
        '''
        board_display = [["."] * 6 for _ in range(7)]
        for x in range(7):
            for y in range(6):
                char = "."
                if state.board[x][y] == 1:
                    char = "[bold yellow]●[/bold yellow]"
                elif state.board[x][y] == 2:
                    char = "[bold red]●[/bold red]"
                elif state.board[x][y] == -1: # designated empty for decision tree display
                    char = "[bold white]○[/bold white]"
                board_display[x][y] = char

        s = ""
        for y in range(5, -1, -1):
            row = f" {y} "
            for x in range(7):
                row += board_display[x][y] + " "
            s += row + "\n"
        s += "   0 1 2 3 4 5 6"
        return s

    def run(self):
        return super().run()
