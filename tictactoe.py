import json
from dataclasses import dataclass, asdict
from typing import Tuple
import environment


# Board Indicies
# c 0, 1, 2,
# b 3, 4, 5,
# a 6, 7, 8
#   1  2  3

@dataclass
class State:
    board: Tuple[int] # ([0-2], ...)
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


def generate_features(state, agent_num):
    # Which player's pov, which player is moving, is POV's player moving?
    features = [agent_num, state.whose_move, 1 if agent_num == state.whose_move else 0]

    # X/O masks
    features.extend((1 if pos == 1 else 0 for pos in state.board))
    features.extend((1 if pos == 2 else 0 for pos in state.board))

    return features


TERMINAL_INDICES = (
    # Horizontal
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),

    # Vertical
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),

    # Diagonal
    (0, 4, 8),
    (2, 4, 6),
)


@dataclass
class Environment(environment.Environment):

    def __post_init__(self):
        super().__post_init__()

    def add_agent(self, agent):
        super().add_agent(agent)

    def get_name(self):
        return "tictactoe"

    def initial_state(self):
        return State(
            board=(
                0, 0, 0,
                0, 0, 0,
                0, 0, 0
            ),
            whose_move=0,
        )

    def transition_state(self, state, move):
        board = list(state.board)
        board[move] = state.whose_move + 1
        whose_move = 1 if state.whose_move == 0 else 0
        return State(board=tuple(board), whose_move=whose_move)

    def find_winner(self, state):
        board = state.board
        for a, b, c in TERMINAL_INDICES:
            for player_num in (1, 2):
                if board[a] == player_num and board[b] == player_num and board[c] == player_num:
                    return player_num
        return 0

    def is_terminal(self, state):
        # 3-in-a-row
        if self.find_winner(state) > 0:
            return True

        # Draw
        num_placed = 0
        for val in state.board:
            if val > 0:
                num_placed += 1
        if num_placed >= 9:
            return True

        return False

    def translate_human_input(self, human_input):
        return int(human_input)

    def enumerate_actions(self, state):
        actions = []
        for i, value in enumerate(state.board):
            if value == 0:
                actions.append(i)
        return actions

    def all_possible_actions(self):
        return tuple(range(9))

    def rewards(self, state):
        winner = self.find_winner(state)
        if winner == 1:
            return (1, -1)
        elif winner == 2:
            return (-1, 1)
        else:
            return (0, 0)

    def text_display(self, state):
        board_display = [" "] * 9
        for i, val in enumerate(state.board):
            if val == 0:
                continue
            elif val == 1:
                board_display[i] = "X"
            else:
                board_display[i] = "O"
        s = "|".join(board_display[:3])
        s += "\n—————"
        s += "\n" + "|".join(board_display[3:6])
        s += "\n—————"
        s += "\n" + "|".join(board_display[6:])
        return s

    def run(self):
        return super().run()
