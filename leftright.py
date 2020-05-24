from dataclasses import dataclass
import environment


@dataclass(frozen=True)
class State:
    board_length: int
    player_position: int
    whose_move: int = 0


@dataclass
class Environment(environment.Environment):
    board_length: int

    def __post_init__(self):
        super().__post_init__()
        assert self.board_length % 2 == 1, "Must be odd"

    def add_agent(self, agent):
        super().add_agent(agent)

    def get_name(self):
        return "leftright"

    def initial_state(self):
        return State(
            board_length=self.board_length,
            player_position=self.board_length // 2,
        )

    def transition_state(self, state, move):
        new_position = state.player_position - 1
        if move == "r":
            new_position = state.player_position + 1

        return State(
            board_length=state.board_length,
            player_position=new_position,
        )

    def is_terminal(self, state):
        if state.player_position in (0, state.board_length - 1):
            return True
        return False

    def translate_human_input(self, human_input):
        return human_input

    def enumerate_actions(self, state):
        return ("l", "r")

    def rewards(self, state):
        # :agents unused for this game cuz only 1
        if state.player_position == 0:
            return [-1]
        if state.player_position == (state.board_length - 1):
            return [1]
        return [0]

    def text_display(self, state):
        board = ["—"] * state.board_length
        board[0] = "L"
        board[state.board_length - 1] = "W"
        board[state.player_position] = "X"
        return "".join(board)

    def run(self):
        super().run()
