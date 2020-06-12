from dataclasses import dataclass
from typing import (
    Any,
)

from agent import Agent


@dataclass
class HumanAgent(Agent):
    current_state: Any = None

    def __post_init__(self):
        super().__post_init__()

    def set_agent_num(self, agent_num):
        super().set_agent_num(agent_num)

    def setup(self, initial_state):
        self.current_state = initial_state
        if not hasattr(initial_state, "whose_move"):
            raise RuntimeError("Agent requires knowing whose move it is each state")

    def handle_move(self, move, resulting_state):
        self.current_state = resulting_state

    def make_move(self):
        allowable_actions = self.environment.enumerate_actions(self.current_state)
        while True:
            move = input("Move? ")
            try:
                move = self.environment.translate_human_input(move)
            except Exception as e:
                print(e)
                print("Not an eligible move!\n")

            if move not in allowable_actions:
                print("Not an eligible move!\n")
                continue
            break
        return move
