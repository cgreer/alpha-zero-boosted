from dataclasses import dataclass
import random
from typing import (
    Any,
)

from agent import Agent


@dataclass
class RandomAgent(Agent):
    current_state: Any = None

    def __post_init__(self):
        super().__post_init__()

    def set_agent_num(self, agent_num):
        super().set_agent_num(agent_num)

    def setup(self, initial_state):
        self.current_state = initial_state

    def handle_move(self, move, resulting_state):
        self.current_state = resulting_state

    def make_move(self):
        eligible_actions = self.environment.enumerate_actions(self.current_state)
        return random.choice(eligible_actions)
