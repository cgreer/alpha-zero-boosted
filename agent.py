from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any,
)


@dataclass
class Agent(ABC):
    environment: Any
    species: str
    generation: int
    agent_num: int = field(init=False)

    @abstractmethod
    def __post_init__(self):
        self.agent_num = None

    @abstractmethod
    def set_agent_num(self, agent_num):
        self.agent_num = agent_num

    @abstractmethod
    def setup(self, initial_state):
        pass

    @abstractmethod
    def handle_move(self, initial_state):
        pass

    @abstractmethod
    def make_move(self, initial_state):
        pass
