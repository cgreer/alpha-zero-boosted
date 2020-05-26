from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from rich import print as rprint
from typing import Any, List
import random
import settings
import time
import uuid
import numpy


@dataclass
class Environment(ABC):
    # init=False will not put it in the constructor (so not passing won't raise error)
    # - These fields are initialized in __post_init__
    id: str = field(init=False)
    agents: List[Any] = field(init=False)
    event_history: List[Any] = field(init=False)
    started_at: float = field(init=False)
    ended_at: float = field(init=False)
    random_seed: int = field(init=False)

    @abstractmethod
    def __post_init__(self):
        self.id = str(uuid.uuid4())
        self.agents = []
        self.event_history = []
        self.started_at = -1.0
        self.ended_at = -1.0
        self.random_seed = random.randint(0, 10_000_000)
        if settings.VERBOSITY >= 1:
            print("RANDOM SEED:", self.random_seed)

        # Record and set the random seed so we can replay games deterministically
        numpy.random.seed(self.random_seed)

        # Uncomment this to make games always play the same way for timing tests.
        # numpy.random.seed(1838298)

    @abstractmethod
    def add_agent(self, agent):
        self.agents.append(agent)
        agent.set_agent_num(len(self.agents) - 1)

    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def initial_state(self):
        pass

    @abstractmethod
    def transition_state(self, state, move):
        pass

    @abstractmethod
    def is_terminal(self, state) -> bool:
        pass

    @abstractmethod
    def translate_human_input(self, human_input):
        pass

    @abstractmethod
    def enumerate_actions(self, state):
        pass

    @abstractmethod
    def all_possible_actions(self, state):
        pass

    @abstractmethod
    def rewards(self, state):
        pass

    @abstractmethod
    def text_display(self, state):
        pass

    @abstractmethod
    def run(self):
        # Setup board
        game_state = self.initial_state()

        # Let agents do any setup
        for agent in self.agents:
            agent.setup(initial_state=game_state)

        # Play
        self.started_at = time.time()
        turn_count = 0
        while True:
            turn_count += 1

            if settings.VERBOSITY >= 1:
                rprint("\n\n====== TURN", turn_count, f"(P{game_state.whose_move + 1}) ======")
                rprint()
                rprint(self.text_display(game_state))
                rprint()

            # Stop if game is over
            # - Record final state before
            if self.is_terminal(game_state):
                self.event_history.append((game_state, None))
                break

            # Get next action
            agent_to_move = self.agents[game_state.whose_move]
            move = agent_to_move.make_move()
            if settings.VERBOSITY >= 1:
                rprint(f"\nAgent {agent_to_move.agent_num} chose [bold green]{move}[/bold green]")

            # Advance game state
            # - Record action first before transitioning
            self.event_history.append((game_state, move))
            game_state = self.transition_state(game_state, move)

            # Tell players about it
            # - mask unobservable state here.
            for agent in self.agents:
                agent.handle_move(move, game_state)

        # Game Over
        self.ended_at = time.time()
        outcome = self.rewards(game_state)
        if settings.VERBOSITY >= 1:
            rprint()
            rprint("Game Over")
            rprint(outcome)
        return outcome
