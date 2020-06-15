from dataclasses import dataclass, asdict
import glob
from typing import (
    Any,
    List,
    Dict,
)
import json
import os
import time
import pprint

import numpy
from rich import print as rprint

from environment import Environment
from environment_registry import get_env_module
from settings import ROOT_DATA_DIRECTORY
from text import stitch_text_blocks


@dataclass
class ActionConsideration:
    id: int
    prior_probability: float
    visit_count: int
    rewards_total: float

    def marshall(self):
        return asdict(self)

    @classmethod
    def unmarshall(cls, data):
        return cls(**data)


@dataclass
class Position:
    index: int
    state: Any # An Environment's State instance
    actions_considered: Dict[int, ActionConsideration] # action_id: ActionConsideration
    chosen_action_id: int # or None if terminal state
    value: float

    def marshall(self):
        return dict(
            index=self.index,
            state=self.state.marshall(),
            actions_considered={k: v.marshall() for k, v in self.actions_considered.items()},
            chosen_action_id=self.chosen_action_id,
            value=self.value,
        )

    @classmethod
    def unmarshall(cls, data, StateClass):
        data["state"] = StateClass.unmarshall(data["state"])
        data["actions_considered"] = {int(k): ActionConsideration.unmarshall(v) for k, v in data["actions_considered"].items()}
        return cls(**data)

    def edge_visits(self):
        num_visits = 0
        for ac in self.actions_considered.values():
            num_visits += ac.visit_count
        return num_visits

    def expectation(self) -> float:
        if self.is_terminal():
            return self.value
        else:
            ac = self.actions_considered[self.chosen_action_id]
            return ac.rewards_total / ac.visit_count

    def is_terminal(self):
        return self.chosen_action_id is None

    def policy(self):
        # XXX: This is not sorted the same way as
        # Environment.all_possible_actions!
        action_visits = [(ac.id, ac.visit_count) for ac in self.actions_considered.values()]
        action_visits.sort()

        policy = numpy.array([x[1] for x in action_visits])
        policy = policy / policy.sum()
        return policy


@dataclass
class AgentSettings:
    agent_num: int
    species: str
    generation: int
    puct_explore_factor: float
    puct_noise_alpha: float
    puct_noise_influence: float
    full_search_proportion: float
    full_search_steps: int
    partial_search_steps: int
    temperature: float
    require_full_steps: bool

    def marshall(self):
        return asdict(self)

    @classmethod
    def unmarshall(cls, data):
        return cls(**data)


@dataclass
class AgentReplay:
    game_id: str
    environment_name: str
    outcomes: List[float]
    started_at: float
    ended_at: float
    agent_nums: List[int]
    agent_settings: AgentSettings
    positions: List[Position]

    def marshall(self):
        return dict(
            game_id=self.game_id,
            environment_name=self.environment_name,
            outcomes=self.outcomes,
            started_at=self.started_at,
            ended_at=self.ended_at,
            agent_nums=self.agent_nums,
            agent_settings=self.agent_settings.marshall(),
            positions=[x.marshall() for x in self.positions],
        )

    @classmethod
    def unmarshall(cls, data, StateClass):
        data["agent_settings"] = AgentSettings.unmarshall(data["agent_settings"])
        data["positions"] = [Position.unmarshall(x, StateClass) for x in data["positions"]]
        return cls(**data)

    @classmethod
    def from_agent(cls, agent, was_early_stopped):
        env = agent.environment

        game_id = env.id
        environment_name = env.get_name()

        # XXX: Get rid of early stopping nonsense
        final_state = env.event_history[-1][0]
        if was_early_stopped:
            outcomes = env.early_stopped_rewards(final_state)
        else:
            outcomes = env.rewards(final_state)

        started_at = env.started_at
        ended_at = env.ended_at
        agent_nums = [x.agent_num for x in env.agents]

        agent_settings = AgentSettings(
            agent_num=agent.agent_num,
            species=agent.species,
            generation=agent.generation,
            puct_explore_factor=agent.puct_explore_factor,
            puct_noise_alpha=agent.puct_noise_alpha,
            puct_noise_influence=agent.puct_noise_influence,
            full_search_proportion=agent.full_search_proportion,
            full_search_steps=agent.full_search_steps,
            partial_search_steps=agent.partial_search_steps,
            temperature=agent.temperature,
            require_full_steps=agent.require_full_steps,
        )

        positions = []
        for position_index, (game_tree_node, chosen_action_id) in enumerate(agent.iter_game_tree_positions()):
            actions_considered = {}
            for child_edge in game_tree_node.child_edges:
                actions_considered[child_edge.move] = ActionConsideration(
                    child_edge.move,
                    child_edge.prior_probability,
                    child_edge.visit_count,
                    child_edge.reward_totals[agent.agent_num],
                )
            positions.append(
                Position(
                    index=position_index,
                    state=game_tree_node.state,
                    actions_considered=actions_considered,
                    chosen_action_id=chosen_action_id,
                    value=game_tree_node.values[agent.agent_num],
                )
            )

        return cls(
            game_id=game_id,
            environment_name=environment_name,
            outcomes=outcomes,
            started_at=started_at,
            ended_at=ended_at,
            agent_nums=agent_nums,
            agent_settings=agent_settings,
            positions=positions,
        )

    @classmethod
    def from_path(cls, replay_path, StateClass):
        data = json.loads(open(replay_path, 'r').read())
        return cls.unmarshall(data, StateClass)

    @classmethod
    def find_path(self, environment, replay_id):
        # aa8fd68f3668-0
        path_glob = f"{ROOT_DATA_DIRECTORY}/{environment}_*/self_play/*/*{replay_id}*.json"
        paths = glob.glob(path_glob)
        assert len(paths) == 1
        return paths[0]

    def iter_expectation_positions(self):
        '''
        Is this a terminal position? OR
        Did I do considerations from this position (i.e., was it my turn)?
        '''
        for i, position in enumerate(self.positions):

            if position.is_terminal():
                yield position
                continue

            if position.state.whose_move == self.agent_num:
                yield position

    def play_cli_video(
        self,
        initial_position_index,
        final_position_index,
        speed=0.3,
    ):
        env_module = get_env_module(self.environment_name)
        environment = env_module.Environment()
        current_state = environment.initial_state()
        current_position_idx = initial_position_index
        while current_position_idx <= final_position_index:
            current_state = self.positions[current_position_idx].state
            os.system("clear")
            print("Game", self.game_id)
            rprint(environment.text_display(current_state))
            time.sleep(speed)
            current_position_idx += 1

    def replay_game_from_position(
        self,
        initial_position,
        environment,
        species,
        generation,
        num_turns_to_play=1_000_000,
        agent_setting_overrides=None,
    ):
        # Setup game
        # - inline import needed for circular dep... XXX: fix
        from agent_configuration import configure_agent
        env_module = get_env_module(self.environment_name)
        env = env_module.Environment()

        Agent, agent_settings = configure_agent(environment, species, generation, "self_play")
        if agent_setting_overrides:
            for k, v in agent_setting_overrides.items():
                agent_settings[k] = v
        agent_1 = Agent(environment=env, **agent_settings)
        agent_2 = Agent(environment=env, **agent_settings)

        env.add_agent(agent_1)
        env.add_agent(agent_2)

        game_state = env.initial_state()

        # Let agents do any setup
        for agent in env.agents:
            agent.setup(initial_state=game_state)

        # Replay moves until you get to position
        current_position_idx = 0
        turn_count = 0
        while current_position_idx < initial_position:
            turn_count += 1
            current_position = self.positions[current_position_idx]
            current_position_idx += 1
            move = current_position.move

            rprint("\n\n====== TURN", turn_count, f"(P{game_state.whose_move + 1}) ======")
            rprint()
            rprint(env.text_display(game_state))
            rprint()

            agent_to_move = env.agents[game_state.whose_move]
            human_readable_move = env.action_name_by_id[move]
            rprint(f"\nAgent {agent_to_move.agent_num} chose [bold green]{human_readable_move}[/bold green]")

            # Advance game state
            # - Record action first before transitioning
            env.event_history.append((game_state, move))
            game_state = env.transition_state(game_state, move)
            for i, agent in enumerate(env.agents):
                print("agent", i)
                agent.handle_move(move, game_state)

        print("POLICY OVERRIDES")
        pprint.pprint(agent.policy_overrides)

        # Play rest of game out
        while True:
            turn_count += 1

            rprint("\n\n====== TURN", turn_count, f"(P{game_state.whose_move + 1}) ======")
            rprint()
            rprint(env.text_display(game_state))
            rprint()

            # Stop if game is over
            # - Record final state before
            if env.is_terminal(game_state):
                env.event_history.append((game_state, None))
                print("GAME OVER", env.rewards(game_state))
                break

            if (turn_count - initial_position) > num_turns_to_play:
                break

            # Get next action
            agent_to_move = env.agents[game_state.whose_move]
            move = agent_to_move.make_move()
            human_readable_move = env.action_name_by_id[move]
            rprint(f"\nAgent {agent_to_move.agent_num} chose [bold green]{human_readable_move}[/bold green]")

            # Advance game state
            # - Record action first before transitioning
            env.event_history.append((game_state, move))
            game_state = env.transition_state(game_state, move)

            # Tell players about it
            # - mask unobservable state here.
            for agent in env.agents:
                agent.handle_move(move, game_state)

        return env, env.agents


def display_best_moves(
    env: Environment,
    considered_actions: List[ActionConsideration],
):
    def build_table(considered_actions):
        ttext = "{:<8}{:<8}{:<8}{:<8}".format("MOVE", "VISITS", "PRIOR", "P(WIN)\n")
        for ca in considered_actions:
            p_win = None
            if ca.visit_count:
                p_win = round(ca.rewards_total / ca.visit_count, 3)

            color = "white"
            if (p_win or 0) > 0:
                color = "green"
            elif (p_win or 0) < 0:
                color = "red"

            prior = round(float(ca.prior_probability), 3)

            ttext += "{:<8}{:<8}{:<8}[{}]{:<8}[/{}]\n".format(
                env.action_name_by_id[ca.id],
                ca.visit_count,
                prior,
                color,
                (p_win or "-"),
                color,
            )
        return ttext

    most_visited_edges = considered_actions[:]
    most_visited_edges.sort(key=lambda x: (x.visit_count, x.prior_probability), reverse=True)

    total_visits = sum(x.visit_count for x in considered_actions)
    rprint(f"Num considerations: {total_visits}\n\n")

    tables = []
    tables.append(build_table(most_visited_edges[:10]))
    if len(most_visited_edges) > 10:
        tables.append(build_table(most_visited_edges[10:20]))
    if len(most_visited_edges) > 19:
        tables.append(build_table(most_visited_edges[20:30]))
    row_1 = stitch_text_blocks(tables, "        ")
    rprint(row_1)
