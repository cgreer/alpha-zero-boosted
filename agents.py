from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from rich import print as rprint
from typing import Any, List, Tuple
import json
import math
import numpy
import pathlib
import random
import settings
import time


@dataclass
class NoiseMaker:
    '''
    Presample and cache dirichlet noise
    '''
    num_samples_per_parameter: int = 10_000
    highest_index: int = None
    noise: Any = None

    def __post_init__(self):
        self.highest_index = self.num_samples_per_parameter - 1
        self.noise = {}

    def _sample_noise(self, alpha, size):
        return numpy.random.dirichlet([alpha] * size, self.num_samples_per_parameter).tolist()

    def make_noise(self, alpha, size):
        # Lookup cached noise
        # - If there is no previously cached noise then make some
        try:
            noise_data = self.noise[(alpha, size)]
        except KeyError:
            # Using an except avoids the "if not in" every call.  Normally I'd call "premature
            # optimization", but entire class exists to sample noise fast for an inner loop that
            # needs to be speedy.
            noise_data = [0, self._sample_noise(alpha, size)]
            self.noise[(alpha, size)] = noise_data

        # If your runway of noise has run out, cache some more.
        if noise_data[0] == self.highest_index:
            self.noise[(alpha, size)] = [0, self._sample_noise(alpha, size)]
        else:
            noise_data[0] = noise_data[0] + 1

        return noise_data[1][noise_data[0]]


NOISE_MAKER = NoiseMaker(1000)


@dataclass
class Agent(ABC):
    environment: Any
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


@dataclass
class GameTreeNode:
    '''
    :values
        - value_model value for each agent if it's a non-terminal node, else the rewards for each
          agent.
    '''
    state: Any
    parent_edge: Any
    child_edges: Any
    values: Tuple[float]
    visit_count: int # XXX: Should this be # selections or sum(edge selections)?
    is_terminal: bool

    def get_child_edge(self, move):
        edge_representing_move = None
        for edge in self.child_edges:
            if edge.move == move:
                edge_representing_move = edge
                break
        if not edge_representing_move:
            raise KeyError(f"Missing edge for move: {move}")
        return edge_representing_move


@dataclass
class GameTreeEdge:
    parent_node: GameTreeNode
    child_node: GameTreeNode
    move: str
    prior_probability: float # Prior for agent moving from this position
    visit_count: int
    reward_totals: List[float] # 1 float for each agent


@dataclass
class MCTSAgent(Agent):
    game_tree: Any
    current_node: Any
    feature_extractor: Any
    value_model: Any
    policy_model: Any
    move_consideration_steps: float
    move_consideration_time: float
    puct_explore_factor: float
    puct_noise_alpha: float
    puct_noise_influence: float

    def __post_init__(self):
        super().__post_init__()

    def set_agent_num(self, agent_num):
        super().set_agent_num(agent_num)

    def setup(self, initial_state):
        node = self.add_node(state=initial_state, parent_edge=None)
        self.game_tree = node
        self.current_node = node

    def handle_move(self, move, resulting_state):
        # :resulting_state is the observable state this agent can see.

        # Move game tree pointer
        # - If the move leads to an unexpanded state then add node. This can happen if
        #   opponent makes move first or a move this agent has never considered.
        edge_representing_move = self.current_node.get_child_edge(move)

        if edge_representing_move.child_node is None:
            # XXX: Does this affect mcts considerations?
            self.add_node(state=resulting_state, parent_edge=edge_representing_move)

        self.current_node = edge_representing_move.child_node
        assert self.current_node is not None, "Current node must always exist"

        self.prune_game_search_tree()

    def prune_game_search_tree(self):
        '''
        Remove all the considered — but unvisited — nodes/edges upstream of the
        current node in the game tree.  These were the nodes that MCTS
        considered, but the actions the players took didn't manifest in those
        tree pathways.  The stats of those pathways might be useful, but it's a
        lot of memory to keep around.

        The child edges of nodes that were visited ARE RETAINED because they contain
        the statistics (node visits) used to train the policy model.

        IF YOU DO NOT DO THIS THEN YOUR MEMORY USAGE WILL ESPLODE! Some games
        can have 1000s of moves and branching factors of 100s or more.
        '''
        # I am root. Ignore.
        if self.current_node.parent_edge is None:
            return

        # Go up the tree one move and prune all the downtree nodes except the
        # one the game ended up moving into.
        parent_node = self.current_node.parent_edge.parent_node
        for child_edge in parent_node.child_edges:
            if child_edge.child_node == self.current_node:
                continue
            del child_edge.child_node

    def add_node(self, state, parent_edge):
        '''
        Create game search tree node, link to parent edge.
        '''
        # Detect terminal once for each state and stash it instead of recalculating each
        # time you do mcts considerations.
        is_terminal = self.environment.is_terminal(state)

        # Calculate allowable_actions, agent values, agent policies
        if is_terminal:
            allowable_actions = []
            values = self.environment.rewards(state) # XXX: Enforce being tuple?
        else:
            allowable_actions = self.environment.enumerate_actions(state)

            agent_features = self.feature_extractor(state, self.environment.agents)
            values = self.value_model.predict(agent_features)

            # Only the policy of the agent that is moving at this position is needed.
            agent_policy = self.policy_model.predict(agent_features[state.whose_move], allowable_actions)

        node = GameTreeNode(
            state=state,
            parent_edge=parent_edge,
            child_edges=[],
            values=values,
            visit_count=0,
            is_terminal=is_terminal,
        )

        # Link to parent edge
        # - Except when it's the root node which has no parent edge
        if parent_edge is not None:
            parent_edge.child_node = node

        # Initialize edges
        for i, move in enumerate(allowable_actions):
            node.child_edges.append(
                GameTreeEdge(
                    parent_node=node,
                    child_node=None,
                    move=move,
                    prior_probability=agent_policy[i],
                    visit_count=0,
                    reward_totals=[0.0] * len(self.environment.agents),
                )
            )

        return node

    def puct(
        self,
        node,
        explore_factor=1.0,
        noise_alpha=1.0,
        noise_influence=0.25,
    ):
        # Get visit count of state
        # XXX: is node visit count (sum edge visits or does an expansion count as 1)
        # - What does a "node visit" mean?
        total_node_visits = node.visit_count

        # Ensure total_node_visits isn't 0
        # - If this is the first visit, then the puct exploitation_term and exploration_term would
        # both be zero without this adjustment. In which case, instead of choosing based on
        # prior for that first visit, it would choose randomly among all actions.
        # Ensuring this is at least 1 will allow the noisy prior to bias the choice.
        if total_node_visits < 1:
            total_node_visits = 1

        # Generate noise for prior probabilities to encourage MCTS exploration
        # - The :noise_alpha is determines the type of variability and the strength. A value below
        # 1.0 will concentrate the variability to one of the N moves. A value of 1.0 will make the
        # noise uniform across all moves.  As the value goes to infinity the noise becomes
        # ineffective.  Chess, shogi, and go had (average_moves, alpha) values of  [(35, .2), (92,
        # .15), (250, .03)] respectively.
        # noise = numpy.random.dirichlet([noise_alpha] * len(node.child_edges))
        noise = NOISE_MAKER.make_noise(noise_alpha, len(node.child_edges))

        # Get highest edge value
        sqrt_total_node_visits = math.sqrt(total_node_visits)
        best_value = 0.0
        best_edge = None
        agent_moving = node.state.whose_move # XXX: Shouldn't be required for state?
        for i, child_edge in enumerate(node.child_edges):
            # XXX: Correct behavior for 0 visits?
            # - Seems like it should be 0 and have the policy's prior affect the choice
            # instead when there is no evidence.
            exploitation_term = 0.0
            if child_edge.visit_count > 0:
                exploitation_term = child_edge.reward_totals[agent_moving] / child_edge.visit_count

            noisy_prior = (child_edge.prior_probability * (1 - noise_influence)) + (noise[i] * noise_influence)
            exploration_term = explore_factor * noisy_prior
            exploration_term = exploration_term * (sqrt_total_node_visits / (1 + child_edge.visit_count))

            puct_value = exploitation_term + exploration_term

            if settings.VERBOSITY >= 5:
                print(
                    f"puct edge ({hex(hash(str(node.state)))[-6:]})",
                    f"move:{child_edge.move}",
                    f"N(s):{total_node_visits}",
                    f"N(s, a):{child_edge.visit_count}",
                    f"W(s, a):{child_edge.reward_totals[agent_moving]}",
                    f"prior:{round(child_edge.prior_probability, 4)}",
                    f"noise:{round(noise[i], 4)}",
                    f"exploit:{round(exploitation_term, 3)}",
                    f"explore:{round(exploration_term, 3)}",
                )
            # No tie-breaker needed because of noise.
            if (puct_value >= best_value) or (best_edge is None):
                best_value = puct_value
                best_edge = child_edge

        return best_edge

    def run_mcts_considerations(
        self,
        num_steps,
        max_seconds,
        puct_explore_factor,
        puct_noise_alpha,
        puct_noise_influence,
    ):
        # Give the agent up to :num_steps and :max_seconds
        steps = 0
        st_time = time.time()
        while (steps < num_steps) or (time.time() - st_time) < max_seconds:
            steps += 1

            # Select / Expand
            # - For a given node, select next action from current state via puct. If resulting state
            # hasn't been explored before, then create it (with dangling edges).  Stop when you've
            # created a new state (i.e., an expansion occured) or are at a terminal state.
            node_exploring = self.current_node
            expansion_occured = False
            while True:
                if node_exploring.is_terminal or expansion_occured:
                    break
                best_edge = self.puct(
                    node_exploring,
                    puct_explore_factor,
                    puct_noise_alpha,
                    puct_noise_influence,
                )

                # Expand if novel action
                if best_edge.child_node is None:
                    resulting_state = self.environment.transition_state(node_exploring.state, best_edge.move)
                    self.add_node(state=resulting_state, parent_edge=best_edge)
                    expansion_occured = True
                node_exploring = best_edge.child_node
            node_selected = node_exploring

            # Rollouts
            # - just kidding!

            # Backup
            # - Reward is the selected node's state value unless it's a terminal node that has an
            #   objective reward.
            rewards = node_selected.values

            backing_node = node_selected
            while backing_node is not self.current_node:
                # XXX: Do you backup past current position?
                # XXX: Do you increment visit here or before?
                backing_node.parent_edge.visit_count += 1
                for i, val in enumerate(backing_node.parent_edge.reward_totals):
                    backing_node.parent_edge.reward_totals[i] = val + rewards[i]
                backing_node = backing_node.parent_edge.parent_node

                # This is the parent of the edges whose visit_counts were incremented above.
                # Increment the visit_count to the node (sum of all visits to edges) to prevent
                # making puct calculation have to do a first-pass to calculate visit count
                # XXX: Double check this is right. assert same.  Is it adding too many visits to
                #      root consideration node?
                backing_node.visit_count += 1

        if settings.VERBOSITY >= 1:
            steps_per_sec = round(steps / (time.time() - st_time), 1)
            print("MCTS", steps, "considerations,", steps_per_sec, "per sec")

    def display_best_move(self):
        print("\n{:<8}{:<8}{:<8}{:<8}".format("MOVE", "VISITS", "PRIOR", "P(WIN)"))
        for child_edge in self.current_node.child_edges:
            p_win = None if not child_edge.visit_count else round(child_edge.reward_totals[self.agent_num] / child_edge.visit_count, 3)
            color = "white"
            if (p_win or 0) > 0:
                color = "green"
            elif (p_win or 0) < 0:
                color = "red"

            prior = round(float(child_edge.prior_probability), 3)
            rprint("{:<8}{:<8}{:<8}[{}]{:<8}[/{}]".format(
                child_edge.move,
                child_edge.visit_count,
                prior,
                color,
                (p_win or "-"),
                color,
            ))

    def choose_best_move(self):
        # XXX: Add noise and exploration factor
        max_visited_edge = self.current_node.child_edges[0]
        for child_edge in self.current_node.child_edges:
            # XXX: Have noise tie-break
            if child_edge.visit_count > max_visited_edge.visit_count:
                max_visited_edge = child_edge
        if settings.VERBOSITY >= 2:
            self.display_best_move()
        return max_visited_edge.move

    def make_move(self):
        self.run_mcts_considerations(
            self.move_consideration_steps,
            self.move_consideration_time,
            puct_explore_factor=self.puct_explore_factor,
            puct_noise_alpha=self.puct_noise_alpha,
            puct_noise_influence=self.puct_noise_influence,
        )
        return self.choose_best_move()

    def record_replay(self, output_dir):
        data = self.replay_data()
        game_id = data["id"]
        agent_num = data["agent"]["agent_num"]

        # mkdir -p replay path
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_path = f"{output_dir}/{game_id}-{agent_num}.json"

        with open(output_path, 'w') as fout:
            fout.write(json.dumps(data))

        if settings.VERBOSITY >= 2:
            print("Saved replay:", output_path)

    def iter_replayed_game_tree_history(self):
        '''Walk the game tree as if you were replaying the game move by move'''
        node = self.game_tree
        for _, move in self.environment.event_history:
            yield (node, move)
            if move is None:
                break
            node = node.get_child_edge(move).child_node

    def replay_data(self):
        env = self.environment
        replay = []
        for game_tree_node, move in self.iter_replayed_game_tree_history():
            policy_info = []
            for child_edge in game_tree_node.child_edges:
                policy_info.append((
                    child_edge.move,
                    float(child_edge.prior_probability),
                    child_edge.visit_count,
                    child_edge.reward_totals[self.agent_num],
                ))

            replay.append(dict(
                state=game_tree_node.state.marshall(format="dict"),
                value=game_tree_node.values[self.agent_num],
                policy_info=policy_info,
                move=move,
            ))

        data = dict(
            id=env.id,
            name=env.get_name(),
            started_at=env.started_at,
            ended_at=env.ended_at,
            agent_nums=list(range(len(env.agents))),
            agent=dict(
                agent_num=self.agent_num,
                model_version=-1, # XXX Update model-specific info
            ),
            outcome=env.rewards(env.event_history[-1][0]), # state of last (s, a) event
            replay=replay,
        )
        # import pprint
        # pprint.pprint(data)
        return data
