from dataclasses import dataclass
from typing import Tuple, Dict, Set, Any
import environment
# from collections import deque


@dataclass(frozen=True)
class State:
    p1_position: Tuple[int]
    p2_position: Tuple[int]
    p1_wall_count: int
    p2_wall_count: int
    playable_walls: Tuple[Any]
    blocked_passages: Set[Any] # XXX: Should be an array of arrays?
    whose_move: int # {0, 1} # Note: 0-based!


@dataclass()
class Action:
    type: str
    x: int
    y: int
    is_vertical: str

    @classmethod
    def from_str(self, action_str):
        # m43, w00h, w88v

        is_vertical = False
        if len(action_str) >= 4:
            if action_str[3] == "v":
                is_vertical = True

        return Action(
            type="move" if action_str[0] == "m" else "wall",
            x=int(action_str[1]),
            y=int(action_str[2]),
            is_vertical=is_vertical,
        )


def victory_distance(initial_x, initial_y, blocked_passages, winning_column):
    # Doesn't take into account hops.  How could you?

    visited = [[False] * 9 for _ in range(9)]
    visited[initial_x][initial_y] = True
    queue = [(initial_x, initial_y, 0)]

    # Biased towards going north/south first. Note that the last element is what will be searched
    # first for the dfs, so for player 1 north is the last element to bias heading in that
    # direction.
    adjacent_deltas = ((0, 1), (1, 0), (-1, 0), (0, -1))
    if winning_column == 8:
        adjacent_deltas = ((0, -1), (-1, 0), (1, 0), (0, 1))

    while queue:
        x, y, distance = queue.pop()
        if y == winning_column:
            return distance

        adjacent_distance = distance + 1 # XXX: better to just add in loop?
        for dx, dy in adjacent_deltas:
            adjacent_x = x + dx
            adjacent_y = y + dy
            if visited[adjacent_x][adjacent_y]:
                continue
            # XXX: better to store adjacent local vars?
            if (x, y, adjacent_x, adjacent_y) in blocked_passages:
                continue
            queue.append((adjacent_x, adjacent_y, adjacent_distance))
            visited[adjacent_x][adjacent_y] = True
    return -1


def victory_distance_2(initial_x, initial_y, blocked_passages, winning_column):
    # Doesn't take into account hops.  How could you?

    visited = {(initial_x, initial_y): 0}
    queue = [(initial_x, initial_y, 0)]

    # Biased towards going north/south first. Note that the last element is what will be searched
    # first for the dfs, so for player 1 north is the last element to bias heading in that
    # direction.
    adjacent_deltas = ((0, 1), (1, 0), (-1, 0), (0, -1))
    if winning_column == 8:
        adjacent_deltas = ((0, -1), (-1, 0), (1, 0), (0, 1))

    while queue:
        x, y, distance = queue.pop()
        if y == winning_column:
            return distance, x, y, visited

        adjacent_distance = distance + 1
        for dx, dy in adjacent_deltas:
            adjacent_x = x + dx
            adjacent_y = y + dy
            if (adjacent_x, adjacent_y) in visited:
                continue
            if (x, y, adjacent_x, adjacent_y) in blocked_passages:
                continue
            queue.append((adjacent_x, adjacent_y, adjacent_distance))
            visited[(adjacent_x, adjacent_y)] = adjacent_distance
    return -1, -1, -1, visited


def victory_path(initial_x, initial_y, final_x, final_y, visited):
    adjacent_deltas = ((0, 1), (1, 0), (-1, 0), (0, -1))

    x = final_x
    y = final_y
    distance = visited[(x, y)]
    path_reversed = [(x, y, distance)]
    while True:
        x, y, distance = path_reversed[-1]
        if (x == initial_x) and (y == initial_y):
            break

        # Find adjacent square with next shortest distance to origin.
        distance_shortest = 100
        x_shortest = None
        y_shortest = None
        for dx, dy in adjacent_deltas:
            distance_next = visited.get((x + dx, y + dy), 100)
            if distance_next < distance_shortest:
                distance_shortest = distance_next
                x_shortest = x + dx
                y_shortest = y + dy

        path_reversed.append((x_shortest, y_shortest, distance_shortest))
    return path_reversed


def initial_blocked_passages():
    # Fill with edges of board
    blocked_passages = set()
    for x in range(9):
        blocked_passages.add((x, -1, x, 0))
        blocked_passages.add((x, 0, x, -1))
        blocked_passages.add((x, 8, x, 9))
        blocked_passages.add((x, 9, x, 8))
    for y in range(9):
        blocked_passages.add((-1, y, 0, y))
        blocked_passages.add((0, y, -1, y))
        blocked_passages.add((8, y, 9, y))
        blocked_passages.add((9, y, 8, y))
    return blocked_passages


initial_x = 4
initial_y = 0
blocked_passages = initial_blocked_passages()
blocked_passages.add((4, 1, 4, 2))
blocked_passages.add((4, 2, 4, 1))
distance, final_x, final_y, visited = victory_distance_2(initial_x, initial_y, blocked_passages, 8)
vic_path = victory_path(initial_x, initial_y, final_x, final_y, visited)
print(vic_path)


@dataclass
class Environment(environment.Environment):

    def __post_init__(self):
        super().__post_init__()

    def add_agent(self, agent):
        super().add_agent(agent)

    def initial_state(self):
        return State(
            p1_position=(4, 0),
            p2_position=(4, 8),
            p1_wall_count=10,
            p2_wall_count=10,
            playable_walls=tuple(),
            blocked_passages=set(),
            whose_move=0,
        )

    def transition_state(self, state, move):
        '''
        Return next state of game history given play
        '''
        current_player_num = self.current_player(game_history)
        new_state = deepcopy(game_history[-1])

        if play[0] == "p":
            # Player is moving (or jumping)
            #  - Move player on board
            new_state.player_positions[current_player_num - 1] = (play[1], play[2])

        else:
            # Player is placing a wall
            # - decrement wall count, add wall, update filled edges
            wall_position, is_vertical = (play[1], play[2]), play[3]
            new_state.walls_left[current_player_num - 1] = new_state.walls_left[current_player_num - 1] - 1
            new_state.wall_positions[wall_position] = is_vertical
            update_filled_edges(new_state.filled_edges, wall_position, is_vertical)

        return new_state

        board = list(state.board)
        board[move] = state.whose_move + 1
        whose_move = 1 if state.whose_move == 0 else 0
        return State(board=tuple(board), whose_move=whose_move)

    def is_terminal(self, state):
        if state.p1_position[1] >= 8:
            return True
        elif state.p2_position[1] <= 0:
            return True
        return False

    def translate_human_input(self, human_input):
        return human_input

    def enumerate_actions(self, state):
        actions = []
        '''
        Return list of plays where each play is either:
            - wall ("w", 4, 4, True) (wall, pos_x, pos_y, is_vertical)
            - player ("p", 4, 3, 1) (player, pos_x, pos_y, player_num)
        '''
        current_state = state_history[-1]
        current_player_num = self.current_player(state_history)

        plays = []

        # Player can move (or jump) his piece along unobstructed paths
        player_position = current_state.player_positions[current_player_num - 1]
        other_player_num = 1 if current_player_num == 2 else 2
        other_player_position = current_state.player_positions[other_player_num - 1]
        for open_adjacent_position in unobstructed_adjacent_positions(player_position, current_state.filled_edges):
            # If an open adjacent square is owned by the other player, you can jump...
            if open_adjacent_position == other_player_position:
                # Find possible places to jump
                for open_position_around_opponent in unobstructed_adjacent_positions(other_player_position, current_state.filled_edges):
                    if open_position_around_opponent != player_position: # you can't jump to where you currently are
                        p = ("p", open_position_around_opponent[0], open_position_around_opponent[1], current_player_num)
                        plays.append(p)

            # If the opponent is not at that position, you can move there
            else:
                p = ("p", open_adjacent_position[0], open_adjacent_position[1], current_player_num)
                plays.append(p)

        # If this player has walls to place, enumerate possible valid wall positions
        if current_state.walls_left[current_player_num - 1] > 0:
            for wall_x, wall_y in WALL_MIDPOINTS:
                for is_vertical in (True, False):
                    if self.is_valid_wall_placement(current_state, (wall_x, wall_y), is_vertical=is_vertical):
                        p = ("w", wall_x, wall_y, is_vertical)
                        plays.append(p)
        return actions

    def rewards(self, state):
        return [0, 0]

    def text_display(self, state):
        pass

    def run(self):
        super().run()
