import json
from dataclasses import dataclass, asdict
from typing import (
    Tuple,
    Set,
    List,
    Dict,
)
import environment
import numpy

SPEAK = 0

EARLY_STOPPING_POSITION = 90


class BootstrapValue:
    def predict(self, features):
        # :features ~ [(0, 1, ...), ...]
        values = []
        for feature_set in features:
            if feature_set[0] < 0.5:
                your_distance = (8.0 - feature_set[4]) / 8.0
                opponent_distance = feature_set[6] / 8.0
            else:
                your_distance = feature_set[6] / 8.0
                opponent_distance = (8.0 - feature_set[4]) / 8.0
            value = (opponent_distance - your_distance) * .25
            values.append(value)
        return values


def update_blocked_passages(
    blocked_passages,
    x,
    y,
    is_vertical,
    adding=True,
):
    '''
    Update :blocked_passages given that player places/removes a :is_vertical wall at (:x, :y).
    '''
    if adding:
        operation = blocked_passages.add
    else:
        operation = blocked_passages.remove

    if is_vertical:
        operation((x, y, x + 1, y))
        operation((x + 1, y, x, y))

        operation((x, y + 1, x + 1, y + 1))
        operation((x + 1, y + 1, x, y + 1))
    else:
        operation((x, y, x, y + 1))
        operation((x, y + 1, x, y))

        operation((x + 1, y, x + 1, y + 1))
        operation((x + 1, y + 1, x + 1, y))


def update_wall_states(
    vertical_wall_states,
    horizontal_wall_states,
    x,
    y,
    is_vertical,
):
    '''
    Update :vertical_wall_states and :horizontal_wall_states given that player places a :is_vertical
    wall at (:x, :y).
    '''
    if is_vertical:
        vertical_wall_states[x][y] = 1
        horizontal_wall_states[x][y] = 2

        if y + 1 <= 7:
            vertical_wall_states[x][y + 1] = 2
        if y - 1 >= 0:
            vertical_wall_states[x][y - 1] = 2
    else:
        horizontal_wall_states[x][y] = 1
        vertical_wall_states[x][y] = 2

        if x - 1 >= 0:
            horizontal_wall_states[x - 1][y] = 2
        if x + 1 <= 7:
            horizontal_wall_states[x + 1][y] = 2


def victory_distance(initial_x, initial_y, blocked_passages, winning_row):
    visited = [[False] * 9 for _ in range(9)]
    visited[initial_x][initial_y] = True
    queue = [(initial_x, initial_y, 0)]

    # Biased towards going north/south first. Note that the last element is what will be searched
    # first for the dfs, so for player 1 north is the last element to bias heading in that
    # direction.
    adjacent_deltas = ((0, 1), (1, 0), (-1, 0), (0, -1))
    if winning_row == 8:
        adjacent_deltas = ((0, -1), (-1, 0), (1, 0), (0, 1))

    while queue:
        x, y, distance = queue.pop()
        if y == winning_row:
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


def victory_distance_heavy(initial_x, initial_y, blocked_passages, winning_row):
    # Doesn't take into account jumps.  How could you?
    visited = {(initial_x, initial_y): 0}
    queue = [(initial_x, initial_y, 0)]

    # Biased towards going north/south first. Note that the last element is what will be searched
    # first for the dfs, so for player 1 north is the last element to bias heading in that
    # direction.
    adjacent_deltas = ((0, 1), (1, 0), (-1, 0), (0, -1))
    if winning_row == 8:
        adjacent_deltas = ((0, -1), (-1, 0), (1, 0), (0, 1))

    while queue:
        x, y, distance = queue.pop()
        if y == winning_row:
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


def victory_distance_lighter(initial_x, initial_y, blocked_passages, winning_row):
    # Doesn't take into account jumps.  How could you?
    visited = {(initial_x, initial_y): True}
    queue = [(initial_x, initial_y, 0)]

    # Biased towards going north/south first. Note that the last element is what will be searched
    # first for the dfs, so for player 1 north is the last element to bias heading in that
    # direction.
    adjacent_deltas = ((0, 1), (1, 0), (-1, 0), (0, -1))
    if winning_row == 8:
        adjacent_deltas = ((0, -1), (-1, 0), (1, 0), (0, 1))

    while queue:
        x, y, distance = queue.pop()
        if y == winning_row:
            return distance

        adjacent_distance = distance + 1
        for dx, dy in adjacent_deltas:
            adjacent_x = x + dx
            adjacent_y = y + dy
            if (adjacent_x, adjacent_y) in visited:
                continue
            if (x, y, adjacent_x, adjacent_y) in blocked_passages:
                continue
            queue.append((adjacent_x, adjacent_y, adjacent_distance))
            visited[(adjacent_x, adjacent_y)] = True
    return -1


def victory_path(initial_x, initial_y, final_x, final_y, visited, blocked_passages):
    adjacent_deltas = ((0, 1), (1, 0), (-1, 0), (0, -1))

    x = final_x
    y = final_y
    distance = visited[(x, y)]
    path_reversed = [(x, y, distance)]
    while True:
        x, y, distance = path_reversed[-1]
        if (x == initial_x) and (y == initial_y):
            break

        # Find traversable, adjacent square with next shortest distance to origin.
        distance_shortest = 100
        x_shortest = None
        y_shortest = None
        for dx, dy in adjacent_deltas:
            distance_next = visited.get((x + dx, y + dy), 100)
            if distance_next < distance_shortest:
                if (x, y, x + dx, y + dy) in blocked_passages:
                    continue
                distance_shortest = distance_next
                x_shortest = x + dx
                y_shortest = y + dy
        # print(x_shortest, y_shortest, distance_shortest)
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


def find_trap_walls(state):
    trap_walls = set() # (x, y, is_vert)

    # Traps are impossible if there aren't at least 2 walls placed.
    if state.p1_wall_count + state.p2_wall_count > 18:
        return trap_walls

    distance, final_x, final_y, visited = victory_distance_heavy(state.p1_x, state.p1_y, state.blocked_passages, 8)
    vic_path_reversed = victory_path(state.p1_x, state.p1_y, final_x, final_y, visited, state.blocked_passages)
    update_trap_walls(
        vic_path_reversed,
        trap_walls,
        state.blocked_passages,
        state.vertical_wall_states,
        state.horizontal_wall_states,
        8,
    )

    distance, final_x, final_y, visited = victory_distance_heavy(state.p2_x, state.p2_y, state.blocked_passages, 0)
    vic_path_reversed = victory_path(state.p2_x, state.p2_y, final_x, final_y, visited, state.blocked_passages)
    update_trap_walls(
        vic_path_reversed,
        trap_walls,
        state.blocked_passages,
        state.vertical_wall_states,
        state.horizontal_wall_states,
        0,
    )

    return trap_walls


def update_trap_walls(
    path_reversed,
    trap_walls,
    blocked_passages,
    vertical_wall_states,
    horizontal_wall_states,
    winning_row,
):
    '''
    For every placeable wall that could obstruct this path, check if placing that wall would prevent
    the player from getting to victory row.
    '''
    stop_index = len(path_reversed) - 1
    leading_cell_index = 0
    while leading_cell_index < stop_index:
        leading_x, leading_y, _ = path_reversed[leading_cell_index]
        lagging_x, lagging_y, _ = path_reversed[leading_cell_index + 1]
        for wall_x, wall_y, is_vertical in blocking_walls(
            lagging_x,
            lagging_y,
            leading_x,
            leading_y
        ):
            # blocking_walls doesn't bounds check walls that aren't possible.  Do that here.
            if (wall_x > 7) or (wall_y > 7):
                continue

            # Can be placed?
            if is_vertical:
                if vertical_wall_states[wall_x][wall_y] != 0:
                    continue
            else:
                if horizontal_wall_states[wall_x][wall_y] != 0:
                    continue

            # If we placed it, would it trap the player?
            # - Temporarilly update the blocked_passages that would occur if we added the wall.
            update_blocked_passages(blocked_passages, wall_x, wall_y, is_vertical, adding=True)
            if victory_distance_lighter(lagging_x, lagging_y, blocked_passages, winning_row) == -1:
                trap_walls.add((wall_x, wall_y, is_vertical))
            update_blocked_passages(blocked_passages, wall_x, wall_y, is_vertical, adding=False)
        leading_cell_index += 1


def blocking_walls(x1, y1, x2, y2):
    '''
    Given passage from (x1, y1) to (x2, y2), which walls could block that passage?
    '''
    if y2 > y1:
        return (x1 - 1, y1, False), (x1, y1, False)
    elif x2 < x1:
        return (x2, y1 - 1, True), (x2, y2, True)
    elif x2 > x1:
        return (x1, y1 - 1, True), (x1, y1, True)
    else:
        return (x1 - 1, y2, False), (x1, y2, False)


@dataclass
class State:
    whose_move: int # {0, 1} # Note: 0-based!
    position_num: int
    p1_x: int
    p1_y: int
    p2_x: int
    p2_y: int
    p1_wall_count: int
    p2_wall_count: int
    blocked_passages: Set[Tuple] # set([(pos1_x, pos1_y, pos2_x, pos2_y])
    vertical_wall_states: List[List[int]] # [8][8]int, [pos_x][pos_y]wall_state
    horizontal_wall_states: List[List[int]] # [8][8]int, [pos_x][pos_y]wall_state

    def marshall(self, format="dict"):
        # XXX: Convert back
        self.blocked_passages = tuple(self.blocked_passages)
        data = asdict(self)
        if format == "dict":
            return data
        elif format == "json":
            return json.dumps(data)
        else:
            raise KeyError(f"Unknown format: {format}")

    @classmethod
    def unmarshall(cls, data, format="dict"):
        if format == "dict":
            instance = cls(**data)
            instance.blocked_passages = set(tuple(x) for x in instance.blocked_passages)
            return instance


def generate_features(state, agents) -> numpy.array:
    # :agents ~ [0, 1]
    # :agents ~ [0, 1, 2]
    # :agents ~ [0, 1, 5]
    # Which player's pov, which player is moving, is POV's player moving?
    features = numpy.zeros((2, 137), dtype=numpy.float32)

    agent_0_features = features[0]
    agent_1_features = features[1]

    # Agent-specific features
    agent_0_features[0] = 0.0
    agent_1_features[0] = 1.0

    # Agent-shared features
    agent_0_features[1] = state.whose_move
    agent_0_features[2] = state.position_num
    agent_0_features[3] = state.p1_x
    agent_0_features[4] = state.p1_y
    agent_0_features[5] = state.p2_x
    agent_0_features[6] = state.p2_y
    agent_0_features[7] = state.p1_wall_count
    agent_0_features[8] = state.p2_wall_count
    i = 9
    for x in (0, 1, 2, 3, 4, 5, 6, 7):
        for y in (0, 1, 2, 3, 4, 5, 6, 7):
            if state.vertical_wall_states[x][y] == 1:
                agent_0_features[i] = 1
            if state.horizontal_wall_states[x][y] == 1:
                agent_0_features[i + 64] = 1
            i += 1

    # Copy over agent-shared features to other agent.
    agent_1_features[1:137] = agent_0_features[1:137]

    return features


@dataclass
class Environment(environment.Environment):
    action_info: List[Tuple] = None
    move_action_info: List[Tuple] = None
    jump_action_info: List[Tuple] = None
    vertical_wall_action_info: List[Tuple] = None
    horizontal_wall_action_info: List[Tuple] = None
    action_name_by_id: Dict[int, str] = None

    def __post_init__(self):
        super().__post_init__()

        # (movement_direction, wall_x, wall_y, is_vertical)
        # - :movement_direction is [0-7], see move_player
        self.action_info = []

        _, self.action_name_by_id = self.build_action_maps()

        self.move_action_info = [
            (0, 0, 1),
            (1, 1, 0),
            (2, 0, -1),
            (3, -1, 0),
        ]
        self.jump_action_info = [
            (4, 0, 1),
            (5, 1, 0),
            (6, 0, -1),
            (7, -1, 0),
        ]
        for info in self.move_action_info + self.jump_action_info:
            self.action_info.append((info[0], None, None, None))

        # Order the wall actions by y-axis
        # - Hopefully models will make use of that...
        self.vertical_wall_action_info = []
        self.horizontal_wall_action_info = []
        i = 8
        for y in range(8):
            for x in range(8):
                self.vertical_wall_action_info.append((i, x, y))
                self.action_info.append((None, x, y, True))
                i += 1

                self.horizontal_wall_action_info.append((i, x, y))
                self.action_info.append((None, x, y, False))
                i += 1

        # Cache action_ids by wall info for human input
        self.action_id_by_wall_info = {} # (x, y, is_vertical):action_id
        for i, info in enumerate(self.action_info):
            # Skip movements
            if info[0] is not None:
                continue
            self.action_id_by_wall_info[(info[1], info[2], info[3])] = i

        # Another view of action info used by transition_state
        # self.num_actions = sum([
            # len(self.move_action_info),
            # len(self.jump_action_info),
            # len(self.vertical_wall_states),
            # len(self.horizontal_wall_states),
        # ])

    def add_agent(self, agent):
        super().add_agent(agent)

    def get_name(self):
        return "quoridor"

    def initial_state(self):
        return State(
            whose_move=0,
            position_num=0,
            p1_x=4,
            p1_y=0,
            p2_x=4,
            p2_y=8,
            p1_wall_count=10,
            p2_wall_count=10,
            blocked_passages=initial_blocked_passages(),
            vertical_wall_states=[[0] * 8 for _ in range(8)],
            horizontal_wall_states=[[0] * 8 for _ in range(8)],
        )

    def move_player(self, state, movement_direction):
        '''
        :movement_direction [0, 3] are up/right/down/left simple movements
        :movement_direction [4, 7] are up/right/down/left jumps
        '''
        if state.whose_move == 0:
            mover_x = state.p1_x
            mover_y = state.p1_y
            opponent_x = state.p2_x
            opponent_y = state.p2_y
        else:
            mover_x = state.p2_x
            mover_y = state.p2_y
            opponent_x = state.p1_x
            opponent_y = state.p1_y

        # Simple movements (up, right, down, left)
        if movement_direction == 0:
            mover_y += 1
        elif movement_direction == 1:
            mover_x += 1
        elif movement_direction == 2:
            mover_y -= 1
        elif movement_direction == 3:
            mover_x -= 1

        # Jumps (up, right, down, left)
        elif movement_direction == 4:
            mover_x = opponent_x
            mover_y = opponent_y + 1
        elif movement_direction == 5:
            mover_x = opponent_x + 1
            mover_y = opponent_y
        elif movement_direction == 6:
            mover_x = opponent_x
            mover_y = opponent_y - 1
        elif movement_direction == 7:
            mover_x = opponent_x - 1
            mover_y = opponent_y
        else:
            raise KeyError(f"Unknown movement_direction: {movement_direction}")

        return mover_x, mover_y

    def transition_state(self, state, action_id):
        movement_direction, wall_pos_x, wall_pos_y, is_vertical = self.action_info[action_id]

        blocked_passages = set(state.blocked_passages)
        vertical_wall_states = [x[:] for x in state.vertical_wall_states]
        horizontal_wall_states = [x[:] for x in state.horizontal_wall_states]

        # Player is moving/jumping
        if movement_direction is not None:
            if state.whose_move == 0:
                p1_x, p1_y = self.move_player(state, movement_direction)
                p2_x = state.p2_x
                p2_y = state.p2_y
            else:
                p1_x = state.p1_x
                p1_y = state.p1_y
                p2_x, p2_y = self.move_player(state, movement_direction)

            # XXX: Technically you don't have to make a copy of the passages/walls.
            return State(
                whose_move=1 if state.whose_move == 0 else 0,
                position_num=state.position_num + 1,
                p1_x=p1_x,
                p1_y=p1_y,
                p2_x=p2_x,
                p2_y=p2_y,
                p1_wall_count=state.p1_wall_count,
                p2_wall_count=state.p2_wall_count,
                blocked_passages=blocked_passages,
                vertical_wall_states=vertical_wall_states,
                horizontal_wall_states=horizontal_wall_states,
            )

        # Player is placing a wall
        else:
            update_blocked_passages(blocked_passages, wall_pos_x, wall_pos_y, is_vertical)
            update_wall_states(
                vertical_wall_states,
                horizontal_wall_states,
                wall_pos_x,
                wall_pos_y,
                is_vertical,
            )

            return State(
                whose_move=1 if state.whose_move == 0 else 0,
                position_num=state.position_num + 1,
                p1_x=state.p1_x,
                p1_y=state.p1_y,
                p2_x=state.p2_x,
                p2_y=state.p2_y,
                p1_wall_count=state.p1_wall_count - 1 if state.whose_move == 0 else state.p1_wall_count,
                p2_wall_count=state.p2_wall_count - 1 if state.whose_move == 1 else state.p2_wall_count,
                blocked_passages=blocked_passages,
                vertical_wall_states=vertical_wall_states,
                horizontal_wall_states=horizontal_wall_states,
            )

    def is_terminal(self, state):
        if state.p1_y >= 8:
            return True
        elif state.p2_y <= 0:
            return True

        if state.position_num == EARLY_STOPPING_POSITION:
            return True

        return False

    def build_action_maps(self):
        action_id_by_name = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
            "ju": 4,
            "jr": 5,
            "jd": 6,
            "jl": 7,
        }
        i = 8
        for y in range(8):
            for x in range(8):
                name = f"{x}{y}v"
                action_id_by_name[name] = i
                i += 1

                name = f"{x}{y}h"
                action_id_by_name[name] = i
                i += 1

        # Make inverse
        action_name_by_id = {v: k for k, v in action_id_by_name.items()}

        return action_id_by_name, action_name_by_id

    def translate_human_input(self, human_input):
        # u, uj, 89v
        input_cleaned = human_input.strip().lower()

        # Movements
        if input_cleaned == "u":
            return 0
        elif input_cleaned == "r":
            return 1
        elif input_cleaned == "d":
            return 2
        elif input_cleaned == "l":
            return 3

        # Jumps
        elif input_cleaned == "ju":
            return 4
        elif input_cleaned == "jr":
            return 5
        elif input_cleaned == "jd":
            return 6
        elif input_cleaned == "jl":
            return 7

        # Walls
        else:
            x = int(input_cleaned[0])
            y = int(input_cleaned[1])
            is_vertical = True if input_cleaned[2] == "v" else False
            return self.action_id_by_wall_info[(x, y, is_vertical)]

    def all_possible_actions(self):
        # 4 moves, 4 jumps, 64 vertical walls, 64 horizontal
        # XXX: assert this matches up with other actions when you modify it else
        # the policy model will be messed up without you knowing.
        return tuple(range(8 + 64 + 64))

    def enumerate_actions(self, state):
        actions = []

        if state.whose_move == 0:
            mover_x = state.p1_x
            mover_y = state.p1_y
            mover_wall_count = state.p1_wall_count
            opponent_x = state.p2_x
            opponent_y = state.p2_y
        else:
            mover_x = state.p2_x
            mover_y = state.p2_y
            mover_wall_count = state.p2_wall_count
            opponent_x = state.p1_x
            opponent_y = state.p1_y

        # Player movements
        # - Player can move into adjacent cells not obstructed by walls or opponent
        # - If adjacent cell is occupied by opponent player, player can't move to that position,
        #   but instead can move to where the opponent player could move to (excluding player's
        #   position).
        for move_action_id, mdx, mdy in self.move_action_info:
            adjacent_x = mover_x + mdx
            adjacent_y = mover_y + mdy

            # Can't move there, passage is blocked.
            if (mover_x, mover_y, adjacent_x, adjacent_y) in state.blocked_passages:
                continue

            # Jump over opponent case.
            # - Can jump anywhere opponent could move, except where mover is.
            if (adjacent_x == opponent_x) and (adjacent_y == opponent_y):
                for jump_action_id, jdx, jdy in self.jump_action_info:
                    if (mover_x == (opponent_x + jdx)) and (mover_y == (opponent_y + jdy)):
                        continue
                    if (opponent_x, opponent_y, opponent_x + jdx, opponent_y + jdy) in state.blocked_passages:
                        continue
                    actions.append(jump_action_id)

            # Move up/right/down/left case
            else:
                actions.append(move_action_id)

        # Can't place any walls if you won't have any
        if mover_wall_count <= 0:
            return actions

        # Player can place walls
        #  - Player can never place a wall that traps itself or opponent so that they aren't able to
        #    get to their respective goal rows.
        trap_walls = find_trap_walls(state)
        for action_id, wall_x, wall_y in self.vertical_wall_action_info:
            if state.vertical_wall_states[wall_x][wall_y] == 0:
                if (wall_x, wall_y, True) in trap_walls:
                    continue
                actions.append(action_id)
        for action_id, wall_x, wall_y in self.horizontal_wall_action_info:
            if state.horizontal_wall_states[wall_x][wall_y] == 0:
                if (wall_x, wall_y, False) in trap_walls:
                    continue
                actions.append(action_id)
        return actions

    def rewards(self, state):
        if state.p1_y >= 8:
            return [1, -1]
        elif state.p2_y <= 0:
            return [-1, 1]
        return [0, 0]

    def text_display(self, state):
        '''
        · · · · · · · · · · · · · · · · · · ·
        ·   ·   ·   ·   ·   ·   ·   ·   ·   · 8
        · · · · · · · · · · · · · · · · · · ·
        ·   ·   ·   ·   ·   ·   ·   ·   ·   · 7
        · · · · · · ·---○---· · · · · · · · ·
        ·   ·   ·   ·   │ ● ·   ·   ·   ·   · 6
        · · · · · · · · ○ · · · · · · · · · ·
        ·   ·   ·   ·   │   ·   ·   ·   ·   · 5
        · · · · · · · · · · · · · · · · · · ·
        ·   ·   ·   ·   ·   ·   ·   ·   ·   · 4
        · · · · · · · · · · · · · · · · · · ·
        ·   ·   ·   ·   ·   ·   ·   ·   ·   · 3
        · · · · · · · · · · · · · · · · · · ·
        ·   ·   ·   ·   ·   ·   ·   ·   ·   · 2
        · · · · · · · · · · · · · · · · · · ·
        ·   ·   ·   ·   ·   ·   ·   ·   ·   · 1
        · · · · · · · · · · · · · · · · · · ·
        ·   ·   ·   ·   ·   ·   ·   ·   ·   · 0
        · · · · · · · · · · · · · · · · · · ·
          0   1   2   3   4   5   6   7   8
        '''
        board_cells = [[" "] * 19 for _ in range(37)]
        for x in range(37):
            for y in range(19):
                if x % 2 != 0:
                    continue
                if y % 2 == 0:
                    board_cells[x][y] = "·"
                else:
                    if x % 4 == 0:
                        board_cells[x][y] = "·"

        # Players
        board_cells[(4 * state.p1_x) + 2][(2 * state.p1_y) + 1] = "[bold green]●[/bold green]"
        board_cells[(4 * state.p2_x) + 2][(2 * state.p2_y) + 1] = "[bold red]●[/bold red]"

        # Walls
        for x in range(8):
            for y in range(8):
                dot_x = (4 * x) + 4
                dot_y = (2 * y) + 2
                if state.horizontal_wall_states[x][y] == 1:
                    board_cells[dot_x][dot_y] = "[bold yellow]◉[/bold yellow]"
                    for dc in (-3, -2, -1, 1, 2, 3):
                        wx = dot_x + dc
                        wy = dot_y
                        board_cells[wx][wy] = "[bold yellow]-[/bold yellow]"
                if state.vertical_wall_states[x][y] == 1:
                    board_cells[dot_x][dot_y] = "[bold yellow]◉[/bold yellow]"
                    for dc in (-1, 1):
                        wx = dot_x
                        wy = dot_y + dc
                        board_cells[wx][wy] = "[bold yellow]│[/bold yellow]"

        # Convert to string
        rows = []
        y_label = 8
        for y in range(18, -1, -1):
            row_string = ""
            for x in range(37):
                row_string += board_cells[x][y]
            if (y % 2) == 1:
                row_string += f" {y_label}"
                if y_label == 8:
                    row_string += f"  [bold red]# Walls:[/bold red] [white]{state.p2_wall_count}[/white]"
                if y_label == 0:
                    row_string += f"  [bold green]# Walls:[/bold green] [white]{state.p1_wall_count}[/white]"
                y_label -= 1
            rows.append(row_string)

        x_labels = [" "] * 37
        for x in range(9):
            x_labels[(4 * x) + 2] = f"{x}"
        rows.append("".join(x_labels))

        return "\n".join(rows)

    def reconstruct_position(self, agent_replay, position):
        return super().reconstruct_position(agent_replay, position)

    def setup(self):
        return super().setup()

    def run(self):
        return super().run()


def inspect_victory_path():
    initial_x = 4
    initial_y = 0
    blocked_passages = initial_blocked_passages()
    blocked_passages.add((4, 1, 4, 2))
    blocked_passages.add((4, 2, 4, 1))
    distance, final_x, final_y, visited = victory_distance_heavy(initial_x, initial_y, blocked_passages, 8)
    vic_path = victory_path(initial_x, initial_y, final_x, final_y, visited)
    print(vic_path)


def inspect_trap_walls():
    blocked_passages = initial_blocked_passages()
    vertical_wall_states = [[0] * 8 for _ in range(8)]
    horizontal_wall_states = [[0] * 8 for _ in range(8)]
    state = State(
        whose_move=0,
        p1_x=4,
        p1_y=0,
        p2_x=4,
        p2_y=8,
        p1_wall_count=10,
        p2_wall_count=10,
        blocked_passages=blocked_passages,
        vertical_wall_states=vertical_wall_states,
        horizontal_wall_states=horizontal_wall_states,
    )

    # Place wall at (3/4, 0) to see if it finds the trap at (3/4, 1)
    update_blocked_passages(state.blocked_passages, 3, 0, True)
    update_wall_states(state.vertical_wall_states, state.horizontal_wall_states, 3, 0, True)

    update_blocked_passages(state.blocked_passages, 4, 0, True)
    update_wall_states(state.vertical_wall_states, state.horizontal_wall_states, 4, 0, True)

    blocked_before = set(state.blocked_passages)
    tw = find_trap_walls(state)

    assert blocked_passages == blocked_before, "finding trap walls modified it"
    print("traps:", tw)

    from rich import print as rprint # noqa
    env = Environment()
    rprint(env.text_display(state))


def inspect_environment():
    import random
    from rich import print as rprint # noqa
    env = Environment()

    # Generate a bunch of random states to "fuzz out" any bugs
    for i in range(500):
        state = env.initial_state()
        history = []
        for a in range(50):
            try:
                actions = env.enumerate_actions(state)
                chosen_action = random.choice(actions)

                history.append(chosen_action)
                state = env.transition_state(state, chosen_action)
                if env.is_terminal(state):
                    break
            except Exception as e:
                global SPEAK
                SPEAK = 1

                failed_state = state

                print("\nfail", e)
                print("replaying")
                state = env.initial_state()
                for a in history:
                    print("\nBefore move")
                    rprint(env.text_display(state))
                    possible_actions = env.enumerate_actions(state)
                    for pa in possible_actions:
                        print('pa', env.action_info[pa])
                    state = env.transition_state(state, a)

                print("FINAL STATE")
                rprint(env.text_display(failed_state))
                raise

    rprint(env.text_display(state))


def inspect_environment_bug():
    import random # noqa
    from rich import print as rprint # noqa
    env = Environment()
    state = env.initial_state()

    actions_by_wall_info = {}
    for i, info in enumerate(env.action_info):
        if info[0] is not None:
            continue
        actions_by_wall_info[(info[1], info[2], info[3])] = i

    walls = [
        (1, 0, True),
        # (3, 0, True), # Trap wall
        (5, 0, True),
        (0, 2, True),
        (4, 2, True),
        (2, 3, True),
        (7, 3, True),
        (3, 4, True),
        (1, 7, True),
        (4, 7, True),

        (7, 0, False),
        (3, 1, False),
        (5, 1, False),
        (6, 3, False),
        (0, 4, False),
        (1, 5, False),
        (7, 5, False),
        (1, 6, False),
        (5, 7, False),
    ]
    actions = [0] # move p1 up once
    for wall in walls:
        actions.append(actions_by_wall_info[wall])

    for action in actions:
        state = env.transition_state(state, action)
    rprint(env.text_display(state))

    tw_action_id = actions_by_wall_info[(3, 0, True)]
    next_actions = env.enumerate_actions(state)
    print("Trap wall action id", tw_action_id)
    print("trap wall in allowable actions?", tw_action_id in next_actions)


# inspect_trap_walls()
# inspect_environment()
# inspect_environment_bug()
