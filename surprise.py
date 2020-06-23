import sys
from dataclasses import dataclass, asdict
import pprint

import numpy
from rich import print as rprint

from agent_replay import AgentReplay, display_best_moves
from environment_registry import get_env_module
from helpers import max_info
from replay import sample_batch_replay_files
from stats import describe_sample


MIN_SCORE = 0.3
MAX_STATE_SPAN = 2
ERROR_RANGE = [-10.0, 0.0]


@dataclass
class Surprise:
    initial_position_index: int
    final_position_index: int
    raw_error: float
    discounted_error: float
    score: float

    def marshall(self):
        return asdict(self)

    @classmethod
    def unmarshall(cls, data):
        return cls(**data)


def find_surprises(
    agent_replay,
    raw_error_range=None,
    discounted_error_range=None,
    max_state_span=5, # XXX: Clear up which span this is.
    only_terminal=False,
):
    expectation_positions = list(agent_replay.iter_expectation_positions())

    # What's the index of the last position in the replay?
    # - This is index in terms of expected position, not replay's positions
    final_game_position_index = len(expectation_positions) - 1
    surprises = []
    for i, initial_position in enumerate(expectation_positions):
        current_expectation = initial_position.expectation()
        states_from_terminal = final_game_position_index - i # XXX: right?
        for j, final_position in enumerate(expectation_positions[i + 1:]):
            if final_position.index - initial_position.index > max_state_span:
                break
            if only_terminal:
                if not final_position.is_terminal():
                    continue

            # Raw error is the difference in what the agent believed the E(final
            # score) between an initial_position and a downstream
            # final_position.
            raw_error = final_position.expectation() - current_expectation
            if raw_error_range:
                if not (raw_error_range[0] <= raw_error <= raw_error_range[1]):
                    continue

            # Discount expectations outcomes further away in time.
            # - "j" is number of positions with expectations away, not
            #   position states away.
            # XXX: I think we want position states actually.
            num_states_span = j + 1
            discounted_error = raw_error * (.9**num_states_span)

            # Trust assessments closer to final states
            discount_factor = .8 + (.2 * (.8**states_from_terminal))
            discounted_error *= discount_factor
            if discounted_error_range:
                if not (discounted_error_range[0] <= discounted_error <= discounted_error_range[1]):
                    continue

            surprises.append(
                Surprise(
                    initial_position.index,
                    final_position.index,
                    raw_error=raw_error,
                    discounted_error=discounted_error,
                    score=abs(discounted_error),
                )
            )

    surprises.sort(key=lambda x: x.score, reverse=True)
    return surprises


def sample_surprise(
    environment,
    species,
    batch,
    raw_error_range=None,
    discounted_error_range=None,
    max_state_span=MAX_STATE_SPAN,
    only_terminal=False,
):
    env_module = get_env_module(environment)
    rep_paths = sample_batch_replay_files(environment, species, batch)
    for replay_path in rep_paths:
        agent_replay = AgentReplay.from_path(replay_path, env_module.State)
        surprises = find_surprises(
            agent_replay=agent_replay,
            raw_error_range=raw_error_range,
            discounted_error_range=discounted_error_range,
            max_state_span=max_state_span,
            only_terminal=only_terminal,
        )
        if not surprises:
            continue
        analyze_surprise(agent_replay, replay_path, surprises[0])
        break


def analyze_surprise(agent_replay, replay_path, surprise):
    position_span = (surprise.initial_position_index, surprise.final_position_index)
    rprint("\n\n\n\n\n\n[bold green]######## SURPRISE INFO ########[\bold green]")
    rprint("Player", agent_replay.agent_settings.agent_num + 1, "Replay")
    rprint(f"replay", replay_path)
    rprint(f"surprise ID:", agent_replay.game_id, f"{position_span[0]}-{position_span[1]}")
    rprint("score:", round(surprise.score, 3))
    rprint("raw_error:", round(surprise.raw_error, 3))
    rprint("discounted_error:", round(surprise.discounted_error, 3))
    rprint("position span:", position_span[0], "to", position_span[1])
    rprint("[bold green]###############################[/bold green]\n")
    narrate_surprise(surprise, agent_replay)


def narrate_surprise(surprise, agent_replay):
    '''
    - Display surprise info
    - Starting from initial position, display every game state up to final
      state.
    '''
    env_module = get_env_module(agent_replay.environment_name)
    env = env_module.Environment()

    final_expectation = agent_replay.positions[surprise.final_position_index].expectation()
    initial_position_index = surprise.initial_position_index

    surprise_positions = agent_replay.positions[initial_position_index: surprise.final_position_index + 1]

    # Make position, error table
    rprint(f"\n{'POSITION':<15}{'EXPECTATION':<15}{'ERROR':<15}")
    for position in surprise_positions:
        is_my_move = agent_replay.agent_settings.agent_num == position.state.whose_move
        if not is_my_move:
            continue
        expectation = round(position.expectation(), 2)
        error_from_here = round(abs(final_expectation - expectation), 2)
        rprint(f"{position.index:<15}{expectation:<15}{error_from_here:<15}")

    for position in surprise_positions:
        player_num = position.state.whose_move + 1
        is_my_move = agent_replay.agent_settings.agent_num == position.state.whose_move

        position_progress = f"{position.index} / {surprise.final_position_index}"
        rprint(f"\n\n===== POSITION {position_progress}, PLAYER", player_num, "MOVING =====")
        current_state = position.state
        rprint(env.text_display(current_state))

        rprint("State Value:", round(position.value, 3))
        if is_my_move:
            display_best_moves(env, list(position.actions_considered.values()))

        action_id = position.chosen_action_id
        human_move = env.action_name_by_id.get(action_id)
        rprint(f"\nPlayer {player_num} chose [bold green]{human_move}[/bold green] ({action_id})")

        if is_my_move:
            expectation = position.expectation()
            expectation = round(expectation, 2)
            error_from_here = round(abs(final_expectation - expectation), 3)
            rprint(f"Player {player_num} expects {expectation}, error: {error_from_here}")
        else:
            expectation = None


def surprise_policy_overrides(self, surprise, only_opponents=True):
    '''
    Collect all actions after the initial position and including the last
    position.  The initial position's action isn't included because it is
    the action that lead to the initial expecation that is surprising.
    '''
    # {agent_num: move_id: override_value}
    policy_overrides = []
    for agent_num in self.agent_nums:
        policy_overrides.append({})

    # XXX: This initial_idx does not include surprised agent's initial move.
    #  - Depending on the context, this might not be what you want.
    initial_idx = surprise.initial_position_index + 1
    final_idx = surprise.final_position_index + 1
    for position in self.positions[initial_idx:final_idx]:
        if position.move is None:
            continue
        agent_num = position.state.whose_move
        policy_overrides[agent_num][position.move] = 1.0
    if only_opponents:
        policy_overrides = [None if i == self.agent_num else x for i, x in enumerate(policy_overrides)]
    return policy_overrides


def find_surprise(self, position_span):
    for surprise in self.iter_surprises():
        if surprise.initial_position_index != position_span[0]:
            continue
        if surprise.final_position_index != position_span[1]:
            continue
        return surprise


def distribution_difference_analysis():
    ntiles = [1, 5, 10, 50, 90, 95, 99]

    differences = []
    for i in range(30_000):
        diff = numpy.random.normal(0, .25) - numpy.random.normal(.1, .25)
        differences.append(abs(diff))
    p_ntiles = numpy.percentile(differences, ntiles)
    for t, pn in zip(ntiles, p_ntiles):
        print(f"{t:<10} {round(pn, 3):<10}")

    print("d", p_ntiles[4] - p_ntiles[2])

    differences = []
    for i in range(30_000):
        diff = numpy.random.normal(0, .25) - numpy.random.normal(.1, .0001)
        differences.append(abs(diff))
    p_ntiles = numpy.percentile(differences, ntiles)
    print()
    for t, pn in zip(ntiles, p_ntiles):
        print(f"{t:<10} {round(pn, 3):<10}")

    print("d", p_ntiles[4] - p_ntiles[2])


def batch_surprise_replays(environment, species, batch):
    rep_paths = sample_batch_replay_files(environment, species, batch)

    collected_surprises = []
    for rep in rep_paths:
        print("finding good surprise replays")
        ar = AgentReplay.from_path(rep)

        # Get best surprises for a game
        ar.annotate_surprises()
        ar.surprises.sort(key=lambda x: x.score, reverse=True)
        surprises = list(ar.iter_surprises(only_terminal=False))
        if not surprises:
            # Some have no terminal?
            print("No terminal?")
            continue

        best_surprise = surprises[0]
        if best_surprise.score < 0.4:
            continue
        collected_surprises.append(ar)
        if len(collected_surprises) >= 20:
            break

    # Play replays of surprise spans
    while True:
        for rep in collected_surprises:
            best_surprise = rep.surprises[0]
            rep.cli_play(
                best_surprise.initial_position_index,
                best_surprise.final_position_index,
            )


def replay_from_surprise(environment, bot, replay_id, position_span):
    ar = AgentReplay.from_path(replay_path)
    ar.annotate_surprises(min_score=MIN_SCORE, max_state_span=MAX_STATE_SPAN)
    for surprise in ar.iter_surprises(
        error_range=ERROR_RANGE,
    ):
        best_surprise = surprise
        break

    species, generation = bot.split("-")
    generation = int(generation)

    initial_position = best_surprise.initial_position_index
    policy_overrides = ar.surprise_policy_overrides(best_surprise)
    pprint.pprint(policy_overrides)

    ar.replay_game_from_position(
        initial_position,
        species,
        generation, # original bot
        3,
        {
            # "move_consideration_time": 0.2,
            "policy_overrides": policy_overrides,
        }
    )


def kl_divergence(p, q):
    return numpy.sum(numpy.where(p != 0, p * numpy.log(p / q), 0))


def hindsight_comparison(
    agent_replay,
    species,
    generation,
    ideal_generation,
    position,
    policy_overrides,
):
    original_policy = agent_replay.position_policy(position)

    # Get ideal policy
    # - Use a mature model and give it more consideratino time.
    _, agents = agent_replay.replay_game_from_position(
        position,
        species,
        ideal_generation,
        1, # num moves
        {
            "full_search_steps": 3000,
        }
    )
    agent_surprised = agents[agent_replay.agent_num]
    position_node = agent_surprised.current_node.parent_edge.parent_node
    ideal_policy = position_node.policy()

    # Get hindsight policy
    _, agents = agent_replay.replay_game_from_position(
        position,
        species,
        generation,
        1, # num moves
        {
            "full_search_steps": 500,
            "policy_overrides": policy_overrides,
        }
    )
    agent_surprised = agents[agent_replay.agent_num]
    position_node = agent_surprised.current_node.parent_edge.parent_node
    hindsight_policy = position_node.policy()

    # What's the relative entropy between the "ideal" distribution and
    # original/hindsight distribution?
    # XXX: Is ideal supposed to be "p" or "q" here for kl_divergence?
    original_div = kl_divergence(ideal_policy, original_policy)
    hindsight_div = kl_divergence(ideal_policy, hindsight_policy)

    ideal_policy_indexed = [(prob, i) for i, prob in ideal_policy]
    original_policy_indexed = [(prob, i) for i, prob in original_policy]
    hindsight_policy_indexed = [(prob, i) for i, prob in hindsight_policy]

    ideal_policy_indexed.sort(reverse=True)
    original_policy_indexed.sort(reverse=True)
    hindsight_policy_indexed.sort(reverse=True)

    ideal_max_i, _ = max_info(ideal_policy)
    original_max_i, _ = max_info(original_policy)
    hindsight_max_i, _ = max_info(hindsight_policy)

    original_map1 = 1 if original_policy_indexed[0][1] == ideal_policy_indexed[0][1] else 0
    hindsight_map1 = 1 if hindsight_policy_indexed[0][1] == ideal_policy_indexed[0][1] else 0

    # top_2 = set([x[1] for x in ideal_policy_indexed[:2]])
    # o_top_2 = set([x[1] for x in ideal_policy_indexed[:2]])
    # original_map2 = 0.0

    print(ideal_max_i, ideal_policy)
    print(original_max_i, original_policy)

    return (
        original_div,
        original_map1,
        hindsight_div,
        hindsight_map1,
    )


def hindsight_convergence(
    environment,
    species,
    batch,
):
    # Gather all the surprises
    rep_paths = sample_batch_replay_files(environment, species, batch)
    surprises = []
    orig_divs = []
    orig_maps = []

    hind_divs = []
    hind_maps = []
    for i, rep in enumerate(rep_paths):
        ar = AgentReplay.from_path(rep)
        ar.annotate_surprises(min_score=.2, max_state_span=MAX_STATE_SPAN)
        for surprise in ar.iter_surprises(
            error_range=ERROR_RANGE,
            only_terminal=False,
        ):
            surprises.append(surprise)
            policy_overrides = ar.surprise_policy_overrides(surprise, only_opponents=True)
            orig_div, orig_map, hind_div, hind_map = hindsight_comparison(
                ar,
                species,
                2,
                14,
                surprise.initial_position_index,
                policy_overrides,
            )
            orig_divs.append(orig_div)
            orig_maps.append(orig_map)

            hind_divs.append(hind_div)
            hind_maps.append(hind_map)

            # Only take highest surprise
            break

    rprint("\n\n[bold green]Original[/bold green]")
    rprint("\nDivergence:")
    describe_sample(orig_divs)
    rprint("\nMAP@1:")
    describe_sample(orig_maps)

    rprint("\n[bold green]Hindsight[/bold green]")
    rprint("\nDivergence:")
    describe_sample(hind_divs)
    rprint("\nMAP@1:")
    describe_sample(hind_maps)


if __name__ == "__main__":

    if sys.argv[1] == "hindsight_convergence_analysis":
        distribution_difference_analysis()
        hindsight_convergence("connect_four", "gbdt", 2)
        sys.exit()

    elif sys.argv[1] == "replay_game":
        # Replay a game from a replay position
        # connect_four gbdt-11 aa8fd68f3668-0 26-28
        environment, bot, replay_id, position_span, hindsight, num_moves = sys.argv[2:]
        species, generation = bot.split("-")
        generation = int(generation)
        ipos, fpos = position_span.split("-")
        position_span = [int(ipos), int(fpos)]
        hindsight = True if hindsight == "hind" else False
        num_moves = int(num_moves)

        replay_path = AgentReplay.find_path(environment, replay_id)

        ar = AgentReplay.from_path(replay_path)

        ar.annotate_surprises()
        policy_overrides = [None] * len(ar.agent_nums)
        if hindsight:
            surprise = ar.find_surprise(position_span=position_span)
            policy_overrides = ar.surprise_policy_overrides(surprise)

        ar.replay_game_from_position(
            position_span[0],
            species,
            generation, # original bot
            num_moves, # num moves
            {
                "full_search_steps": 800,
                # "move_consideration_time": 0.2,
                "policy_overrides": policy_overrides,
            }
        )

        sys.exit()

    elif sys.argv[1] == "analyze":
        # connect_four aa8fd68f3668-0 26-28
        environment, replay_id, position_span = sys.argv[2:]
        ipos, fpos = position_span.split("-")
        position_span = [int(ipos), int(fpos)]

        replay_path = AgentReplay.find_path(environment, replay_id)
        ar = AgentReplay.from_path(replay_path)
        ar.annotate_surprises()
        surprise = ar.find_surprise(position_span=position_span)
        analyze_surprise(ar, replay_path, surprise)

    elif sys.argv[1] == "hindsight_comparison":
        # connect_four aa8fd68f3668-0 26-28
        environment, replay_id, position_span = sys.argv[2:]
        ipos, fpos = position_span.split("-")
        position_span = [int(ipos), int(fpos)]
        species = "gbdt"

        replay_path = AgentReplay.find_path(environment, replay_id)
        ar = AgentReplay.from_path(replay_path)
        ar.annotate_surprises()

        surprise = ar.find_surprise(position_span=position_span)
        policy_overrides = ar.surprise_policy_overrides(surprise)

        hindsight_comparison(
            ar,
            species,
            1,
            14,
            position_span[0],
            policy_overrides,
        )

    elif sys.argv[1] == "sample_surprise":
        environment, species, batch = sys.argv[2:]
        batch = int(batch)
        sample_surprise(
            environment,
            species,
            batch,
            raw_error_range=[-2.0, -0.50],
        )
