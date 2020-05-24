import sys
from agents import RandomAgent, MCTSAgent, UnopinionatedValue, uniform_policy
import tictactoe
import intuition_model
import copy
from collections import defaultdict

# build TTT naive value model
replay_directory = sys.argv[1]
samples = list(intuition_model.generate_training_samples(replay_directory, tictactoe.State, tictactoe.generate_features))
naive_value_model = intuition_model.NaiveValue.from_training_samples(samples, tictactoe.generate_features)

unopinionated_value = UnopinionatedValue()

mcts_base_settings = dict(
    game_tree=None,
    current_node=None,
    policy_model=uniform_policy,
    move_consideration_steps=100,
    move_consideration_time=None,
    puct_explore_factor=1.0,
    puct_noise_alpha=0.2,
    puct_noise_influence=0.25,
)

# XXX: Better way to copy?
mcts_agent_unopinionated_settings = copy.deepcopy(mcts_base_settings)
mcts_agent_unopinionated_settings["value_model"] = unopinionated_value

mcts_agent_naive_settings = copy.deepcopy(mcts_base_settings)
mcts_agent_naive_settings["value_model"] = naive_value_model
mcts_agent_naive_settings["move_consideration_steps"] = 100

entrants = [
    # ("random", RandomAgent, {}),
    ("unopinionated_mcts", MCTSAgent, mcts_agent_unopinionated_settings),
    ("naive_mcts", MCTSAgent, mcts_agent_naive_settings),
]

matchups = []
for a in entrants:
    for b in entrants:
        if a == b:
            continue
        matchups.append((a, b))

matchup_games = defaultdict(int)
matchup_wins = defaultdict(int)
num_games = 200
for agent_info_1, agent_info_2 in matchups:
    for i in range(num_games):
        name_1, class_1, settings_1 = agent_info_1
        name_2, class_2, settings_2 = agent_info_2

        environment = tictactoe.Environment()

        if i % 2 == 0:
            agent_1 = class_1(environment=environment, **settings_1)
            agent_1_name = name_1

            agent_2 = class_2(environment=environment, **settings_2)
            agent_2_name = name_2
        else:
            agent_1 = class_2(environment=environment, **settings_2)
            agent_1_name = name_2

            agent_2 = class_1(environment=environment, **settings_1)
            agent_2_name = name_1

        environment.add_agent(agent_1)
        environment.add_agent(agent_2)

        outcome = environment.run()
        is_draw = True
        for val in outcome:
            if val != 0:
                is_draw = False
        if is_draw:
            continue

        outcome_names = [(outcome[0], agent_1_name), (outcome[1], agent_2_name)]
        outcome_names.sort(key=lambda x: x[1])
        win = 1 if outcome_names[0][0] == 1 else 0

        key = " v. ".join([x[1] for x in outcome_names])
        matchup_games[key] = matchup_games[key] + 1
        matchup_wins[key] = matchup_wins[key] + win

for matchup_name in matchup_games.keys():
    num_games = matchup_games[matchup_name]
    num_wins = matchup_wins[matchup_name]
    print(matchup_name, "win rate:", round(num_wins / num_games, 2), "num non-draw games:", num_games)
