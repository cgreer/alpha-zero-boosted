from agents import HumanAgent, MCTSAgent
import tictactoe
import connect_four
import intuition_model


#################
# Naive models
#################
naive_value_model = intuition_model.NaiveValue()
# naive_value_model.load("./ttt_naive_value.model")
naive_value_model.load("./c4_naive_value.model")

naive_policy_model = intuition_model.NaivePolicy()
# naive_policy_model.load("./ttt_naive_policy.model")
naive_policy_model.load("./c4_naive_policy.model")

####################
# Play
####################

mcts_agent_default_settings = dict(
    game_tree=None,
    current_node=None,
    feature_extractor=None,
    value_model=naive_value_model,
    policy_model=naive_policy_model,
    move_consideration_steps=800,
    move_consideration_time=.1,
    puct_explore_factor=1.0,
    puct_noise_alpha=0.2,
    puct_noise_influence=0.25,
)

env_module = connect_four
# env_module = tictactoe

for i in range(5):
    environment = env_module.Environment()

    mcts_agent_default_settings.update(feature_extractor=env_module.generate_features)

    mcts_agent_1 = MCTSAgent(environment=environment, **mcts_agent_default_settings)
    mcts_agent_2 = MCTSAgent(environment=environment, **mcts_agent_default_settings)
    human_agent_1 = HumanAgent(environment=environment)

    # Play
    environment.add_agent(mcts_agent_1)
    environment.add_agent(human_agent_1)
    environment.run()

    # mcts_agent_1.record_replay('./recorded_games/')
    # mcts_agent_2.record_replay('./recorded_games/')

'''
# Setup some environments (aka games)
environment = leftright.Environment(board_length=35)
environment = tictactoe.Environment()

# Setup some agents we can use
human_agent_1 = HumanAgent(environment=environment)
human_agent_2 = HumanAgent(environment=environment)

random_agent_1 = RandomAgent(environment=environment)
random_agent_2 = RandomAgent(environment=environment)
'''

##################
# Debugging
##################
'''
state = tictactoe.State(board=(0, 0, 1, 0, 2, 2, 0, 0, 1), whose_move=0)
print(state.marshall())

print(mcts_agent_1.game_data())

for node, move in mcts_agent_1.iter_replayed_game_tree_history():
    print(node.state)
    print(move)
'''

##################
# Debug a state
##################
'''
state = tictactoe.State(board=(0, 0, 1, 0, 2, 2, 0, 0, 1), whose_move=0)
print(state.marshall())
print(environment.text_display(state))

mcts_agent_1.setup(state)
move = mcts_agent_1.make_move()
print("Made move", move)

new_state = environment.transition_state(state, move)
print()
print(environment.text_display(new_state))

state = tictactoe.State(board=(0, 0, 1, 0, 2, 2, 0, 1, 1), whose_move=1)
print()
print(environment.text_display(state))

state = tictactoe.State(board=(0, 0, 1, 2, 2, 2, 0, 1, 1), whose_move=0)
print()
print(environment.text_display(state))

print(environment.find_winner(state))
'''
