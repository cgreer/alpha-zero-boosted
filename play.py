import sys
from agents import HumanAgent, MCTSAgent
from self_play import configure_bot
from environment_registry import get_env_module

# script.py <environment> <p1 bot name> <p2 bot_name>
# script.py connect_four mcts_naive-6 human
environment, p1_bot_name, p2_bot_name = sys.argv[1:]


def configure_agent(bot_name):
    bot_species, bot_generation = bot_name.split("-")
    bot_generation = int(bot_generation)

    bot_settings = configure_bot(environment, bot_species, bot_generation)
    bot_settings["move_consideration_time"] = None
    bot_settings["move_consideration_steps"] = 1000

    return bot_settings


if p1_bot_name == "human":
    p1_agent_settings = {}
    P1_agent_class = HumanAgent
else:
    p1_agent_settings = configure_agent(p1_bot_name)
    P1_agent_class = MCTSAgent

if p2_bot_name == "human":
    p2_agent_settings = {}
    P2_agent_class = HumanAgent
else:
    p2_agent_settings = configure_agent(p2_bot_name)
    P2_agent_class = MCTSAgent

env_module = get_env_module(environment)

# Play N games
for i in range(5):
    environment = env_module.Environment()

    agent_1 = P1_agent_class(environment=environment, **p1_agent_settings)
    agent_2 = P2_agent_class(environment=environment, ** p2_agent_settings)

    environment.add_agent(agent_1)
    environment.add_agent(agent_2)

    environment.run()
