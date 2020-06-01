import sys

import numpy

from environment_registry import get_env_module
from evaluation import Bot, Tournament
from agents import MCTSAgent
from self_play import configure_bot # XXX: Refactor this out of self_play
import settings


def run_generation_ladder(
    environment_name,
    species,
    highest_generation,
    num_workers=1,
):
    num_entrants = 6
    generations = [int(round(x)) for x in numpy.linspace(1, highest_generation, num_entrants)]
    bots = []
    for i in generations:
        bot_settings = configure_bot(environment_name, species, i)
        bots.append(
            Bot(
                f"{species}-{i}",
                MCTSAgent,
                bot_settings,
            )
        )

    env_class = get_env_module(environment_name)
    tournament = Tournament.setup(
        environment=env_class.Environment,
        bots=bots,
    )
    # tournament.round_robin(num_games=200)
    for i in range(100):
        tournament.ladder(num_rounds=1, num_workers=num_workers)
        tournament.display_results()


def run_faceoff(
    environment_name,
    bot_species,
    bot_generation,
    num_rounds,
    num_workers=1,
):
    env_class = get_env_module(environment_name)

    # The bot your testing and the current best bot
    bots = []
    for i in range(bot_generation - 1, bot_generation + 1):
        bot_settings = configure_bot(environment_name, bot_species, i)
        bots.append(
            Bot(
                f"{bot_species}-{i}",
                MCTSAgent,
                bot_settings,
            )
        )

    # Run the faceoff
    tournament = Tournament.setup(
        environment=env_class.Environment,
        bots=bots,
    )
    for i in range(num_rounds):
        tournament.ladder(num_rounds=1, num_workers=num_workers) # 2 x 3 games each round
        tournament.display_results()

    # Return contender matchup
    contender_entrant = tournament.entrants[bots[-1].name]
    contender_matchup_info = contender_entrant.matchup_histories[bots[0].name]
    return contender_matchup_info


if __name__ == "__main__":
    # run_faceoff(ENVIRONMENT, BOT_SPECIES, BOT_GENERATION)
    run_generation_ladder(
        environment_name=sys.argv[1],
        species=sys.argv[2],
        highest_generation=int(sys.argv[3]),
        num_workers=settings.NUM_CORES,
    )
