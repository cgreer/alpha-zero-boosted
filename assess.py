import sys
import time

import numpy

from environment_registry import get_env_module
from evaluation import Bot, Tournament
from species import get_species
from paths import build_tournament_results_path
import settings


def run_generation_ladder(
    environment_name,
    species_list, # [(species, low_gen, high_gen), ...]
    num_workers=1,
    entrants_per_species=7,
):
    bots = []
    for species, lowest_generation, highest_generation in species_list:
        num_entrants = entrants_per_species
        generations = [int(round(x)) for x in numpy.linspace(lowest_generation, highest_generation, num_entrants)]
        generations = list(set(generations))
        for i in generations:
            sp = get_species(species)
            Agent = sp.AgentClass
            agent_settings = sp.agent_settings(environment_name, i, play_setting="evaluation")
            print(f"Adding bot {species}-{i} to tourney")
            bots.append(
                Bot(
                    f"{species}-{i}",
                    Agent,
                    agent_settings,
                )
            )

    species_str = []
    for species, lg, hg in species_list:
        species_str.append(f"{species}-{lg}-{hg}")
    species_str.sort()
    species_str = "__".join(species_str)
    tournament_key = f"{round(time.time())}-{species_str}"
    results_path = build_tournament_results_path(tournament_key)

    env_class = get_env_module(environment_name)
    tournament = Tournament.setup(
        environment=env_class.Environment,
        bots=bots,
    )
    for i in range(300):
        tournament.ladder(num_rounds=1, num_workers=num_workers)
        tournament.display_results()
        tournament.save_results(results_path)


def run_faceoff(
    environment_name,
    species,
    generation,
    num_rounds,
    num_workers=1,
):
    env_class = get_env_module(environment_name)

    # The bot your testing and the current best bot
    bots = []
    for i in range(generation - 1, generation + 1):
        sp = get_species(species)
        Agent = sp.AgentClass
        agent_settings = sp.agent_settings(environment_name, i, play_setting="evaluation")
        bots.append(
            Bot(
                f"{species}-{i}",
                Agent,
                agent_settings,
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
    species_list_str = sys.argv[2]
    species_list = species_list_str.split(",")
    species_list = [x.split("-") for x in species_list]
    species_list = [(x[0], int(x[1].split("/")[0]), int(x[1].split("/")[1])) for x in species_list]

    run_generation_ladder(
        environment_name=sys.argv[1],
        species_list=species_list,
        num_workers=settings.NUM_CORES,
    )
