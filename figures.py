from dataclasses import dataclass
from collections import defaultdict
import json
from typing import (
    List,
)

import matplotlib.pyplot as plt

from training_info import TrainingInfo


@dataclass
class BotFigureStats:
    environment: str
    species: str
    generation: int

    num_batches_to_train: int = None
    wall_clock_time_to_train: float = None
    cpu_seconds_to_train: float = None

    skill_level: float = None
    skill_sigma: float = None


def training_stats(environment, species, generation):
    training_info = TrainingInfo.load(environment, species)

    # Find batch
    num_batches_to_train = 0
    wall_clock_time_to_train = 0.0
    cpu_seconds_to_train = 0.0
    if generation > 1:
        for tbatch in training_info.batches:
            num_batches_to_train += 1
            wall_clock_time_to_train += tbatch.assessment_end_time - tbatch.self_play_start_time
            cpu_seconds_to_train += 0.0 # XXX: Update

            # This is the batch that trained this generation
            if tbatch.generation_trained == generation:
                break

    return dict(
        num_batches_to_train=num_batches_to_train,
        wall_clock_time_to_train=wall_clock_time_to_train,
        cpu_seconds_to_train=cpu_seconds_to_train,
    )


def collate_bot_figure_stats(environment, tournament_results_path):
    bot_figure_stats = []
    results = json.loads(open(tournament_results_path, 'r').read())
    for species, gen, skill_level, skill_sigma in results:
        data = dict(
            environment=environment,
            species=species,
            generation=gen,
            skill_level=skill_level,
            skill_sigma=skill_sigma,
        )
        data.update(training_stats(environment, species, gen))
        bot_figure_stats.append(BotFigureStats(**data))

    return bot_figure_stats


def training_efficiency(
    bot_figure_stats: List[BotFigureStats],
):
    '''
    Skill level v. cpu_time/energy/num_batches

    [(bot.cpu_seconds_to_train, bot.skill), ...]
    '''

    # Setup figure
    style = "fivethirtyeight"
    style = "dark_background"
    style = "bmh"
    with plt.style.context(style):
        fig, ax = plt.subplots()  # Create a figure and an axes.

        # Plot lines
        by_species = defaultdict(list)
        for bi in bot_figure_stats:
            by_species[bi.species].append(bi)
        for species, bis in by_species.items():
            bis.sort(key=lambda x: x.generation)
            gens = [bi.generation for bi in bis]
            x = [(bi.wall_clock_time_to_train / 3600) for bi in bis]
            y = [bi.skill_level for bi in bis]

            # ax.plot(x, y, 'o-', label=species)  # Plot some data on the axes.
            ax.errorbar(
                x,
                y,
                fmt='o-',
                yerr=bi.skill_sigma * 2,
                capsize=3.0,
                label=species,
            )

            xys = list(zip(x, y))
            for xy, gen in zip(xys, gens):
                ax.annotate(
                    f'{gen}',
                    xy=xy,
                    xytext=(1, 1),
                    # arrowprops=dict(facecolor='black', shrink=0.05)
                )

        # Annotate figure
        # ax.set_xlabel('CPU-Hours')  # Add an x-label to the axes.
        ax.set_title("Training Efficiency")  # Add a title to the axes.
        ax.set_xlabel('Training time (hours)')  # Add an x-label to the axes.
        ax.set_ylabel('Skill (TrueSkill)')  # Add a y-label to the axes.
        ax.legend()  # Add a legend.
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    import sys
    bfstats = collate_bot_figure_stats("connect_four", sys.argv[1])
    training_efficiency(bfstats)
