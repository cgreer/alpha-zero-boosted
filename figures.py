from dataclasses import dataclass
from collections import defaultdict
import json
from typing import (
    List,
)

import matplotlib.pyplot as plt

from training_info import TrainingInfo
from gbdt_model import GBDTTrainingInfo
from paths import build_model_directory
from table_operations import min_aggregation, group_by
from collections import namedtuple


@dataclass
class EvalData:
    environment: str
    species: str
    generation: int
    model_type: str
    dataset: str
    metric: str
    iteration: int
    value: float


@dataclass
class EvalDataTable:
    rows: List[EvalData]

    @classmethod
    def build(cls, environment, species):
        ti = TrainingInfo.load(environment, species)
        generations = [x.generation_trained for x in ti.batches if x.generation_trained]

        rows = []
        for generation in generations:
            for model_type in ("value", "policy"):
                model_directory = build_model_directory(environment, species, generation)
                info_path = f"{model_directory}/{model_type}_model_training_info_{generation:06d}.json"
                info = GBDTTrainingInfo.load(info_path)
                for eval_stat in info.eval_stats:
                    rows.append(
                        EvalData(
                            environment=environment,
                            species=species,
                            generation=generation,
                            model_type=model_type,
                            dataset=eval_stat.dataset,
                            metric=eval_stat.metric,
                            iteration=eval_stat.iteration,
                            value=eval_stat.value
                        )
                    )
        return cls(rows=rows)


def collate_generation_loss(environment, species):
    eval_data = EvalDataTable.build(environment, species)

    # Find the min loss per generation
    # - There are multiple datasets (training, valid_1) and can be multiple
    #   metrics per dataset (l1, l2, xentroopy).
    out_rows = min_aggregation(
        eval_data.rows,
        key_fxn=lambda x: (
            x.environment,
            x.species,
            x.model_type,
            x.dataset,
            x.generation,
            x.metric,
        ),
        value_fxn=lambda x: x.value,
    )
    BestLossData = namedtuple(
        "BestLoss",
        [
            "environment",
            "species",
            "model_type",
            "dataset",
            "generation",
            "metric",
            "value",
        ]
    )
    out_rows = [BestLossData(*x) for x in out_rows]

    # Make series
    # - species.model_type.dataset.metric
    figure_data = []
    for row in out_rows:
        if row.metric == "l1":
            continue
        if "valid" not in row.dataset:
            continue
        series = f"{row.species}.{row.model_type}.{row.dataset}.{row.metric}"
        figure_data.append((series, row.generation, row.value))
    return figure_data


def generation_loss_figure(data):
    # :data ~ [(series_key, dataset, generation, loss), ...]

    by_series = group_by(
        data,
        key_fxn=lambda x: x[0],
        values_fxn=lambda x: (x[1], x[2]),
    )

    # Setup figure
    style = "bmh"
    with plt.style.context(style):
        fig, ax = plt.subplots()  # Create a figure and an axes.
        for series_key, dps in by_series.items():
            print(dps)
            x = [x[0] for x in dps]
            y = [x[1] for x in dps]

            # ax.plot(x, y, 'o-', label=series_key)  # Plot some data on the axes.
            ax.errorbar(
                x,
                y,
                fmt='o-',
                yerr=0,
                capsize=0,
                label=series_key,
            )

        # Annotate figure
        # ax.set_xlabel('CPU-Hours')  # Add an x-label to the axes.
        ax.set_title("Training Loss")  # Add a title to the axes.
        ax.set_xlabel('Generation')  # Add an x-label to the axes.
        ax.set_ylabel('Loss')  # Add a y-label to the axes.
        ax.legend()  # Add a legend.
        plt.grid(True)
        plt.show()


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
            wall_clock_time_to_train += tbatch.self_play_end_time - tbatch.self_play_start_time
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


def training_efficiency_figure(
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
        ax.set_title("Training Efficiency")  # Add a title to the axes.
        ax.set_xlabel('Training time (hours)')  # Add an x-label to the axes.
        ax.set_ylabel('Skill (TrueSkill)')  # Add a y-label to the axes.
        ax.legend()  # Add a legend.
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    import sys
    command = sys.argv[1]
    if command == "training_efficiency":
        environment, tournament_results_path = sys.argv[2:]
        bfstats = collate_bot_figure_stats(environment, tournament_results_path)
        training_efficiency_figure(bfstats)
    elif command == "generation_loss":
        environment, species_list_str = sys.argv[2:]
        figure_data = []
        for species in species_list_str.split(','):
            figure_data.extend(collate_generation_loss(environment, species))
        generation_loss_figure(figure_data)
