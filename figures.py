from dataclasses import dataclass
from collections import defaultdict, namedtuple
import json
from typing import (
    List,
    Tuple,
)

import matplotlib.pyplot as plt

from training_info import TrainingInfo
from generation_info import GenerationInfo
from gbdt_model import GBDTTrainingInfo
from paths import build_model_directory, build_tournament_results_path
from table_operations import min_aggregation, group_by


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
class TournamentStats:
    environment: str
    species: str
    generation: int
    skill_level: float = None
    skill_sigma: float = None


def collate_training_efficiency_stats(environment, tournament_id):
    '''
    :environment ~ "connect_four"

    :tournament_id ~ "1593043900-gbdt_pcrR2-1-21"
    - This is just the basename of the path.
    '''
    bot_figure_stats = []
    tournament_results_path = build_tournament_results_path(tournament_id)
    results = json.loads(open(tournament_results_path, 'r').read())

    # Update all the batch info that needs to be updated
    for species in set(x[0] for x in results):
        training_info = TrainingInfo.load(environment, species)
        training_info.update_batch_stats()

    # Collect info for each bot in tournament
    for species, generation, skill_level, skill_sigma in results:
        gen_info = GenerationInfo.from_generation_info(
            environment,
            species,
            generation,
        )

        tourn_info = TournamentStats(
            environment,
            species,
            generation,
            skill_level,
            skill_sigma,
        )

        bot_figure_stats.append((gen_info, tourn_info))

    return bot_figure_stats


def training_efficiency_figure(
    data: List[
        Tuple[
            GenerationInfo,
            TournamentStats
        ]
    ]
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
        nrows = 2
        ncols = 2
        figsize = [6.4 * ncols, 4.8 * nrows]
        fig, axs = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=figsize,
            squeeze=False,
        )
        fig.suptitle("Training Efficiency", fontsize="xx-large")
        axs_flattened = []
        for r in range(nrows):
            for c in range(ncols):
                axs_flattened.append(axs[r][c])

        # Aggregate by species
        by_species = defaultdict(list)
        for x in data:
            by_species[x[0].species].append(x)

        # Plot lines
        for axis_num, metric in enumerate((
            'cpu_seconds_to_train',
            'mcts_considerations',
            'wall_clock_time_to_train',
        )):
            ax = axs_flattened[axis_num]
            for species, bot_infos in by_species.items():
                bot_infos.sort(key=lambda x: x[0].generation)
                gen_infos = [x[0] for x in bot_infos]
                tourn_infos = [x[1] for x in bot_infos]
                if metric == "cpu_seconds_to_train":
                    x = [gi.cpu_seconds_to_train / (60 * 60 * 24) for gi in gen_infos]
                elif metric == "mcts_considerations":
                    x = [gi.mcts_considerations for gi in gen_infos]
                elif metric == "wall_clock_time_to_train":
                    x = [(gi.wall_clock_time_to_train / 3600) for gi in gen_infos]

                y = [ti.skill_level for ti in tourn_infos]
                yerr = [ti.skill_sigma * 2 for ti in tourn_infos]

                # ax.plot(x, y, 'o-', label=species)  # Plot some data on the axes.
                ax.errorbar(
                    x,
                    y,
                    fmt='o-',
                    yerr=yerr,
                    capsize=3.0,
                    label=species,
                )

                '''
                gens = [bi.generation for bi in bis]
                xys = list(zip(x, y))
                for xy, gen in zip(xys, gens):
                    ax.annotate(
                        f'{gen}',
                        xy=xy,
                        xytext=(1, 1),
                        # arrowprops=dict(facecolor='black', shrink=0.05)
                    )
                '''

            if metric == "cpu_seconds_to_train":
                xlabel = "Training Time (CPU-days)"
            elif metric == "mcts_considerations":
                xlabel = "MCTS Considerations"
            elif metric == "wall_clock_time_to_train":
                xlabel = "Training Time (Hours)"

            # Annotate figure
            # ax.set_title("Training Efficiency")  # Add a title to the axes.
            ax.set_xlabel(xlabel)  # Add an x-label to the axes.
            ax.set_ylabel('Skill (TrueSkill)')  # Add a y-label to the axes.
            ax.legend()  # Add a legend.
        plt.grid(True)
        plt.show()


class GenerationTimings:

    def build(self, *args):
        data = self.collect_figure_data(*args)
        self.build_figure(data)

    def collect_figure_data(
        self,
        environment: str,
        species_list: List[str],
    ):
        data = [] # (species, training_time, generation_number)
        # Update all the batch info that needs to be updated
        for species in species_list:
            training_info = TrainingInfo.load(environment, species)
            training_info.update_batch_stats()

            for generation in range(1, training_info.current_self_play_generation()):
                gen_info = GenerationInfo.from_generation_info(
                    environment,
                    species,
                    generation,
                )

                data.append((species, gen_info.cpu_seconds_to_train, generation))

        return data

    def build_figure(self, data):

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
            ax.set_title("Self-play Times")  # Add a title to the axes.
            ax.set_xlabel('Time')  # Add an x-label to the axes.
            ax.set_ylabel('Generation')  # Add a y-label to the axes.
            ax.legend()  # Add a legend.
            plt.grid(True)
            plt.show()


if __name__ == "__main__":
    import sys
    command = sys.argv[1]
    if command == "training_efficiency":
        environment, tournament_id = sys.argv[2:]
        bfstats = collate_training_efficiency_stats(environment, tournament_id)
        training_efficiency_figure(bfstats)
    elif command == "generation_loss":
        environment, species_list_str = sys.argv[2:]
        figure_data = []
        for species in species_list_str.split(','):
            figure_data.extend(collate_generation_loss(environment, species))
        generation_loss_figure(figure_data)
    elif command == "generation_timing":
        environment, species_list_str = sys.argv[2:]
        species_list = species_list_str.split(',')
        fig = GenerationTimings()
        fig.build(environment, species_list)
