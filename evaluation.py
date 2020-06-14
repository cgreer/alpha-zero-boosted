from dataclasses import dataclass
from itertools import combinations
import json
import random
import typing
from multiprocessing import Pool

import numpy
from rich import print as rprint
from trueskill import Rating, quality_1vs1, rate_1vs1

from agent_configuration import configure_agent
from environment_registry import get_env_module
from paths import full_path_mkdir_p
from training_samples import fast_deterministic_hash


def run_game_worker(args):
    # :matchup_info ~ [(bot_1_species, bot_1_generation), ...]
    environment_name, matchup_info = args

    env_module = get_env_module(environment_name)
    environment = env_module.Environment()

    bot_1_species, bot_1_generation = matchup_info[0]
    bot_2_species, bot_2_generation = matchup_info[1]

    Agent1, agent_1_settings = configure_agent(
        environment_name,
        bot_1_species,
        bot_1_generation,
        play_setting="evaluation",
    )
    agent_1 = Agent1(environment=environment, **agent_1_settings)

    Agent2, agent_2_settings = configure_agent(
        environment_name,
        bot_2_species,
        bot_2_generation,
        play_setting="evaluation",
    )
    agent_2 = Agent2(environment=environment, **agent_2_settings)

    environment.add_agent(agent_1)
    environment.add_agent(agent_2)

    outcomes, _ = environment.run()

    return (matchup_info, outcomes)


@dataclass
class MatchupHistory:
    opponent: str
    wins: int = 0
    draws: int = 0
    losses: int = 0

    def games_played(self):
        return self.wins + self.draws + self.losses

    def win_rate(self, draw_weight=0.0):
        return (self.wins + (draw_weight * self.draws)) / self.games_played()

    def loss_rate(self):
        return self.losses / self.games_played()

    def draw_rate(self):
        return self.draws / self.games_played()

    def record(self):
        pct_record = f"{int(round(self.win_rate() * 100, 0))}%-{int(round(self.loss_rate() * 100, 0))}%-{int(round(self.draw_rate() * 100, 0))}%"
        return f"{self.wins}-{self.losses}-{self.draws}  {pct_record}"

    def handle_outcome(self, outcome):
        if outcome == 1:
            self.wins += 1
        elif outcome == 0:
            self.draws += 1
        elif outcome == -1:
            self.losses += 1
        else:
            raise KeyError(f"Unhandleable outcome: {outcome}")


@dataclass
class Entrant:
    bot: typing.Any
    skill_rating: typing.Any = None
    matchup_histories: typing.Any = None

    def __post_init__(self):
        self.skill_rating = Rating()
        self.matchup_histories = {}

    def handle_outcome(self, opponent_entrant, outcome):
        # Update skill rating
        # XXX: Todo

        # Update opponent match history stats
        opponent_name = opponent_entrant.bot.name
        if opponent_name not in self.matchup_histories:
            self.matchup_histories[opponent_name] = MatchupHistory(opponent_name)
        self.matchup_histories[opponent_name].handle_outcome(outcome)


@dataclass
class Bot:
    name: str
    agent_class: typing.Any
    agent_settings: typing.Any


@dataclass
class Tournament:
    entrants: typing.Dict[str, typing.Any]
    environment: typing.Any

    @classmethod
    def setup(cls, environment, bots):
        '''
        :environment is environment class used for game
        :bots is list of Bot that will compete in tournament
        '''
        entrants = {}
        for bot in bots:
            entrants[bot.name] = Entrant(bot)

        return cls(
            entrants=entrants,
            environment=environment,
        )

    def play_single_game(self, entrant_1, entrant_2):
        print("playing game", entrant_1.bot.name, entrant_2.bot.name)
        env = self.environment()

        agent_1 = entrant_1.bot.agent_class(environment=env, **entrant_1.bot.agent_settings)
        agent_2 = entrant_2.bot.agent_class(environment=env, **entrant_2.bot.agent_settings)

        env.add_agent(agent_1)
        env.add_agent(agent_2)

        outcomes, _ = env.run()
        return outcomes

    def head_to_head(self, num_games=100):
        # H2H is just a specific round robin case
        assert len(self.entrants) == 2, "H2H only works with 2 entrants"
        self.round_robin(num_games=num_games)

    def round_robin(self, num_games=100):
        entrants = list(self.entrants.values())
        matchups = list(combinations(entrants, 2))
        for matchup in matchups:
            for i, game_num in enumerate(range(num_games)):
                # Take turns being p1/p2
                players = [matchup[0], matchup[1]]
                if i % 2 == 0:
                    players = [matchup[1], matchup[0]]

                outcomes = self.play_single_game(*players)

                self.handle_game_outcome(players, outcomes)

    def handle_game_outcome(self, players, outcomes):
        # :players ~ [entrant_1, entrant_2]

        # Update skill rating
        # - First player passed to rate_1vs1 is the winner
        p1 = players[0]
        p2 = players[1]
        if outcomes[0] == 0:
            # Draw, order doesn't matter
            p1_new, p2_new = rate_1vs1(p1.skill_rating, p2.skill_rating, drawn=True)
        elif outcomes[0] == 1:
            # P1 won
            p1_new, p2_new = rate_1vs1(p1.skill_rating, p2.skill_rating)
        else:
            # P2 won
            p2_new, p1_new = rate_1vs1(p2.skill_rating, p1.skill_rating)
        p1.skill_rating = p1_new
        p2.skill_rating = p2_new

        # Update stats
        for entrant, outcome in zip(players, outcomes):
            opponent_entrant = [p for p in players if p != entrant][0]
            entrant.handle_outcome(opponent_entrant, outcome)

    def run_games(self, matchups, num_workers):
        '''
        Don't make pool the len(matchups) because it could cause bots to share
        cpu time.
        '''
        environment_name = self.environment().get_name()

        worker_args = []
        for matchup in matchups:
            bot_1_name = matchup[0].bot.name
            bot_1_species, bot_1_generation = bot_1_name.split("-")
            bot_1_generation = int(bot_1_generation)

            bot_2_name = matchup[1].bot.name
            bot_2_species, bot_2_generation = bot_2_name.split("-")
            bot_2_generation = int(bot_2_generation)

            matchup_info = [
                (bot_1_species, bot_1_generation),
                (bot_2_species, bot_2_generation),
            ]
            random.shuffle(matchup_info)

            worker_args.append(
                (
                    environment_name,
                    matchup_info,
                )
            )
        with Pool(num_workers) as p:
            results = p.map(run_game_worker, worker_args)

        return results

    def calculate_matchup_probabilities(self):
        # Calculate [P(matchup), ...] for every entrant.
        matchup_probs_by_entrant = {} # name: ([opp_name...], [matchup_prob...])
        for name, entrant in self.entrants.items():
            opp_names = []
            p_draws = []

            # Get P(draw | opp) for every opp
            for opp_name, opp_entrant in self.entrants.items():
                if name == opp_name:
                    continue
                opp_names.append(opp_name)
                p_draw = quality_1vs1(entrant.skill_rating, opp_entrant.skill_rating)
                p_draws.append(p_draw)

            # Normalize by sum(p_draw)
            p_draw_sum = sum(p_draws)
            matchup_probs = [x / p_draw_sum for x in p_draws]

            # Stash
            matchup_probs_by_entrant[name] = (opp_names, matchup_probs)
        return matchup_probs_by_entrant

    def matchmake(self, num_games):
        # Matchmaker, matchmaker, make me a match
        matchup_probs_by_entrant = self.calculate_matchup_probabilities()
        matchups = []
        while len(matchups) < num_games:
            # Select a random entrant to matchmake for.
            entrants = list(self.entrants.items())
            entrants.sort(key=lambda x: random.random())
            name, entrant = entrants[0]

            # Sample a game based on P(draw | opponent)
            opp_names, opp_matchup_probs = matchup_probs_by_entrant[name]
            opp_name = numpy.random.choice(
                opp_names,
                size=1,
                replace=True,
                p=opp_matchup_probs,
            )[0]
            matchups.append([entrant, self.entrants[opp_name]])
        return matchups

    def ladder(self, num_rounds, num_workers=1):
        '''
        Each round every entrant plays N games with opponents. The games are
        selected so that your P(match | opponent) is proportional to your P(draw
        | opponent)
        '''
        for round_num in range(num_rounds):
            matchups = self.matchmake(num_workers)
            results = self.run_games(matchups, num_workers)
            for matchup_info, outcomes in results:
                matchup_entrants = []
                for bot_species, bot_generation in matchup_info:
                    bot_name = f"{bot_species}-{bot_generation}"

                    # XXX: Tired... isn't this just a lookup?
                    for en in self.entrants.values():
                        if en.bot.name == bot_name:
                            matchup_entrants.append(en)
                            break
                    else:
                        raise KeyError(f"Couldn't find the bot: {bot_name}")
                self.handle_game_outcome(matchup_entrants, outcomes)

    def save_results(self, output_path):
        results = []
        for entrant in self.entrants.values():
            species, generation = entrant.bot.name.split("-")
            generation = int(generation)
            results.append(
                (
                    species,
                    generation,
                    entrant.skill_rating.mu,
                    entrant.skill_rating.sigma,
                )
            )

        full_path_mkdir_p(output_path)
        with open(output_path, 'w') as f:
            f.write(json.dumps(results))
        print(f"\nSaved results to: {output_path}")

    def display_results(self):
        # Print table of win rates (order by descending)
        table_contents = []
        for entrant in self.entrants.values():
            for other_entrant in self.entrants.values():
                if other_entrant == entrant:
                    continue
                vs_str = f"{entrant.bot.name:<15} v. {other_entrant.bot.name:<15}"
                if other_entrant.bot.name not in entrant.matchup_histories:
                    record = "0-0-0"
                else:
                    record = entrant.matchup_histories[other_entrant.bot.name].record()

                sort_key = (entrant.bot.name.split("-")[0], int(entrant.bot.name.split("-")[1]))
                table_contents.append((vs_str, record, sort_key))

        # Order by name
        print()
        print()
        print("{:<60}{:>30}".format("BOT", "RECORD"))
        table_contents.sort(key=lambda x: x[2])
        for vs_str, record, sort_key in table_contents:
            color = fast_deterministic_hash(f"{sort_key}") % 256
            row = "{:<60}{:>30}".format(vs_str, record)
            rprint(f"[{color}]{row}[/{color}]")

        # Print table of skill_rating (order by descending skill)
        table_contents = []
        for entrant in self.entrants.values():
            table_contents.append((entrant.bot.name, entrant.skill_rating.mu, entrant.skill_rating.sigma))
        print()
        print("{:<30}{:>30}{:>30}".format("BOT", "SKILL", "SIGMA"))
        table_contents.sort(key=lambda x: x[1], reverse=True)
        for name, skill, sigma in table_contents:
            print("{:<30}{:>30}{:>30}".format(name, round(skill, 2), round(sigma, 2)))
