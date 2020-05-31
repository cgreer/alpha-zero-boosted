from environment_registry import get_env_module
from evaluation import Bot, Tournament
from agents import RandomAgent, MCTSAgent
from self_play import configure_bot # XXX: Refactor this out of self_play


def run_ladder(environment_name, bot_species, bot_generation):
    env_class = get_env_module(environment_name)

    bots = [
        Bot(
            "random",
            RandomAgent,
            {},
        ),
    ]
    bots = []

    # XXX Truncate down to 10 or so bots
    # - RandomAgent
    # - Bot Generation 1
    # - Bot Generation N - 1
    # - Ones in between
    # for i in range(1, bot_generation + 1):
    # for i in (1, 3, 7, 10, 11):
    for i in (1, 11):
        bot_settings = configure_bot(ENVIRONMENT, bot_species, i)
        bot_settings["move_consideration_time"] = 0.1
        bots.append(
            Bot(
                f"{bot_species}-{i}",
                MCTSAgent,
                bot_settings,
            )
        )

    tournament = Tournament.setup(
        environment=env_class.Environment,
        bots=bots,
    )
    # tournament.round_robin(num_games=200)
    for i in range(100):
        tournament.ladder(num_rounds=1)
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
    ENVIRONMENT = "connect_four"
    # BOT_SPECIES = "mcts_naive"
    BOT_SPECIES = "mcts_gbdt"
    BOT_GENERATION = 5
    run_faceoff(ENVIRONMENT, BOT_SPECIES, BOT_GENERATION)
    # run_ladder(ENVIRONMENT, BOT_SPECIES, BOT_GENERATION)
