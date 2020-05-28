import connect_four
import tictactoe
import quoridor

ENVIRONMENT_REGISTRY = dict(
    connect_four=connect_four,
    tictactoe=tictactoe,
    quoridor=quoridor,
)


def get_env_module(environment_name):
    return ENVIRONMENT_REGISTRY[environment_name]
