import connect_four
import tictactoe

ENVIRONMENT_REGISTRY = dict(
    connect_four=connect_four,
    tictactoe=tictactoe,
)


def get_env_module(environment_name):
    return ENVIRONMENT_REGISTRY[environment_name]
