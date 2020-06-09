from agents import HumanAgent, MCTSAgent
from environment_registry import get_env_module
import intuition_model
from paths import build_model_paths


def default_models(environment_name):
    env_module = get_env_module(environment_name)
    value_model = intuition_model.UnopinionatedValue()
    if hasattr(env_module, "BootstrapValue"):
        value_model = env_module.BootstrapValue()
    policy_model = intuition_model.UniformPolicy()
    return value_model, policy_model


def gbdt_configuration(
    environment_name,
    generation,
    play_setting="self_play",
):
    '''
    :play_setting ~ {"self_play", "evaluation"}
    '''
    species = "mcts_gbdt"
    env_module = get_env_module(environment_name)

    value_model, policy_model = default_models(environment_name)
    if generation > 1:
        value_model_path, policy_model_path = build_model_paths(
            environment_name,
            species,
            generation,
        )
        value_model = intuition_model.GBDTValue()
        value_model.load(value_model_path)

        policy_model = intuition_model.GBDTPolicy()
        policy_model.load(policy_model_path)

    # Setup consideration steps (could be fxn of env)
    if play_setting == "self_play":
        move_consideration_steps = 800
        move_consideration_time = 0.1
    else:
        move_consideration_steps = 800
        move_consideration_time = 3.0

    # Setup puct params (could be fxn of env)
    # - XXX TODO

    # Setup temperature (could be fxn of env)
    # - XXX Make a profile per move
    if play_setting == "self_play":
        temperature = .3
    else:
        temperature = 0.0

    return dict(
        species=species,
        generation=generation,
        game_tree=None,
        current_node=None,
        feature_extractor=env_module.generate_features,
        value_model=value_model,
        policy_model=policy_model,
        move_consideration_steps=move_consideration_steps, # at least N steps
        move_consideration_time=move_consideration_time, # at least N seconds
        puct_explore_factor=1.0,
        puct_noise_alpha=0.4,
        puct_noise_influence=0.25,
        temperature=temperature,
    )


def configure_agent(
    environment_name,
    species,
    generation,
    play_setting="self_play",
):
    '''
    :play_setting ~ {"self_play", "evaluation"}
    '''
    if species == "mcts_gbdt":
        return (
            MCTSAgent,
            gbdt_configuration(environment_name, generation, play_setting),
        )
    elif species == "human":
        settings = {"species": "human", "generation": 1}
        return (
            HumanAgent,
            settings,
        )
    else:
        raise KeyError(f"Unknown species: {species}")
