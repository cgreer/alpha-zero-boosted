from human_agent import HumanAgent
from mcts_agent import MCTSAgent
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
    species,
    generation,
    play_setting="self_play",
):
    '''
    :play_setting ~ {"self_play", "evaluation"}
    '''
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

    move_consideration_time = 3.0
    temperature = 0.0
    full_search_proportion = 1.0
    full_search_steps = 800
    partial_search_steps = full_search_steps // 5

    # Play-setting dependent
    # - XXX Add Puct params
    lower_bound_time = 0.01
    if play_setting == "self_play":
        temperature = .3 # XXX Make a temp profile per move
        move_consideration_time = lower_bound_time
    elif play_setting == "evaluation":
        move_consideration_time = lower_bound_time

    return dict(
        species=species,
        generation=generation,
        game_tree=None,
        current_node=None,
        feature_extractor=env_module.generate_features,
        value_model=value_model,
        policy_model=policy_model,
        move_consideration_time=move_consideration_time, # at least N seconds
        puct_explore_factor=1.0,
        puct_noise_alpha=0.4,
        puct_noise_influence=0.25,
        full_search_proportion=full_search_proportion,
        full_search_steps=full_search_steps,
        partial_search_steps=partial_search_steps,
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
    if species in (
        "gbdt",
        "gbdt_rg",
        "gbdt_rg2",
        "gbdt_pcr",
        "gbdt_pcr_v",
    ):
        Agent = MCTSAgent
        agent_settings = gbdt_configuration(
            environment_name,
            species,
            generation,
            play_setting,
        )

        # PCR does PCR during self-play
        if species == "gbdt_pcr" and play_setting == "self_play":
            agent_settings["full_search_proportion"] = .2

        if species == "gbdt_pcr_v" and play_setting == "self_play":
            agent_settings["full_search_proportion"] = .2
            agent_settings["require_full_steps"] = False

        return Agent, agent_settings
    elif species == "human":
        settings = {"species": "human", "generation": 1}
        return (
            HumanAgent,
            settings,
        )
    else:
        raise KeyError(f"Unknown species: {species}")
