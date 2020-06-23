from dataclasses import dataclass
from agent import Agent

from human_agent import HumanAgent
from mcts_agent import MCTSAgent
from environment_registry import get_env_module
import intuition_model
from paths import build_model_paths


NUM_WORKERS = 12

SPECIES_REGISTRY = set()


@dataclass
class Species:
    name: str
    AgentClass: Agent

    def agent_settings(self, environment, generation, play_setting):
        raise NotImplementedError()

    def self_play_settings(self, environment, generation):
        raise NotImplementedError()

    def training_settings(self, environment, generation):
        raise NotImplementedError()


@dataclass
class Human(Species):
    name: str = "human"
    AgentClass: Agent = HumanAgent

    def agent_settings(self, environment, generation, play_setting):
        return dict(species=self.name, generation=generation)
SPECIES_REGISTRY.add(Human) # noqa


@dataclass
class GBDT(Species):
    name: str = "gbdt"
    AgentClass: Agent = MCTSAgent

    def agent_settings(self, environment, generation, play_setting):
        env_module = get_env_module(environment)

        # Setup value/policy models
        if generation == 1:
            value_model = intuition_model.UnopinionatedValue()
            if hasattr(env_module, "BootstrapValue"):
                value_model = env_module.BootstrapValue()
            policy_model = intuition_model.UniformPolicy()
        else:
            value_model_path, policy_model_path = build_model_paths(
                environment,
                self.name,
                generation,
            )
            value_model = intuition_model.GBDTValue()
            value_model.load(value_model_path)

            policy_model = intuition_model.GBDTPolicy()
            policy_model.load(policy_model_path)

        # Settings
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
            species=self.name,
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

    def self_play_settings(self, environment, generation):
        return dict(num_games=3000)

    def training_settings(self, environment, generation):
        return dict(
            ValueModel=intuition_model.GBDTValue,
            value_model_settings=dict(),
            # Value model settings for reweighting
            # dict(weighting_strat=strat, highest_generation=generation - 1)
            PolicyModel=intuition_model.GBDTPolicy,
            policy_model_settings=dict(num_workers=NUM_WORKERS),
        )
SPECIES_REGISTRY.add(GBDT) # noqa


@dataclass
class GBDTR1(GBDT):
    name: str = "gbdtR1"
SPECIES_REGISTRY.add(GBDTR1) # noqa


@dataclass
class GBDTR2(GBDT):
    name: str = "gbdtR2"
SPECIES_REGISTRY.add(GBDTR2) # noqa


@dataclass
class PCR(GBDT):
    name: str = "gbdt_pcr"

    def agent_settings(self, environment, generation, play_setting):
        sets = super().agent_settings(environment, generation, play_setting)
        if play_setting == "self_play":
            sets["full_search_proportion"] = .2
        return sets
SPECIES_REGISTRY.add(PCR) # noqa


@dataclass
class PCRR1(PCR):
    name: str = "gbdt_pcrR1"
SPECIES_REGISTRY.add(PCRR1) # noqa


@dataclass
class PCRR2(PCR):
    name: str = "gbdt_pcrR2"
SPECIES_REGISTRY.add(PCRR2) # noqa


@dataclass
class PCRP1(GBDT):
    name: str = "gbdt_pcrp1"

    def agent_settings(self, environment, generation, play_setting):
        sets = super().agent_settings(environment, generation, play_setting)
        if play_setting == "self_play":
            sets["full_search_proportion"] = .1
        return sets
SPECIES_REGISTRY.add(PCRP1) # noqa


@dataclass
class PCRP2(GBDT):
    name: str = "gbdt_pcrp2"

    def agent_settings(self, environment, generation, play_setting):
        sets = super().agent_settings(environment, generation, play_setting)
        if play_setting == "self_play":
            sets["full_search_proportion"] = .2
        return sets
SPECIES_REGISTRY.add(PCRP2) # noqa


@dataclass
class PCRP3(GBDT):
    name: str = "gbdt_pcrp3"

    def agent_settings(self, environment, generation, play_setting):
        sets = super().agent_settings(environment, generation, play_setting)
        if play_setting == "self_play":
            sets["full_search_proportion"] = .3
        return sets
SPECIES_REGISTRY.add(PCRP3) # noqa


@dataclass
class PCRV(GBDT):
    name: str = "gbdt_pcrv"

    def agent_settings(self, environment, generation, play_setting):
        sets = super().agent_settings(environment, generation, play_setting)
        sets["full_search_proportion"] = .2
        sets["require_full_steps"] = False
        return sets
SPECIES_REGISTRY.add(PCRV) # noqa


@dataclass
class RVE(GBDT):
    name: str = "gbdt_rve"

    def agent_settings(self, environment, generation, play_setting):
        sets = super().agent_settings(environment, generation, play_setting)
        sets["require_full_steps"] = False
        sets["revisit_violated_expectations"] = True
        if play_setting == "self_play":
            sets["full_search_proportion"] = .2
        return sets

    def self_play_settings(self, environment, generation):
        sps = super().self_play_settings(environment, generation)
        sps["num_games"] = 300
        return sps
SPECIES_REGISTRY.add(RVE) # noqa


@dataclass
class RVER1(RVE):
    name: str = "gbdt_rveR1"
SPECIES_REGISTRY.add(RVER1) # noqa


SPECIES_BY_NAME = {}
for species_class in SPECIES_REGISTRY:
    sp = species_class()
    SPECIES_BY_NAME[sp.name] = sp


def get_species(name):
    return SPECIES_BY_NAME[name]
