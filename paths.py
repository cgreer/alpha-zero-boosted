from settings import ROOT_DATA_DIRECTORY
import os


def build_training_info_path(environment, species):
    return f"{ROOT_DATA_DIRECTORY}/{environment}_{species}/info.json"


def build_self_play_dir(environment, species):
    return f"{ROOT_DATA_DIRECTORY}/{environment}_{species}/self_play"


def build_replay_directory(environment, species, generation, batch):
    root_part = build_self_play_dir(environment, species)
    file_part = f"batch_{batch:06d}_{generation:06d}"

    return f"{root_part}/{file_part}"


def find_batch_directory(environment, species, batch):
    root_part = build_self_play_dir(environment, species)

    batch_sub_path = None
    for sub_path in os.listdir(root_part):
        if f"batch_{batch:06d}" in sub_path:
            batch_sub_path = sub_path
            break

    if batch_sub_path is None:
        raise KeyError("No batch directory found")

    full_path = f"{root_part}/{sub_path}"
    return full_path


def build_model_directory(environment, species, generation):
    return f"{ROOT_DATA_DIRECTORY}/{environment}_{species}/models"


def build_model_paths(environment, species, generation):
    # XXX VOMIT! Abstract this out.
    suffix = "model"
    if "gbdt" in species:
        suffix = "dylib"
    base = build_model_directory(environment, species, generation)
    value_basename = f"value_model_{generation:06d}.{suffix}"
    policy_basename = f"policy_model_{generation:06d}.{suffix}"
    value_path = f"{base}/{value_basename}"
    policy_path = f"{base}/{policy_basename}"
    return value_path, policy_path
