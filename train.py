from multiprocessing import Pool
import os

import numpy

from environment_registry import get_env_module
from paths import build_model_paths, find_batch_directory
from intuition_model import NaiveValue, NaivePolicy, GBDTValue, GBDTPolicy
from training_samples import generate_training_samples, SampleData


def run_worker(args):
    # What is this, Perl??
    (
        environment,
        bot_species,
        batch_num,
        max_positions,
        worker_num,
        num_workers,
    ) = args

    env_class = get_env_module(environment)
    env = env_class.Environment()

    replay_directory = find_batch_directory(environment, bot_species, batch_num)

    print("Collecting Samples", replay_directory)
    value_meta = []
    value_features = []
    value_labels = []
    policy_meta = []
    policy_features = []
    policy_labels = []
    for position_num, sample in enumerate(
        generate_training_samples(
            replay_directory,
            env_class.State,
            env_class.generate_features,
            env,
            worker_num=worker_num,
            num_workers=num_workers,
        )
    ):
        if position_num >= (max_positions - 1):
            break

        # game_bucket, features, labels (or just label for value)
        meta_info = sample[1] # [game_bucket, generation, ...]
        if sample[0] == "value":
            value_meta.append(meta_info)
            value_features.append(sample[2]) # [[float, ...]]
            value_labels.append(sample[3]) # [int, ...]
        else:
            policy_meta.append(meta_info)
            policy_features.append(sample[2])
            policy_labels.append(sample[3]) # [[float, ...], ...]

    datasets = [
        ("value_meta", value_meta),
        ("value_features", value_features),
        ("value_labels", value_labels),

        ("policy_meta", policy_meta),
        ("policy_features", policy_features),
        ("policy_labels", policy_labels),
    ]
    for sample_type, data in datasets:
        basename = f"{sample_type}_samples_{worker_num + 1:04d}of{num_workers:04d}.npy"
        parsed_samples_path = f"{replay_directory}/{basename}"
        numpy.save(parsed_samples_path, data)
        print(f"Saved: {parsed_samples_path}")

    return position_num


def generate_batch_samples(
    environment,
    bot_species,
    batch_num,
    num_workers,
    positions_per_batch=1_000_000_000,
    force_regeneration=False,
):
    # Don't generate samples if they exist already
    if force_regeneration is False:
        existing_batch_sample_files = find_batch_sample_files(environment, bot_species, batch_num, "value")
        if existing_batch_sample_files["features"]:
            print(f"Samples already exist for {environment}/{bot_species}/{batch_num}. Skipping.")
            return

    # Delete any existing samples to prevent duplicate data.
    delete_batch_samples(environment, bot_species, batch_num)

    # Collect samples in parallel
    worker_args = []
    for worker_num in range(num_workers):
        worker_args.append(
            (
                environment,
                bot_species,
                batch_num,
                positions_per_batch // num_workers,
                worker_num,
                num_workers
            )
        )
    with Pool(num_workers) as p:
        results = p.map(run_worker, worker_args)
    print(results)


def find_batch_sample_files(environment, bot_species, batch_num, model_type):
    sample_file_paths = dict(
        meta=[],
        features=[],
        labels=[],
    )
    replay_directory = find_batch_directory(environment, bot_species, batch_num)
    for file_name in os.listdir(replay_directory):
        if not file_name.endswith(".npy"):
            continue

        # Find the type of data this file is
        for data_type in sample_file_paths.keys():
            if file_name.startswith(f"{model_type}_{data_type}_samples"):
                file_path = os.path.join(replay_directory, file_name)
                sample_file_paths[data_type].append(file_path)
                break

    # Presort them just to be safe.
    # - Operations on these files will assume they are paired up by index id.
    for _, paths in sample_file_paths.items():
        paths.sort()

    assert len(sample_file_paths["meta"]) == len(sample_file_paths["features"])
    assert len(sample_file_paths["features"]) == len(sample_file_paths["labels"])

    return sample_file_paths


def delete_batch_samples(environment, bot_species, batch_num):
    print("Deleting sample files")
    for model_type in ("value", "policy"):
        batch_file_paths = find_batch_sample_files(
            environment,
            bot_species,
            batch_num,
            model_type,
        )
        for _, file_paths in batch_file_paths.items():
            for file_path in file_paths:
                os.remove(file_path)


def load_game_samples(
    environment,
    bot_species,
    batches,
    model_type,
):
    # Load all the paths for all the samples requested.
    sample_file_paths = dict(
        meta=[],
        features=[],
        labels=[],
    )
    for batch_num in batches:
        batch_file_paths = find_batch_sample_files(
            environment,
            bot_species,
            batch_num,
            model_type,
        )
        for k, v in batch_file_paths.items():
            sample_file_paths[k].extend(v)

    # Load the sorted file paths for each type of data into one array for each
    # data type.
    #
    # YOU MUST SORT THESE FILE NAMES.  If you don't then the ith feature won't
    # match the ith label and nothing will train correctly.
    #
    # XXX: parse out batch/file names and to 100% ensure they are sorted
    # correctly.
    samples = dict(
        meta=[],
        features=[],
        labels=[],
    )
    for k, paths in sample_file_paths.items():
        paths.sort()
        print("Loading:", len(paths), k, f"{model_type} samples")
        datasets = []
        for p in paths:
            datasets.append(numpy.load(p))
        samples[k] = numpy.concatenate(datasets)

    # They should all be the same size
    assert samples["meta"].shape[0] == samples["features"].shape[0] == samples["labels"].shape[0]

    return SampleData(
        features=samples["features"],
        labels=samples["labels"],
        meta_info=samples["meta"],
    )


def run(
    environment,
    species,
    generation,
    head_batch_num,
    num_workers,
    max_games=500_000,
    max_generational_lookback=10,
    positions_per_batch=1_000_000_000,
):
    batch_nums = list(range(head_batch_num, 0, -1))

    ####################
    # Generate Samples
    ####################
    for batch_num in batch_nums:
        generate_batch_samples(
            environment,
            species,
            batch_num=batch_num,
            num_workers=num_workers,
            positions_per_batch=positions_per_batch,
        )

    ####################
    # Train Models
    ####################
    value_model_path, policy_model_path = build_model_paths(
        environment,
        species,
        generation,
    )

    # XXX: Put in agent_configuration
    # Make Species.value_model()
    if species == "mcts_naive":
        value_model = NaiveValue()
        policy_model = NaivePolicy()
    elif species == "gbdt":
        value_model = GBDTValue()
        policy_model = GBDTPolicy(num_workers=num_workers)
    elif species in ("gbdt_rg", "gbdt_rg2"):
        strat = "rg2" if species == "gbdt_rg2" else "rg"
        value_model = GBDTValue(
            weighting_strat=strat,
            highest_generation=generation - 1,
        )
        policy_model = GBDTPolicy(num_workers=num_workers)
    else:
        raise KeyError(f"Unknown species: {species}")

    model_settings = [
        ("value", value_model, value_model_path),
        ("policy", policy_model, policy_model_path),
    ]
    for model_type, model, model_path in model_settings:
        game_samples = load_game_samples(
            environment,
            species,
            batches=batch_nums,
            model_type=model_type,
        )
        model.train(game_samples)
        model.save(model_path)
        print("Saved model to", model_path)

        # Attempt to clear the memory. It's Python, so we'll see what happens...
        del game_samples


if __name__ == "__main__":
    ENVIRONMENT = "quoridor"
    BOT_SPECIES = "gbdt"
    BOT_GENERATION = 6 # XXX Highest + 1
    HEAD_BATCH_NUM = 6 # Highest
    MAX_GAMES = 500_000
    MAX_GENERATIONAL_LOOKBACK = 10
    POSITIONS_PER_BATCH = 100 * 64_000 * 10 # moves/game * max games/batch * safety multiplier
    run(
        ENVIRONMENT,
        BOT_SPECIES,
        BOT_GENERATION,
        HEAD_BATCH_NUM,
        12,
        MAX_GAMES,
        MAX_GENERATIONAL_LOOKBACK,
        POSITIONS_PER_BATCH,
    )
