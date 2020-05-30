import typing
from dataclasses import dataclass
import shutil
import settings
from uuid import uuid4
import itertools
import pprint
import json

from treelite.runtime import (
    Predictor as TreelitePredictor,
)
import numpy
import lightgbm

from treelite_model import build_treelite_model


@dataclass
class GBDTModel:
    treelite_model_path: typing.Any = None
    treelite_predictor: typing.Any = None

    def save(self, output_path):
        if self.treelite_model_path is None:
            raise RuntimeError("Model hasn't been created/loaded, can't save elsewhere.")

        # Copy the current dylib file (which might be temporary path) to another location
        shutil.copyfile(self.treelite_model_path, output_path)

    def load(self, model_path, nthread=1):
        self.treelite_model_path = model_path
        self.treelite_predictor = TreelitePredictor(
            model_path,
            nthread=nthread,
        )

    def stash_training_data(
        self,
        train_features,
        train_labels,
        test_features,
        test_labels,
    ):
        base_path = f"{settings.TMP_DIRECTORY}/policy_model-{str(uuid4())}_"
        train_path = f"{base_path}train_features.npy"
        train_labels_path = f"{base_path}train_labels.npy"
        test_path = f"{base_path}test_features.npy"
        test_labels_path = f"{base_path}test_labels.npy"
        numpy.save(train_path, train_features)
        numpy.save(train_labels_path, train_labels)
        numpy.save(test_path, test_features)
        numpy.save(test_labels_path, test_labels)
        print("\nSTASHING TRAINING DATA")
        for p in (train_path, train_labels_path, test_path, test_labels_path):
            print(p)

    @classmethod
    def load_stashed_training_data(self, base_path):
        train_path = f"{base_path}train_features.npy"
        train_labels_path = f"{base_path}train_labels.npy"
        test_path = f"{base_path}test_features.npy"
        test_labels_path = f"{base_path}test_labels.npy"

        train_features = numpy.load(train_path)
        train_labels = numpy.load(train_labels_path)
        test_features = numpy.load(test_path)
        test_labels = numpy.load(test_labels_path)

        return train_features, train_labels, test_features, test_labels

    def extract_training_observations(self, game_samples, test_fraction):
        # Each set has same structure as :samples, just partitioned into games that are in the
        # training set and games that are in the test set.
        raise NotImplementedError()

    def train(
        self,
        objective,
        eval_metrics,
        samples,
        test_fraction=.2,
    ):
        # :samples ~ dict(meta_info=..., features=..., labels=...)

        # Make training observations from game posisitions
        (
            train_features,
            train_labels,
            test_features,
            test_labels,
        ) = self.extract_training_observations(samples, test_fraction)

        print("Sample train features:", train_features[:1])
        print("Sample train labels:", train_labels[:1])

        print(f"\nTrain features shape: {train_features.shape}")
        print(f"Train labels shape: {train_labels.shape}")
        print(f"Test features shape: {test_features.shape}")
        print(f"Test labels shape: {test_labels.shape}")

        # Stash the data so we can reload it later to easily tweak with stuff
        self.stash_training_data(train_features, train_labels, test_features, test_labels)

        # Train lgbm model/treelite model
        self.train_from_training_data(
            objective,
            eval_metrics,
            train_features,
            train_labels,
            test_features,
            test_labels,
        )

    def train_from_training_data(
        self,
        objective,
        eval_metrics,
        train_features,
        train_labels,
        test_features,
        test_labels,
    ):
        train_data = lightgbm.Dataset(train_features, label=train_labels)
        test_data = lightgbm.Dataset(test_features, label=test_labels)

        num_round = 15000
        early_stopping_rounds = 10
        learning_rate = 0.15
        learning_rate_fxn = lambda x: learning_rate # noqa
        # learning_rate_fxn=lambda x: (lr - lrs) + (lrs * (lrsh ** x)),  # Start with a higher learning rate and adjust lower over time

        # bagging_fractions = [.05, .1, .2, .3, .4]
        # bagging_freqs = [5, 10, 20, 30]
        # num_leaves_choices = [2**7, 2**8, 2**9, 2**10, 2**11]

        bagging_fractions = [.2]
        bagging_freqs = [10]
        num_leaves_choices = [2**8]

        # Best, but treelite trains slowly so beware
        # bagging_fractions = [.3]
        # bagging_freqs = [20]
        # num_leaves_choices = [2**11]

        for (
            bagging_fraction,
            bagging_freq,
            num_leaves,
        ) in itertools.product(
            bagging_fractions,
            bagging_freqs,
            num_leaves_choices
        ):
            params = {
                'objective': objective, # aka xentropy
                'boosting': "gbdt",  # gbdt is slower but better accuracy, goss is faster (but only slightly)
                'metric': eval_metrics,

                'bagging_fraction': bagging_fraction,
                'bagging_freq': bagging_freq,
                'learning_rate': learning_rate,  # This is overriden in the case where dynamic learning_rates are specified below
                'num_leaves': num_leaves,
                'max_bin': 128,
                'min_data_in_leaf': 10,
                'num_threads': 16,  # 0 is as many as CPUs for server
                'verbose': 1,
                # 'max_depth': 3,
                # 'min_gain_to_split': 0.01,
            }

            print("\nTraining")
            lightgbm_booster = lightgbm.train(
                params,
                train_data,
                num_round,
                valid_sets=[train_data, test_data],
                learning_rates=learning_rate_fxn,
                early_stopping_rounds=early_stopping_rounds, # Stops if ANY metric in metrics doesn't improve in N rounds
            )

            print("\nTrained with following params:")
            pprint.pprint(params)

        # Save lightgbm model to disk so treelite can load it
        lightgbm_model_path = f"{settings.TMP_DIRECTORY}/lightgbm-{str(uuid4())}.model"
        lightgbm_booster.save_model(lightgbm_model_path)
        print("Dumped LGBM model here:", lightgbm_model_path)

        model_dict = lightgbm_booster.dump_model()
        lightgbm_model_dump_path = f"{settings.TMP_DIRECTORY}/lightgbm-{str(uuid4())}.json"
        with open(lightgbm_model_dump_path, 'w') as f:
            f.write(json.dumps(model_dict))
        print("Dumped LGBM model (JSON) here:", lightgbm_model_dump_path)

        # Build treelite model
        #  - stash path in self.treelite_model_path
        num_rows = test_features.shape[0]
        annotation_samples = test_features[numpy.random.choice(num_rows, 500_000), :]
        self.treelite_model_path = build_treelite_model(
            lightgbm_model_path,
            annotation_samples=annotation_samples,
        )

        # Load up the just-made treelite model for use
        self.load(self.treelite_model_path)
