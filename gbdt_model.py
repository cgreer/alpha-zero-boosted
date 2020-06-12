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
from training_samples import SampleData


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

    def extract_training_observations(
        self,
        game_samples: SampleData,
        test_fraction,
    ) -> (SampleData, SampleData):
        raise NotImplementedError()

    def train(
        self,
        objective,
        eval_metrics,
        samples: SampleData,
        test_fraction=.2,
    ):
        # Make training observations from game posisitions
        train_samples, test_samples = self.extract_training_observations(samples, test_fraction)

        print("Sample train feature:", train_samples.features[:1])
        print("Sample train label:", train_samples.labels[:1])
        if train_samples.weights is not None:
            print("Sample train weight:", train_samples.weights[:1])

        print(f"\nTrain features shape: {train_samples.features.shape}")
        print(f"Train labels shape: {train_samples.labels.shape}")
        if train_samples.weights is not None:
            print(f"Train weights shape: {train_samples.weights.shape}")

        print(f"Test features shape: {test_samples.features.shape}")
        print(f"Test labels shape: {test_samples.labels.shape}")

        # Stash the data so we can reload it later to easily tweak with stuff
        print("Stashing training samples")
        train_samples.stash_data()

        print("Stashing test samples")
        test_samples.stash_data()

        # Train lgbm model/treelite model
        self.train_from_training_data(
            objective,
            eval_metrics,
            train_samples,
            test_samples,
        )

    def train_from_training_data(
        self,
        objective,
        eval_metrics,
        train_samples: SampleData,
        test_samples: SampleData,
    ):

        train_data = lightgbm.Dataset(
            train_samples.features,
            label=train_samples.labels,
            weight=train_samples.weights,
        )
        test_data = lightgbm.Dataset(
            test_samples.features,
            label=test_samples.labels,
            weight=test_samples.weights,
        )

        num_round = 15000
        early_stopping_rounds = 10
        # bagging_fractions = [.05, .1, .2, .3, .4]
        # bagging_freqs = [5, 10, 20, 30]
        # num_leaves_choices = [2**7, 2**8, 2**9, 2**10, 2**11]
        learning_rates = [.2]
        bagging_fractions = [.2]
        bagging_freqs = [10]
        num_leaves_choices = [2**10]

        # bagging_fractions = [.3]
        # bagging_freqs = [20]
        # num_leaves_choices = [2**11]

        for (
            bagging_fraction,
            bagging_freq,
            num_leaves,
            learning_rate,
        ) in itertools.product(
            bagging_fractions,
            bagging_freqs,
            num_leaves_choices,
            learning_rates,
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

            learning_rate_fxn = lambda x: learning_rate # noqa
            # learning_rate_fxn=lambda x: (lr - lrs) + (lrs * (lrsh ** x)),  # Start with a higher learning rate and adjust lower over time

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
        num_rows = test_samples.features.shape[0]
        annotation_samples = test_samples.features[numpy.random.choice(num_rows, 500_000), :]
        self.treelite_model_path = build_treelite_model(
            lightgbm_model_path,
            annotation_samples=annotation_samples,
        )

        # Load up the just-made treelite model for use
        self.load(self.treelite_model_path)
