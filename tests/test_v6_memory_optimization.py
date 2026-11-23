import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import (
    V6_IDENTITY_COLUMNS,
    build_v6_research_history,
    prepare_v6_fold_model_data,
    prepare_v6_model_data,
)
from src.models import (
    create_lightgbm_model,
    predict_validated_lightgbm,
    train_validated_lightgbm,
)
from src.pipeline import _synthetic_v6_smoke_inputs
from src.validation import prepare_v6_walk_forward_fold


class V6MemoryOptimizationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        (
            self.historical,
            self.research,
            self.features,
            self.folds,
            _,
        ) = _synthetic_v6_smoke_inputs(self.root)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_direct_fold_inputs_match_legacy_values_dtypes_and_order(self) -> None:
        history = build_v6_research_history(self.historical, self.research)
        for fold in self.folds:
            legacy_train, legacy_valid = prepare_v6_walk_forward_fold(history, fold)
            legacy_train = prepare_v6_model_data(legacy_train, self.features)
            legacy_valid = prepare_v6_model_data(legacy_valid, self.features)
            optimized_train, optimized_valid = prepare_v6_fold_model_data(
                self.historical, self.research, self.features, fold
            )

            pd.testing.assert_frame_equal(
                optimized_train.features.reset_index(drop=True),
                legacy_train.features.reset_index(drop=True),
                check_dtype=True,
            )
            pd.testing.assert_series_equal(
                optimized_train.target.reset_index(drop=True),
                legacy_train.target.reset_index(drop=True),
                check_dtype=True,
            )
            pd.testing.assert_frame_equal(
                optimized_valid.features.reset_index(drop=True),
                legacy_valid.features.reset_index(drop=True),
                check_dtype=True,
            )
            pd.testing.assert_frame_equal(
                optimized_valid.identity.reset_index(drop=True),
                legacy_valid.identity.reset_index(drop=True),
                check_dtype=True,
            )
            self.assertEqual(
                set(optimized_valid.identity["date"]), set(fold.validation_dates)
            )
            self.assertTrue(
                set(optimized_valid.identity["date"]).isdisjoint(fold.train_dates)
            )

    def test_bounded_predictions_match_legacy_path(self) -> None:
        fold = self.folds[0]
        history = build_v6_research_history(self.historical, self.research)
        legacy_train, legacy_valid = prepare_v6_walk_forward_fold(history, fold)
        legacy_train = prepare_v6_model_data(legacy_train, self.features)
        legacy_valid = prepare_v6_model_data(legacy_valid, self.features)
        optimized_train, optimized_valid = prepare_v6_fold_model_data(
            self.historical, self.research, self.features, fold
        )
        params = {"n_estimators": 3, "n_jobs": 1, "verbose": -1}
        legacy_model = train_validated_lightgbm(
            create_lightgbm_model(params),
            legacy_train.features,
            legacy_train.target,
            self.features,
        )
        optimized_model = train_validated_lightgbm(
            create_lightgbm_model(params),
            optimized_train.features,
            optimized_train.target,
            self.features,
        )
        legacy_predictions = predict_validated_lightgbm(
            legacy_model, legacy_valid.features
        )
        optimized_predictions = predict_validated_lightgbm(
            optimized_model, optimized_valid.features
        )
        np.testing.assert_allclose(
            optimized_predictions.to_numpy(),
            legacy_predictions.to_numpy(),
            rtol=0.0,
            atol=1e-12,
        )
        pd.testing.assert_frame_equal(
            optimized_valid.identity.reset_index(drop=True),
            legacy_valid.identity.reset_index(drop=True),
        )

    def test_only_approved_research_frame_can_supply_validation_rows(self) -> None:
        fold = self.folds[-1]
        _, valid = prepare_v6_fold_model_data(
            self.historical, self.research, self.features, fold
        )
        self.assertEqual(tuple(valid.identity.columns), V6_IDENTITY_COLUMNS)
        self.assertEqual(set(valid.identity["date"]), set(fold.validation_dates))
        self.assertLessEqual(max(valid.identity["date"]), "2025-04-25")


if __name__ == "__main__":
    unittest.main()
