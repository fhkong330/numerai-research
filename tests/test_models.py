import json
import unittest

import numpy as np
import pandas as pd

from src.config import DEFAULT_LIGHTGBM_PARAMS, get_research_config
from src.data import prepare_v6_model_data, resolve_v6_numeric_feature_set
from src.models import (
    create_lightgbm_model,
    predict_model,
    predict_validated_lightgbm,
    train_model,
    train_validated_lightgbm,
)
from src.paths import resolve_data_dir


class V6LightGBMInterfaceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.feature_names = ["feature_a", "feature_b"]
        self.X = pd.DataFrame(
            {
                "feature_a": np.linspace(0, 1, 40),
                "feature_b": np.tile([0, 1, 2, 3, 4], 8),
            },
            index=pd.Index(range(100, 140), name="row_id"),
        )
        self.y = pd.Series(
            np.tile([0.0, 0.25, 0.5, 0.75, 1.0], 8),
            index=self.X.index,
            name="target_jupiter_60",
        )
        self.test_params = {
            **DEFAULT_LIGHTGBM_PARAMS,
            "n_estimators": 8,
            "n_jobs": 1,
            "verbosity": -1,
        }

    def test_official_v6_medium_resolves_350_numeric_features_in_order(self) -> None:
        data_dir = resolve_data_dir("v6.0")
        with (data_dir / "features.json").open("r", encoding="utf-8") as file:
            official = json.load(file)["feature_sets"]["medium"]

        selected = resolve_v6_numeric_feature_set("medium")

        self.assertEqual(len(official), 351)
        self.assertIn("feature_country", official)
        self.assertEqual(selected, [name for name in official if name != "feature_country"])
        self.assertEqual(len(selected), 350)
        self.assertNotIn("feature_country", selected)

    def test_missing_feature_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Missing selected feature"):
            train_validated_lightgbm(
                create_lightgbm_model(self.test_params),
                self.X[["feature_a"]],
                self.y,
                self.feature_names,
            )

    def test_nonnumeric_feature_is_rejected(self) -> None:
        X = self.X.copy()
        X["feature_b"] = "category"
        with self.assertRaisesRegex(ValueError, "must be numeric"):
            train_validated_lightgbm(
                create_lightgbm_model(self.test_params),
                X,
                self.y,
                self.feature_names,
            )

    def test_null_and_nonnumeric_targets_are_rejected(self) -> None:
        null_target = self.y.copy()
        null_target.iloc[0] = np.nan
        with self.assertRaisesRegex(ValueError, "must not contain null"):
            train_validated_lightgbm(
                create_lightgbm_model(self.test_params),
                self.X,
                null_target,
                self.feature_names,
            )
        with self.assertRaisesRegex(ValueError, "must be numeric"):
            train_validated_lightgbm(
                create_lightgbm_model(self.test_params),
                self.X,
                self.y.astype(str),
                self.feature_names,
            )

    def test_misaligned_training_target_is_rejected(self) -> None:
        misaligned = self.y.reset_index(drop=True)
        with self.assertRaisesRegex(ValueError, "indexes do not align"):
            train_validated_lightgbm(
                create_lightgbm_model(self.test_params),
                self.X,
                misaligned,
                self.feature_names,
            )

    def test_prediction_order_must_match_training_order(self) -> None:
        model = train_validated_lightgbm(
            create_lightgbm_model(self.test_params),
            self.X,
            self.y,
            self.feature_names,
        )
        with self.assertRaisesRegex(ValueError, "ordering"):
            predict_validated_lightgbm(model, self.X[["feature_b", "feature_a"]])

    def test_predictions_are_deterministic_and_preserve_row_alignment(self) -> None:
        model_one = train_validated_lightgbm(
            create_lightgbm_model(self.test_params),
            self.X,
            self.y,
            self.feature_names,
        )
        model_two = train_validated_lightgbm(
            create_lightgbm_model(self.test_params),
            self.X,
            self.y,
            self.feature_names,
        )

        first = predict_validated_lightgbm(model_one, self.X)
        second = predict_validated_lightgbm(model_two, self.X)

        self.assertEqual(len(first), len(self.X))
        self.assertTrue(first.index.equals(self.X.index))
        np.testing.assert_array_equal(first.to_numpy(), second.to_numpy())

    def test_v6_model_data_separates_features_target_and_identity(self) -> None:
        config = get_research_config("v6.0")
        data = self.X.copy()
        data[config.target_column] = self.y
        data["date"] = "2020-01-03"
        data["numerai_ticker"] = [f"ticker_{i}" for i in range(len(data))]
        data["composite_figi"] = [f"figi_{i}" for i in range(len(data))]

        prepared = prepare_v6_model_data(data, self.feature_names)

        self.assertEqual(prepared.features.columns.tolist(), self.feature_names)
        self.assertEqual(prepared.target.name, config.target_column)
        self.assertEqual(
            prepared.identity.columns.tolist(),
            ["numerai_ticker", "composite_figi", "date"],
        )
        self.assertTrue(prepared.identity.index.equals(data.index))

    def test_legacy_v5_model_functions_remain_compatible(self) -> None:
        model = create_lightgbm_model(self.test_params)
        fitted = train_model(model, self.X, self.y)
        predictions = predict_model(fitted, self.X)

        self.assertEqual(len(predictions), len(self.X))
        self.assertIsInstance(predictions, np.ndarray)


if __name__ == "__main__":
    unittest.main()
