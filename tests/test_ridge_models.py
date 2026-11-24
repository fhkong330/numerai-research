import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.models import (
    load_streaming_ridge,
    predict_streaming_ridge,
    save_streaming_ridge,
    train_streaming_ridge,
)


class StreamingRidgeModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.features = ["feature_a", "feature_b", "feature_constant"]
        self.frame = pd.DataFrame(
            {
                "feature_a": np.asarray([0, 1, 2, 3, 4, 5], dtype="int8"),
                "feature_b": np.asarray(
                    [1.0, np.nan, 3.0, 4.0, 5.0, 6.0], dtype="float32"
                ),
                "feature_constant": np.asarray([7] * 6, dtype="int8"),
            },
            index=pd.Index([10, 11, 12, 13, 14, 15]),
        )
        self.target = pd.Series(
            [0.1, 0.2, 0.35, 0.5, 0.7, 0.9],
            index=self.frame.index,
            dtype="float32",
        )
        self.params = {
            "alpha": 0.001,
            "batch_size": 2,
            "near_constant_threshold": 1e-6,
            "solver": "batched_sufficient_statistics",
        }

    def test_fit_predict_missing_constant_and_bounded_batches(self) -> None:
        model = train_streaming_ridge(
            self.frame, self.target, self.features, self.params
        )
        predictions = predict_streaming_ridge(model, self.frame)
        self.assertEqual(predictions.index.tolist(), self.frame.index.tolist())
        self.assertTrue(np.isfinite(predictions).all())
        self.assertEqual(model.max_batch_rows_seen, 2)
        self.assertAlmostEqual(model.means[1], 3.8, places=6)
        self.assertEqual(model.scales[2], 1.0)
        self.assertAlmostEqual(model.coefficients[2], 0.0, places=12)
        self.assertEqual(model.constant_or_near_constant_count, 1)
        self.assertEqual(model.entirely_missing_count, 0)
        self.assertGreater(model.training_runtime_seconds, 0)

    def test_preprocessing_is_training_only_and_deterministic(self) -> None:
        first = train_streaming_ridge(
            self.frame, self.target, self.features, self.params
        )
        second = train_streaming_ridge(
            self.frame, self.target, self.features, self.params
        )
        np.testing.assert_array_equal(first.means, second.means)
        np.testing.assert_array_equal(first.coefficients, second.coefficients)
        validation = self.frame.iloc[:2].copy()
        validation["feature_b"] = [1000.0, np.nan]
        predict_streaming_ridge(first, validation)
        self.assertAlmostEqual(first.means[1], 3.8, places=6)

    def test_feature_order_and_categorical_rejection(self) -> None:
        model = train_streaming_ridge(
            self.frame, self.target, self.features, self.params
        )
        with self.assertRaisesRegex(ValueError, "ordering"):
            predict_streaming_ridge(model, self.frame[self.features[::-1]])
        categorical = self.frame.copy()
        categorical["feature_b"] = "country"
        with self.assertRaisesRegex(ValueError, "numeric"):
            train_streaming_ridge(
                categorical, self.target, self.features, self.params
            )

    def test_save_reload_prediction_consistency(self) -> None:
        model = train_streaming_ridge(
            self.frame, self.target, self.features, self.params
        )
        expected = predict_streaming_ridge(model, self.frame)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "ridge.json"
            save_streaming_ridge(model, path)
            loaded = load_streaming_ridge(path)
            actual = predict_streaming_ridge(loaded, self.frame)
        np.testing.assert_array_equal(expected.to_numpy(), actual.to_numpy())
        self.assertEqual(loaded.feature_names, tuple(self.features))
        self.assertEqual(
            loaded.constant_or_near_constant_count,
            model.constant_or_near_constant_count,
        )

    def test_medium_width_fit_remains_batch_bounded(self) -> None:
        rng = np.random.default_rng(42)
        features = [f"feature_{index:03d}" for index in range(350)]
        frame = pd.DataFrame(
            rng.normal(size=(1024, 350)).astype("float32"), columns=features
        )
        target = pd.Series(rng.normal(size=1024).astype("float32"))
        model = train_streaming_ridge(
            frame,
            target,
            features,
            {**self.params, "batch_size": 128},
        )
        self.assertEqual(model.max_batch_rows_seen, 128)
        self.assertEqual(model.coefficients.shape, (350,))
        self.assertTrue(np.isfinite(model.coefficients).all())

    def test_normalized_objective_matches_sklearn_penalty_convention(self) -> None:
        model = train_streaming_ridge(
            self.frame, self.target, self.features, self.params
        )
        values = self.frame.to_numpy(dtype=np.float64, copy=True)
        missing = np.isnan(values)
        rows, columns = np.nonzero(missing)
        values[rows, columns] = model.means[columns]
        standardized = (values - model.means) / model.scales
        centered_target = self.target.to_numpy(dtype=np.float64) - model.intercept
        reference = Ridge(
            alpha=len(self.frame) * model.alpha,
            fit_intercept=False,
            solver="cholesky",
        ).fit(standardized, centered_target)
        np.testing.assert_allclose(
            model.coefficients, reference.coef_, rtol=1e-6, atol=1e-8
        )


if __name__ == "__main__":
    unittest.main()
