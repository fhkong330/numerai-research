import tempfile
import unittest
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.hosted_model import make_hosted_predict, verify_hosted_package


def fitted_model_text(features):
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(rng.normal(size=(100, len(features))), columns=features)
    target = frame.sum(axis=1)
    model = lgb.train(
        {"objective": "regression", "verbosity": -1, "num_threads": 1, "seed": 42},
        lgb.Dataset(frame, label=target, feature_name=list(features)),
        num_boost_round=5,
    )
    return model.model_to_string()


class HostedModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.medium_features = ("feature_a", "feature_b", "feature_c")
        cls.top_features = ("feature_c", "feature_a")
        cls.predict = staticmethod(make_hosted_predict(
            fitted_model_text(cls.medium_features),
            fitted_model_text(cls.top_features),
            cls.medium_features,
            cls.top_features,
        ))
        cls.live = pd.DataFrame(
            {
                "feature_c": [0.0, 1.0, 2.0, 3.0],
                "irrelevant": [9, 8, 7, 6],
                "feature_a": [1.0, 0.0, 1.0, 2.0],
                "feature_b": [0.0, 1.0, 0.0, 1.0],
            },
            index=pd.Index(["t3", "t1", "t2", "t4"], name="numerai_ticker"),
        )

    def test_two_argument_contract_order_range_and_weights(self):
        output = self.predict(self.live, pd.DataFrame(index=self.live.index))
        self.assertEqual(output.columns.tolist(), ["prediction"])
        self.assertTrue(output.index.equals(self.live.index))
        self.assertTrue(output["prediction"].between(0, 1).all())

    def test_shuffle_and_extra_columns_do_not_change_identity_aligned_predictions(self):
        first = self.predict(self.live, pd.DataFrame(index=self.live.index))
        shuffled = self.live.sample(frac=1, random_state=7)
        second = self.predict(shuffled, pd.DataFrame(index=shuffled.index))
        pd.testing.assert_series_equal(
            first["prediction"].sort_index(), second["prediction"].sort_index()
        )

    def test_missing_features_and_duplicate_index_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, "missing required"):
            self.predict(self.live.drop(columns="feature_b"), pd.DataFrame())
        duplicate = pd.concat([self.live, self.live.iloc[[0]]])
        with self.assertRaisesRegex(ValueError, "unique"):
            self.predict(duplicate, pd.DataFrame())

    def test_cloudpickle_roundtrip_when_available(self):
        try:
            import cloudpickle
        except ImportError:
            self.skipTest("cloudpickle is intentionally isolated to .venv_hosted")
        restored = cloudpickle.loads(cloudpickle.dumps(self.predict))
        expected = self.predict(self.live, pd.DataFrame(index=self.live.index))
        actual = restored(self.live, pd.DataFrame(index=self.live.index))
        pd.testing.assert_frame_equal(expected, actual)

    def test_incomplete_artifact_directory_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "incomplete"):
                verify_hosted_package(Path(temp_dir))


if __name__ == "__main__":
    unittest.main()
