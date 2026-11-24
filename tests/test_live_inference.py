import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.live_inference import (
    LiveModelArtifacts,
    audit_live_frame,
    persist_live_predictions,
    predict_live,
    rank_normalize,
    load_model_artifacts,
    verify_prediction_artifacts,
)


class FakeModel:
    def __init__(self, features, nonfinite=False):
        self.features = list(features)
        self.nonfinite = nonfinite

    def predict(self, frame):
        if frame.columns.tolist() != self.features:
            raise AssertionError("Feature order changed")
        result = frame.fillna(0).sum(axis=1).to_numpy(dtype=float)
        if self.nonfinite:
            result[0] = np.nan
        return result


def artifacts(nonfinite=False):
    medium = ("feature_a", "feature_b", "feature_c")
    top = ("feature_c", "feature_a")
    return LiveModelArtifacts(
        FakeModel(medium, nonfinite),
        FakeModel(top),
        medium,
        top,
        "v6_v1_production_002",
        {"medium": "a", "top250": "b"},
    )


def frame():
    return pd.DataFrame(
        {
            "numerai_ticker": ["t3", "t1", "t2", "t4"],
            "date": ["2026-11-20"] * 4,
            "feature_a": np.array([1, 0, 1, 2], dtype=np.int8),
            "feature_b": np.array([0, 1, 0, 1], dtype=np.int8),
            "feature_c": np.array([1, 1, 1, 0], dtype=np.int8),
        }
    )


class LiveInferenceTest(unittest.TestCase):
    def test_production_artifact_loading_verifies_identity_and_order(self):
        medium = [f"feature_{index:03d}" for index in range(350)]
        top250 = medium[50:300]
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            (model_dir / "source_snapshot_manifest.json").write_text(
                json.dumps({"local_audit": {"schema": {"medium_features": medium}}})
            )
            (model_dir / "selected_features.json").write_text(
                json.dumps({"model_feature_names": top250})
            )
            (model_dir / "artifact_hashes.json").write_text(
                json.dumps({"models/medium.txt": "medium-hash", "models/top250.txt": "top-hash"})
            )
            with patch(
                "src.live_inference.verify_production_artifacts",
                return_value={"config": {"model_version": "v6_v1_production_002"}},
            ) as verify, patch("src.live_inference.lgb.Booster", side_effect=["medium", "top"]):
                loaded = load_model_artifacts(model_dir)
            verify.assert_called_once_with(model_dir)
            self.assertEqual(loaded.medium_features, tuple(medium))
            self.assertEqual(loaded.top250_features, tuple(top250))
            self.assertEqual(loaded.model_hashes["medium"], "medium-hash")

    def test_rank_normalization_average_ties(self):
        actual = rank_normalize(pd.Series([1.0, 1.0, 3.0, 4.0]))
        np.testing.assert_allclose(actual, [0.25, 0.25, 0.625, 0.875])

    def test_prediction_order_weights_and_network_independence(self):
        result = predict_live(frame(), artifacts())
        self.assertEqual(result["ensemble"]["numerai_ticker"].tolist(), ["t3", "t1", "t2", "t4"])
        expected = 0.5 * result["medium"]["prediction"] + 0.5 * result["top250"]["prediction"]
        np.testing.assert_allclose(result["ensemble"]["prediction"], expected)
        self.assertTrue(result["ensemble"]["prediction"].between(0, 1).all())

    def test_input_order_invariance(self):
        first = predict_live(frame(), artifacts())["ensemble"].set_index("numerai_ticker")
        second = predict_live(frame().sample(frac=1, random_state=7), artifacts())["ensemble"].set_index("numerai_ticker")
        pd.testing.assert_series_equal(first["prediction"].sort_index(), second["prediction"].sort_index())

    def test_missing_features_duplicate_identity_and_nonfinite_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing required"):
            predict_live(frame().drop(columns="feature_b"), artifacts())
        duplicate = pd.concat([frame(), frame().iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "duplicated"):
            predict_live(duplicate, artifacts())
        with self.assertRaisesRegex(ValueError, "non-finite"):
            predict_live(frame(), artifacts(nonfinite=True))

    def test_audit_reports_feature_coverage_dtypes_and_missingness(self):
        data = frame()
        data.loc[0, "feature_b"] = np.nan
        report = audit_live_frame(data, artifacts())
        self.assertEqual(report["medium_required"], 3)
        self.assertEqual(report["required_missing_values"], 1)
        self.assertEqual(report["medium_missing_columns"], [])

    def test_submission_roundtrip_hashes_and_overwrite_protection(self):
        predictions = predict_live(frame(), artifacts())
        manifest = {
            "snapshot_id": "test",
            "file": {"sha256": "live-hash"},
            "public_metadata_after": {"round": {"number": 1404, "data_datestamp": 20261120}},
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_dir = root / "model"
            model_dir.mkdir()
            (model_dir / "PRODUCTION_TRAINING_COMPLETED.json").write_text("{}")
            destination = persist_live_predictions(
                predictions, manifest, model_dir, artifacts(), output_root=root
            )
            verify_prediction_artifacts(destination)
            saved = pd.read_csv(destination / "submission.csv")
            self.assertEqual(saved.columns.tolist(), ["numerai_ticker", "prediction"])
            self.assertEqual(len(saved), len(frame()))
            with self.assertRaises(FileExistsError):
                persist_live_predictions(predictions, manifest, model_dir, artifacts(), output_root=root)

    def test_incomplete_artifacts_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "incomplete"):
                verify_prediction_artifacts(Path(temp_dir))


if __name__ == "__main__":
    unittest.main()
