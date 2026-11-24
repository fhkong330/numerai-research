import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.config import DATASET_VERSION, TARGET_COLUMN
from src.compare_experiments import load_comparable_v6_artifacts
from src.ensemble import (
    APPROVED_ENSEMBLE_POLICIES,
    IDENTITY_COLUMNS,
    construct_rank_ensemble,
    load_ensemble_artifacts,
    main,
    rank_normalize_by_date,
    save_ensemble_artifacts,
    score_ensemble_predictions,
    validate_weights,
)


DATES = pd.date_range("2019-01-04", "2025-04-25", freq="W-FRI").strftime("%Y-%m-%d").tolist()
FOLD_LENGTHS = (78, 78, 78, 78, 18)


def folds():
    result = []
    offset = 0
    for fold_id, length in enumerate(FOLD_LENGTHS, 1):
        values = DATES[offset : offset + length]
        result.append(
            {
                "fold": fold_id,
                "validation_start": values[0],
                "validation_end": values[-1],
                "validation_dates": length,
            }
        )
        offset += length
    return result


def artifact(experiment_id, values, shuffled=False):
    rows = []
    offset = 0
    for fold_id, length in enumerate(FOLD_LENGTHS, 1):
        for date in DATES[offset : offset + length]:
            for ticker, value in zip(("a", "b", "c", "d"), values):
                rows.append((date, ticker, f"figi_{ticker}", fold_id, value))
        offset += length
    frame = pd.DataFrame(rows, columns=[*IDENTITY_COLUMNS, "prediction"])
    if shuffled:
        frame = frame.sample(frac=1, random_state=7).reset_index(drop=True)
    metadata = {
        "experiment_id": experiment_id,
        "dataset_version": "v6.0",
        "target": "target_jupiter_60",
        "folds": folds(),
        "metric_identity": ["NCORR", "NUMERAI_CORR_JUPITER_60"],
        "ncorr_reference": {"sha256": "neutralizer"},
    }
    return {"metadata": metadata, "oof_predictions": frame}


class EnsembleTest(unittest.TestCase):
    def test_cli_requires_explicit_real_scoring_confirmation(self):
        arguments = [
            "--version", "v6.0", "--policy", "medium_top250_rank_equal_v1",
            "--ensemble-id", "never_written",
        ]
        with self.assertRaisesRegex(SystemExit, "requires --run"):
            main(arguments)

    def test_rank_normalization_ties_dates_and_constants(self):
        frame = pd.DataFrame(
            {"date": ["d1"] * 4 + ["d2"] * 3, "prediction": [1, 1, 3, 4, 9, 9, 9]}
        )
        result = rank_normalize_by_date(frame, "prediction")
        np.testing.assert_allclose(result[:4], [0.25, 0.25, 0.625, 0.875])
        np.testing.assert_allclose(result[4:], [0.5, 0.5, 0.5])

    def test_invalid_nonfinite_and_weights_are_rejected(self):
        for weights in ({"a": 0.9}, {"a": 1.1, "b": -0.1}, {"a": np.nan}):
            with self.assertRaises(ValueError):
                validate_weights(weights)
        frame = pd.DataFrame({"date": ["d"], "prediction": [np.inf]})
        with self.assertRaisesRegex(ValueError, "finite"):
            rank_normalize_by_date(frame, "prediction")

    def test_fixed_policies_and_weighted_average_are_exact(self):
        policy = APPROVED_ENSEMBLE_POLICIES["medium_top250_rank_equal_v1"]
        self.assertEqual(policy.weights, {
            "v6_medium_baseline_001": 0.5,
            "v6_medium_top250_ic_002": 0.5,
        })
        artifacts = {
            "v6_medium_baseline_001": artifact("v6_medium_baseline_001", [1, 2, 3, 4]),
            "v6_medium_top250_ic_002": artifact(
                "v6_medium_top250_ic_002", [4, 3, 2, 1], shuffled=True
            ),
        }
        result, diagnostics = construct_rank_ensemble(artifacts, policy.weights)
        np.testing.assert_allclose(result["prediction"], 0.5)
        self.assertNotIn("target_jupiter_60", result.columns)
        self.assertEqual(diagnostics["v6_medium_baseline_001"]["constant_date_count"], 0)

    def test_duplicate_mismatch_and_protected_dates_are_rejected(self):
        ids = ("v6_medium_baseline_001", "v6_medium_top250_ic_002")
        good = {name: artifact(name, [1, 2, 3, 4]) for name in ids}
        duplicate = {**good, ids[1]: artifact(ids[1], [1, 2, 3, 4])}
        duplicate[ids[1]]["oof_predictions"] = pd.concat(
            [duplicate[ids[1]]["oof_predictions"], duplicate[ids[1]]["oof_predictions"].iloc[[0]]]
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            construct_rank_ensemble(duplicate, {ids[0]: 0.5, ids[1]: 0.5})
        mismatch = {**good, ids[1]: artifact(ids[1], [1, 2, 3, 4])}
        mismatch[ids[1]]["oof_predictions"].loc[0, "numerai_ticker"] = "wrong"
        with self.assertRaisesRegex(ValueError, "identities"):
            construct_rank_ensemble(mismatch, {ids[0]: 0.5, ids[1]: 0.5})
        protected = {name: artifact(name, [1, 2, 3, 4]) for name in ids}
        for value in protected.values():
            value["oof_predictions"].loc[0:3, "date"] = "2025-08-22"
        with self.assertRaisesRegex(ValueError, "approved"):
            construct_rank_ensemble(protected, {ids[0]: 0.5, ids[1]: 0.5})

    def test_synthetic_scoring_integration_calls_existing_functions(self):
        predictions = pd.DataFrame(
            {
                "date": ["2019-01-04", "2019-01-04"],
                "numerai_ticker": ["a", "b"],
                "composite_figi": ["x", "y"],
                "fold_id": [1, 1],
                "prediction": [0.25, 0.75],
            }
        )
        target = pd.Series([0.0, 1.0])
        legacy = pd.DataFrame({"date": ["2019-01-04"], "fold_id": [1], "numerai_corr_jupiter_60": [0.1], "observation_count": [2]})
        ncorr = pd.DataFrame({"date": ["2019-01-04"], "fold_id": [1], "ncorr": [0.2], "observation_count": [2]})
        with patch("src.ensemble.calculate_v6_numerai_corr_by_date", return_value=legacy), patch(
            "src.ensemble.calculate_v6_ncorr_by_date", return_value=ncorr
        ), patch("src.ensemble.summarize_v6_oof_corr", return_value={"overall": {"metric": "NUMERAI_CORR_JUPITER_60"}}), patch(
            "src.ensemble.summarize_v6_ncorr", return_value={"overall": {"metric": "NCORR"}}
        ), patch("src.ensemble.combine_v6_metric_summaries", return_value={"approved": True}):
            per_date, metrics = score_ensemble_predictions(predictions, target)
        self.assertEqual(metrics, {"approved": True})
        self.assertEqual(per_date.loc[0, "ncorr"], 0.2)

    def test_artifact_round_trip_overwrite_and_v5_compatibility(self):
        policy = APPROVED_ENSEMBLE_POLICIES["medium_top250_rank_equal_v1"]
        predictions = pd.DataFrame(
            {
                "date": ["2019-01-04"], "numerai_ticker": ["a"],
                "composite_figi": ["x"], "fold_id": [1], "prediction": [0.5],
            }
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "v6.0" / "experiments"
            for experiment_id, _ in policy.components:
                directory = source / experiment_id
                directory.mkdir(parents=True)
                (directory / "metadata.json").write_text("{}", encoding="utf-8")
                (directory / "oof_predictions.parquet").write_bytes(b"source")
            saved = save_ensemble_artifacts(
                "synthetic", policy, predictions, source, root,
                git_state={"commit": "abc", "dirty": True},
            )
            loaded = load_ensemble_artifacts(saved)
            self.assertFalse(loaded["metadata"]["scored"])
            self.assertFalse(loaded["metadata"]["contains_trained_models"])
            self.assertEqual(loaded["metadata"]["target_columns_accessed_for_construction"], [])
            with self.assertRaises(FileExistsError):
                save_ensemble_artifacts("synthetic", policy, predictions, source, root)
            per_date = pd.DataFrame(
                {"date": ["2019-01-04"], "fold_id": [1], "ncorr": [0.1],
                 "numerai_corr_jupiter_60": [0.2], "observation_count": [1]}
            )
            scored = save_ensemble_artifacts(
                "scored", policy, predictions, source, root,
                per_date_metrics=per_date, metrics={"ncorr": {}},
                git_state={"commit": "abc", "dirty": True},
            )
            comparable = load_comparable_v6_artifacts(scored)
            self.assertEqual(len(comparable["oof_predictions"]), 1)
            self.assertEqual(len(comparable["per_date_metrics"]), 1)
        self.assertEqual(DATASET_VERSION, "v5.3")
        self.assertEqual(TARGET_COLUMN, "target")


if __name__ == "__main__":
    unittest.main()
