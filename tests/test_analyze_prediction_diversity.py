import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.analyze_prediction_diversity import (
    align_saved_predictions,
    calculate_per_date_diversity,
    save_diversity_artifacts,
    summarize_diversity,
)
from src.config import DATASET_VERSION, TARGET_COLUMN


def artifact(label: str, predictions: list[float], shuffled: bool = False) -> dict:
    frame = pd.DataFrame(
        {
            "date": ["2019-01-04"] * 10 + ["2020-01-03"] * 10,
            "numerai_ticker": [f"ticker_{index:02d}" for index in range(10)] * 2,
            "composite_figi": [f"figi_{index:02d}" for index in range(10)] * 2,
            "fold_id": [1] * 10 + [2] * 10,
            "prediction": predictions,
        }
    )
    if shuffled:
        frame = frame.sample(frac=1, random_state=42).reset_index(drop=True)
    return {
        "metadata": {
            "experiment_id": f"experiment_{label}",
            "dataset_version": "v6.0",
            "target": "target_jupiter_60",
            "folds": [{"fold": 1}, {"fold": 2}],
            "metric_identity": ["NCORR", "NUMERAI_CORR_JUPITER_60"],
            "ncorr_reference": {"metric_version": 1, "sha256": "abc"},
            "model_family": "test",
            "feature_set": "medium",
        },
        "oof_predictions": frame,
    }


class PredictionDiversityTest(unittest.TestCase):
    def setUp(self) -> None:
        increasing = list(range(10)) * 2
        reversed_values = list(reversed(range(10))) * 2
        shifted = list(range(1, 11)) * 2
        self.artifacts = {
            "A": artifact("A", increasing),
            "B": artifact("B", shifted, shuffled=True),
            "C": artifact("C", reversed_values),
        }

    def test_exact_alignment_accepts_shuffled_physical_order_without_targets(self) -> None:
        aligned, diagnostics = align_saved_predictions(self.artifacts)
        self.assertEqual(len(aligned), 20)
        self.assertNotIn("target_jupiter_60", aligned.columns)
        self.assertEqual(diagnostics["A"]["missing_or_nonfinite_predictions"], 0)
        self.assertTrue(
            np.array_equal(
                aligned["prediction_B"].to_numpy(),
                aligned["prediction_A"].to_numpy() + 1,
            )
        )

    def test_pearson_spearman_rank_disagreement_and_decile_overlap(self) -> None:
        aligned, _ = align_saved_predictions(self.artifacts)
        per_date = calculate_per_date_diversity(aligned)
        ab = per_date[per_date["pair"].eq("A_vs_B")]
        ac = per_date[per_date["pair"].eq("A_vs_C")]
        np.testing.assert_allclose(ab["pearson"], 1.0)
        np.testing.assert_allclose(ab["spearman"], 1.0)
        np.testing.assert_allclose(ab["mean_absolute_rank_disagreement"], 0.0)
        np.testing.assert_allclose(ab["top_decile_overlap_percentage"], 100.0)
        np.testing.assert_allclose(ac["pearson"], -1.0)
        np.testing.assert_allclose(ac["spearman"], -1.0)
        np.testing.assert_allclose(ac["top_decile_overlap_percentage"], 0.0)
        np.testing.assert_allclose(ac["bottom_decile_overlap_percentage"], 0.0)

    def test_missing_and_constant_prediction_diagnostics(self) -> None:
        aligned, _ = align_saved_predictions(self.artifacts)
        aligned.loc[0, "prediction_B"] = np.nan
        aligned.loc[aligned["date"].eq("2020-01-03"), "prediction_C"] = 5.0
        per_date = calculate_per_date_diversity(aligned)
        first = per_date[
            per_date["date"].eq("2019-01-04") & per_date["pair"].eq("A_vs_B")
        ].iloc[0]
        constant = per_date[
            per_date["date"].eq("2020-01-03") & per_date["pair"].eq("A_vs_C")
        ].iloc[0]
        self.assertEqual(first["missing_right"], 1)
        self.assertEqual(first["aligned_observations"], 9)
        self.assertTrue(constant["constant_right"])
        self.assertTrue(np.isnan(constant["pearson"]))

    def test_fold_and_year_summaries(self) -> None:
        aligned, _ = align_saved_predictions(self.artifacts)
        per_date = calculate_per_date_diversity(aligned)
        folds = summarize_diversity(per_date, "fold_id")
        years = summarize_diversity(per_date, "year")
        self.assertEqual(len(folds), 6)
        self.assertEqual(len(years), 6)
        self.assertEqual(set(folds["date_count"]), {1})
        self.assertIn("spearman_above_0_9_percentage", folds.columns)

    def test_duplicate_and_mismatched_identity_rejection(self) -> None:
        duplicate = {key: value.copy() for key, value in self.artifacts.items()}
        duplicate["A"] = {**self.artifacts["A"]}
        duplicate["A"]["oof_predictions"] = pd.concat(
            [
                self.artifacts["A"]["oof_predictions"],
                self.artifacts["A"]["oof_predictions"].iloc[[0]],
            ],
            ignore_index=True,
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            align_saved_predictions(duplicate)
        mismatch = {key: value.copy() for key, value in self.artifacts.items()}
        mismatch["C"] = {**self.artifacts["C"]}
        mismatch_frame = self.artifacts["C"]["oof_predictions"].copy()
        mismatch_frame.loc[0, "numerai_ticker"] = "different"
        mismatch["C"]["oof_predictions"] = mismatch_frame
        with self.assertRaisesRegex(ValueError, "identities"):
            align_saved_predictions(mismatch)

    def test_artifact_integrity_overwrite_protection_and_v5_compatibility(self) -> None:
        aligned, diagnostics = align_saved_predictions(self.artifacts)
        per_date = calculate_per_date_diversity(aligned)
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir)
            path = save_diversity_artifacts(
                "synthetic_diversity",
                {label: f"experiment_{label}" for label in ("A", "B", "C")},
                self.artifacts,
                aligned,
                diagnostics,
                per_date,
                output_root,
            )
            expected = {
                "COMPLETED.json",
                "metadata.json",
                "research_summary.json",
                "per_date_correlations.parquet",
                "fold_summaries.parquet",
                "annual_summaries.parquet",
                "rank_disagreement.parquet",
                "decile_overlap.parquet",
            }
            self.assertEqual({item.name for item in path.iterdir()}, expected)
            with self.assertRaises(FileExistsError):
                save_diversity_artifacts(
                    "synthetic_diversity",
                    {label: f"experiment_{label}" for label in ("A", "B", "C")},
                    self.artifacts,
                    aligned,
                    diagnostics,
                    per_date,
                    output_root,
                )
        self.assertEqual(DATASET_VERSION, "v5.3")
        self.assertEqual(TARGET_COLUMN, "target")


if __name__ == "__main__":
    unittest.main()
