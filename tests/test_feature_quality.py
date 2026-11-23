import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.config import DATASET_VERSION, TARGET_COLUMN
from src.feature_quality import (
    FeatureQualitySpec,
    approved_v6_feature_quality_folds,
    build_fold_feature_quality_summaries,
    calculate_cross_sectional_spearman_ic,
    load_feature_quality_artifacts,
    save_feature_quality_artifacts,
    summarize_feature_ic_as_of,
)


class FeatureQualityTest(unittest.TestCase):
    def test_exact_spearman_ties_and_sign_preservation(self) -> None:
        frame = pd.DataFrame(
            {
                "date": ["2020-01-03"] * 6,
                "feature_positive": [0, 1, 1, 2, 3, 3],
                "feature_negative": [3, 3, 2, 1, 1, 0],
                "target_jupiter_60": [0.1, 0.2, 0.4, 0.3, 0.8, 0.9],
            }
        )
        result = calculate_cross_sectional_spearman_ic(
            frame, ["feature_positive", "feature_negative"]
        ).set_index("feature")
        for feature in ("feature_positive", "feature_negative"):
            expected = spearmanr(frame[feature], frame["target_jupiter_60"]).statistic
            self.assertAlmostEqual(result.loc[feature, "ic"], expected, places=14)
        self.assertGreater(result.loc["feature_positive", "ic"], 0)
        self.assertLess(result.loc["feature_negative", "ic"], 0)

    def test_missing_constant_and_insufficient_policies(self) -> None:
        frame = pd.DataFrame(
            {
                "date": ["2020-01-03"] * 5,
                "missing": [1.0, np.nan, 3.0, 4.0, np.nan],
                "constant": [2, 2, 2, 2, 2],
                "insufficient": [1.0, np.nan, np.nan, np.nan, 2.0],
                "zero_ic": [1.0, 2.0, 3.0, 2.0, 1.0],
                "target_jupiter_60": [0.1, 0.2, np.nan, 0.4, 0.5],
            }
        )
        result = calculate_cross_sectional_spearman_ic(
            frame,
            ["missing", "constant", "insufficient", "zero_ic"],
            min_valid_pairs=3,
        ).set_index("feature")
        self.assertEqual(result.loc["missing", "valid_pairs"], 2)
        self.assertEqual(result.loc["missing", "status"], "insufficient_pairs")
        self.assertTrue(np.isnan(result.loc["missing", "ic"]))
        self.assertEqual(result.loc["constant", "status"], "constant_feature")
        self.assertTrue(np.isnan(result.loc["constant", "ic"]))
        self.assertEqual(result.loc["insufficient", "status"], "insufficient_pairs")
        self.assertNotEqual(result.loc["constant", "ic"], 0.0)
        self.assertEqual(result.loc["zero_ic", "status"], "valid")
        self.assertAlmostEqual(result.loc["zero_ic", "ic"], 0.0, places=14)

    @staticmethod
    def _ic_rows(dates, feature="feature_a") -> pd.DataFrame:
        return pd.DataFrame(
            {
                "date": list(dates),
                "feature": feature,
                "ic": [0.1 if index % 3 else -0.05 for index, _ in enumerate(dates)],
                "valid_pairs": 100,
                "total_rows": 100,
                "coverage": 1.0,
                "status": "valid",
            }
        )

    def test_fold_as_of_summary_and_rolling_window(self) -> None:
        fold = approved_v6_feature_quality_folds()[0]
        per_date = self._ic_rows(fold.train_dates)
        result = summarize_feature_ic_as_of(per_date, ["feature_a"], fold)
        summary = result["summary"].iloc[0]
        self.assertEqual(summary["training_dates"], len(fold.train_dates))
        self.assertEqual(summary["valid_date_coverage"], 1.0)
        self.assertAlmostEqual(summary["absolute_mean_ic"], abs(summary["mean_ic"]))
        rolling = result["rolling"]["rolling_mean_ic"]
        self.assertTrue(rolling.iloc[:51].isna().all())
        self.assertTrue(np.isfinite(rolling.iloc[51]))
        self.assertFalse(result["annual"].empty)

    def test_fold_as_of_rejects_validation_purge_holdout_and_future_dates(self) -> None:
        folds = approved_v6_feature_quality_folds()
        fold = folds[0]
        base = self._ic_rows(fold.train_dates)
        forbidden_dates = (
            fold.purge_dates[0],
            fold.validation_dates[0],
            folds[1].train_dates[-1],
            "2025-05-02",
            "2025-08-22",
            "2026-08-21",
        )
        for forbidden in forbidden_dates:
            with self.subTest(forbidden=forbidden):
                bad = pd.concat([base, self._ic_rows([forbidden])], ignore_index=True)
                with self.assertRaisesRegex(ValueError, "outside this fold"):
                    summarize_feature_ic_as_of(bad, ["feature_a"], fold)

    def test_fold_summary_builder_filters_each_cutoff_internally(self) -> None:
        folds = approved_v6_feature_quality_folds()
        eligible = sorted(set().union(*(set(fold.train_dates) for fold in folds)))
        result = build_fold_feature_quality_summaries(
            self._ic_rows(eligible), ["feature_a"], folds
        )
        self.assertEqual(result["summary"]["fold_id"].tolist(), [1, 2, 3, 4, 5])
        self.assertEqual(
            result["summary"]["train_end"].tolist(),
            [fold.train_dates[-1] for fold in folds],
        )
        bad = pd.concat(
            [self._ic_rows(eligible), self._ic_rows(["2025-08-22"])],
            ignore_index=True,
        )
        with self.assertRaisesRegex(ValueError, "non-training dates"):
            build_fold_feature_quality_summaries(bad, ["feature_a"], folds)

    def test_artifact_round_trip_and_overwrite_protection(self) -> None:
        folds = approved_v6_feature_quality_folds()
        eligible = sorted(set().union(*(set(fold.train_dates) for fold in folds)))
        per_date = self._ic_rows(eligible)
        outputs = build_fold_feature_quality_summaries(
            per_date, ["feature_a"], folds
        )
        spec = FeatureQualitySpec("synthetic_ic", "synthetic", ["feature_a"])
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = save_feature_quality_artifacts(
                spec,
                per_date,
                outputs,
                folds,
                output_root=root,
                git_state={"commit": "test", "dirty": False},
            )
            loaded = load_feature_quality_artifacts(path)
            self.assertEqual(loaded["metadata"]["target"], "target_jupiter_60")
            self.assertEqual(loaded["metadata"]["ordered_feature_names"], ["feature_a"])
            pd.testing.assert_frame_equal(loaded["per_date_ic"], per_date)
            self.assertTrue((path / "COMPLETED.json").is_file())
            with self.assertRaises(FileExistsError):
                save_feature_quality_artifacts(
                    spec, per_date, outputs, folds, output_root=root
                )

    def test_historical_v5_defaults_remain_unchanged(self) -> None:
        self.assertEqual(DATASET_VERSION, "v5.3")
        self.assertEqual(TARGET_COLUMN, "target")


if __name__ == "__main__":
    unittest.main()
