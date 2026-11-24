import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation import summarize_v6_oof_corr
from src.experiment_artifacts import (
    V6ExperimentSpec,
    load_v6_experiment_artifacts,
    save_v6_experiment_artifacts,
)
from src.feature_quality import approved_v6_feature_quality_folds
from src.feature_selection import (
    POLICY_ID,
    SELECTED_FEATURE_COUNT,
    build_top250_fold_manifests,
    select_fold_top250,
    summarize_fold_manifest_stability,
)


class FoldFeatureSelectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.features = [f"feature_{index:03d}" for index in range(350)]
        self.folds = approved_v6_feature_quality_folds()
        self.metadata = {
            "analysis_id": "synthetic_ic",
            "dataset_version": "v6.0",
            "target": "target_jupiter_60",
            "feature_set": "medium",
            "ordered_feature_names": self.features,
            "folds": [fold.summary() for fold in self.folds],
        }

    def _summary(self, fold, missing_count=0) -> pd.DataFrame:
        rows = []
        for index, feature in enumerate(self.features):
            mean_ic = ((-1) ** index) * (index + 1) / 10000
            rows.append(
                {
                    "fold_id": fold.fold,
                    "feature": feature,
                    "train_start": fold.train_dates[0],
                    "train_end": fold.train_dates[-1],
                    "training_dates": len(fold.train_dates),
                    "valid_dates": len(fold.train_dates),
                    "mean_ic": mean_ic,
                    "absolute_mean_ic": abs(mean_ic),
                }
            )
        frame = pd.DataFrame(rows)
        if missing_count:
            missing = frame.tail(missing_count).index
            frame.loc[missing, ["mean_ic", "absolute_mean_ic"]] = np.nan
            frame.loc[missing, "valid_dates"] = 0
        return frame

    def test_top250_uses_absolute_ic_and_preserves_model_order(self) -> None:
        fold = self.folds[0]
        manifest = select_fold_top250(
            self._summary(fold), fold, self.metadata, self.features
        )
        self.assertEqual(manifest.policy_id, POLICY_ID)
        self.assertEqual(len(manifest.model_feature_names), SELECTED_FEATURE_COUNT)
        self.assertIn("feature_349", manifest.model_feature_names)
        self.assertIn("feature_348", manifest.model_feature_names)
        self.assertNotIn("feature_000", manifest.model_feature_names)
        self.assertEqual(
            list(manifest.model_feature_names),
            [feature for feature in self.features if feature in manifest.model_feature_names],
        )
        self.assertLess(manifest.selected_ranked[0]["mean_ic"], 0)

    def test_ties_are_broken_by_feature_name_and_missing_rank_last(self) -> None:
        fold = self.folds[0]
        frame = self._summary(fold, missing_count=10)
        frame.loc[frame["feature"].isin(["feature_300", "feature_301"]), "mean_ic"] = [
            0.5,
            -0.5,
        ]
        frame.loc[
            frame["feature"].isin(["feature_300", "feature_301"]),
            "absolute_mean_ic",
        ] = 0.5
        manifest = select_fold_top250(frame, fold, self.metadata, self.features)
        self.assertEqual(
            [row["feature"] for row in manifest.selected_ranked[:2]],
            ["feature_300", "feature_301"],
        )
        self.assertTrue(
            set(frame.tail(10)["feature"]).isdisjoint(manifest.model_feature_names)
        )

    def test_rejects_insufficient_valid_features(self) -> None:
        fold = self.folds[0]
        with self.assertRaisesRegex(ValueError, "Fewer than 250"):
            select_fold_top250(
                self._summary(fold, missing_count=101),
                fold,
                self.metadata,
                self.features,
            )

    def test_rejects_inconsistent_ic_statistics(self) -> None:
        fold = self.folds[0]
        inconsistent = self._summary(fold)
        inconsistent.loc[0, "absolute_mean_ic"] = 1.0
        with self.assertRaisesRegex(ValueError, "internally inconsistent"):
            select_fold_top250(
                inconsistent, fold, self.metadata, self.features
            )
        impossible_coverage = self._summary(fold)
        impossible_coverage.loc[0, "valid_dates"] = len(fold.train_dates) + 1
        with self.assertRaisesRegex(ValueError, "valid-date counts"):
            select_fold_top250(
                impossible_coverage, fold, self.metadata, self.features
            )

    def test_rejects_later_fold_cutoff_and_mismatched_identity(self) -> None:
        fold = self.folds[0]
        later = self._summary(self.folds[1])
        with self.assertRaisesRegex(ValueError, "later or mismatched"):
            select_fold_top250(later, fold, self.metadata, self.features)
        wrong_cutoff = self._summary(fold)
        wrong_cutoff["train_end"] = "2025-08-22"
        with self.assertRaisesRegex(ValueError, "training cutoff"):
            select_fold_top250(wrong_cutoff, fold, self.metadata, self.features)
        bad_metadata = {**self.metadata, "target": "target"}
        with self.assertRaisesRegex(ValueError, "target_jupiter_60"):
            select_fold_top250(
                self._summary(fold), fold, bad_metadata, self.features
            )

    def test_builds_exact_fold_specific_manifests(self) -> None:
        summaries = pd.concat(
            [self._summary(fold) for fold in self.folds], ignore_index=True
        )
        manifests = build_top250_fold_manifests(
            summaries, self.metadata, self.features, self.folds
        )
        self.assertEqual(set(manifests), {1, 2, 3, 4, 5})
        for fold in self.folds:
            self.assertEqual(manifests[fold.fold].train_end, fold.train_dates[-1])
            self.assertEqual(len(manifests[fold.fold].selected_ranked), 250)

    def test_feature_manifests_save_and_reload_without_changing_baseline_contract(self) -> None:
        summaries = pd.concat(
            [self._summary(fold) for fold in self.folds], ignore_index=True
        )
        manifests = build_top250_fold_manifests(
            summaries, self.metadata, self.features, self.folds
        )
        rows = []
        for fold in self.folds:
            for date in fold.validation_dates:
                rows.append(
                    {
                        "date": date,
                        "numerai_ticker": f"ticker_{date}",
                        "composite_figi": f"figi_{date}",
                        "fold_id": fold.fold,
                        "prediction": 0.4 + fold.fold / 100,
                        "target_jupiter_60": 0.5,
                    }
                )
        oof = pd.DataFrame(rows)
        per_date = pd.DataFrame(
            [
                {
                    "date": row["date"],
                    "fold_id": row["fold_id"],
                    "numerai_corr_jupiter_60": 0.01,
                    "observation_count": 1,
                }
                for row in rows
            ]
        )
        metrics = summarize_v6_oof_corr(per_date)
        spec = V6ExperimentSpec(
            experiment_id="selected_synthetic",
            feature_set_name="medium",
            feature_names=self.features,
            model_family="LightGBM",
            model_parameters={"n_estimators": 3},
            random_seed=42,
            run_kind="synthetic_selection",
            feature_selection_policy_id=POLICY_ID,
            source_feature_quality_analysis_id="synthetic_ic",
            fold_feature_manifests=tuple(
                manifests[fold.fold].to_dict() for fold in self.folds
            ),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = save_v6_experiment_artifacts(
                spec,
                self.folds,
                oof,
                per_date,
                metrics,
                output_root=root,
                git_state={"commit": "test", "dirty": False},
            )
            loaded = load_v6_experiment_artifacts(path)
            self.assertEqual(len(loaded["feature_manifests"]), 5)
            self.assertEqual(
                loaded["metadata"]["feature_selection"]["policy_id"], POLICY_ID
            )
            self.assertEqual(
                loaded["feature_manifests"][0]["model_parameters"],
                {"n_estimators": 3},
            )

    def test_saved_manifest_stability_summary(self) -> None:
        summaries = pd.concat(
            [self._summary(fold) for fold in self.folds], ignore_index=True
        )
        manifests = build_top250_fold_manifests(
            summaries, self.metadata, self.features, self.folds
        )
        report = summarize_fold_manifest_stability(
            [manifests[index].to_dict() for index in range(1, 6)],
            self.features[:100],
        )
        self.assertEqual(report["shared_all_five"], 250)
        self.assertEqual(report["selected_only_one_fold"], 0)
        self.assertEqual(len(report["pairwise_jaccard"]), 10)
        self.assertEqual(report["pairwise_jaccard"][0]["jaccard"], 1.0)
        self.assertEqual(
            report["group_membership"][0],
            {"fold_id": 1, "small": 0, "medium_only": 250},
        )


if __name__ == "__main__":
    unittest.main()
