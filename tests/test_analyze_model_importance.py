import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.analyze_model_importance import (
    build_research_summary,
    extract_fold_importance,
    join_importance_with_fold_ic,
    save_model_importance_artifacts,
    summarize_cross_fold_importance,
    summarize_feature_groups,
)


class FakeBooster:
    def __init__(self, names, gain, split):
        self.names = names
        self.gain = gain
        self.split = split

    def feature_name(self):
        return self.names

    def feature_importance(self, importance_type):
        return self.gain if importance_type == "gain" else self.split


class ModelImportanceAnalysisTest(unittest.TestCase):
    def setUp(self) -> None:
        self.features = [f"feature_{index:03d}" for index in range(350)]
        self.small = self.features[:100]

    def _per_fold(self) -> pd.DataFrame:
        parts = []
        for fold_id in range(1, 6):
            gain = np.arange(350, dtype=float) + fold_id
            split = np.arange(350, dtype=float)[::-1] + fold_id
            gain[0] = 0
            split[0] = 0
            parts.append(
                extract_fold_importance(
                    FakeBooster(self.features, gain, split),
                    fold_id,
                    self.features,
                )
            )
        return pd.concat(parts, ignore_index=True)

    def test_extraction_normalization_ranks_and_zero_handling(self) -> None:
        gain = np.zeros(350)
        split = np.zeros(350)
        gain[:2] = [3, 1]
        split[:2] = [1, 3]
        frame = extract_fold_importance(
            FakeBooster(self.features, gain, split), 1, self.features
        )
        self.assertAlmostEqual(frame["normalized_gain_share"].sum(), 1.0)
        self.assertAlmostEqual(frame["normalized_split_share"].sum(), 1.0)
        self.assertEqual(frame.loc[0, "gain_rank"], 1.0)
        self.assertTrue(frame.loc[2:, "zero_importance"].all())
        self.assertEqual(frame.loc[2, "gain_rank"], frame.loc[349, "gain_rank"])

    def test_rejects_mismatched_feature_order(self) -> None:
        with self.assertRaisesRegex(ValueError, "feature names or order"):
            extract_fold_importance(
                FakeBooster(list(reversed(self.features)), np.ones(350), np.ones(350)),
                1,
                self.features,
            )

    def test_cross_fold_group_and_fold_specific_ic_join(self) -> None:
        per_fold = self._per_fold()
        cross_fold, correlations = summarize_cross_fold_importance(per_fold)
        groups = summarize_feature_groups(per_fold, self.small)
        self.assertEqual(len(cross_fold), 350)
        self.assertEqual(len(correlations), 10)
        self.assertTrue(cross_fold.loc[0, "never_used"])
        self.assertEqual(set(groups["feature_group"]), {"small", "medium_only"})
        for fold_id, group in groups.groupby("fold_id"):
            self.assertAlmostEqual(group["total_gain_share"].sum(), 1.0)
            self.assertEqual(group["feature_count"].sum(), 350)

        ic_rows = []
        for fold_id in range(1, 6):
            for index, feature in enumerate(self.features):
                ic_rows.append(
                    {
                        "fold_id": fold_id,
                        "feature": feature,
                        "absolute_mean_ic": (index + fold_id) / 10000,
                        "sign_consistency": 0.5 + index / 1000,
                        "train_start": "2003-01-03",
                        "train_end": f"202{fold_id}-01-01",
                    }
                )
        joined, relationships = join_importance_with_fold_ic(
            per_fold, pd.DataFrame(ic_rows)
        )
        self.assertEqual(len(joined), 1750)
        self.assertEqual(len(relationships), 5)
        self.assertEqual(
            joined.loc[joined["fold_id"].eq(3), "train_end"].unique().tolist(),
            ["2023-01-01"],
        )

        bad = pd.DataFrame(ic_rows[:-1])
        with self.assertRaisesRegex(ValueError, "do not align"):
            join_importance_with_fold_ic(per_fold, bad)

    def test_artifact_integrity_and_overwrite_protection(self) -> None:
        per_fold = self._per_fold()
        cross_fold, correlations = summarize_cross_fold_importance(per_fold)
        groups = summarize_feature_groups(per_fold, self.small)
        ic = per_fold[["fold_id", "feature"]].copy()
        ic["absolute_mean_ic"] = (
            ic["feature"].str.slice(-3).astype(int) + 1
        ) / 10000
        ic["sign_consistency"] = 0.5 + (
            ic["feature"].str.slice(-3).astype(int) / 1000
        )
        ic["train_start"] = "2003-01-03"
        ic["train_end"] = "2024-08-30"
        joined, relationships = join_importance_with_fold_ic(per_fold, ic)
        summary = build_research_summary(
            per_fold, cross_fold, correlations, groups, relationships
        )
        experiment_meta = {
            "experiment_id": "model_test",
            "target": "target_jupiter_60",
            "ordered_feature_names": self.features,
            "model_parameters": {"n_estimators": 3},
            "folds": [{"fold": index} for index in range(1, 6)],
        }
        quality_meta = {"analysis_id": "ic_test"}
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = save_model_importance_artifacts(
                "importance_test",
                experiment_meta,
                quality_meta,
                per_fold,
                cross_fold,
                groups,
                joined,
                correlations,
                relationships,
                summary,
                output_root=root,
                git_state={"commit": "test", "dirty": False},
            )
            self.assertTrue((path / "COMPLETED.json").is_file())
            self.assertEqual(json.loads((path / "metadata.json").read_text())["model_experiment_id"], "model_test")
            self.assertEqual(len(pd.read_parquet(path / "per_fold_importance.parquet")), 1750)
            with self.assertRaises(FileExistsError):
                save_model_importance_artifacts(
                    "importance_test",
                    experiment_meta,
                    quality_meta,
                    per_fold,
                    cross_fold,
                    groups,
                    joined,
                    correlations,
                    relationships,
                    summary,
                    output_root=root,
                )


if __name__ == "__main__":
    unittest.main()
