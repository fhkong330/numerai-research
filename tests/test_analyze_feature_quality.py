import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.analyze_feature_quality import analyze_feature_quality, write_analysis


class FeatureQualityAnalysisTest(unittest.TestCase):
    def _inputs(self):
        features = [f"feature_{index:03d}" for index in range(350)]
        small = features[:100]
        dates = pd.date_range("2023-07-14", periods=60, freq="7D").strftime(
            "%Y-%m-%d"
        )
        per_date_rows = []
        for date_number, date in enumerate(dates):
            for feature_number, feature in enumerate(features):
                sign = -1 if feature_number % 2 else 1
                per_date_rows.append(
                    {
                        "date": date,
                        "feature": feature,
                        "ic": sign * (0.001 + feature_number / 100000),
                        "valid_pairs": 10,
                        "total_rows": 10,
                        "coverage": 1.0,
                        "status": "valid",
                    }
                )
        summaries = []
        for fold_id in range(1, 6):
            ordered = features if fold_id < 5 else list(reversed(features))
            values = {feature: (rank + 1) / 10000 for rank, feature in enumerate(ordered)}
            for feature in features:
                mean = values[feature] * (-1 if int(feature[-3:]) % 2 else 1)
                summaries.append(
                    {
                        "fold_id": fold_id,
                        "feature": feature,
                        "train_start": "2003-01-03",
                        "train_end": dates[-1] if fold_id == 5 else dates[fold_id],
                        "training_dates": 60,
                        "valid_dates": 60,
                        "valid_date_coverage": 1.0,
                        "mean_ic": mean,
                        "median_ic": mean,
                        "ic_std": 0.01,
                        "absolute_mean_ic": abs(mean),
                        "mean_to_std_ratio": mean / 0.01,
                        "sign_consistency": 0.75,
                    }
                )
        rolling = pd.DataFrame(
            {
                "fold_id": 5,
                "date": dates[-1],
                "feature": features,
                "rolling_window_dates": 52,
                "rolling_mean_ic": [
                    (0.002 if index % 2 == 0 else -0.002)
                    for index in range(350)
                ],
            }
        )
        return (
            {"analysis_id": "synthetic", "ordered_feature_names": features},
            pd.DataFrame(per_date_rows),
            pd.DataFrame(summaries),
            rolling,
            small,
        )

    def test_analysis_partitions_groups_and_reports_rank_change(self) -> None:
        result = analyze_feature_quality(*self._inputs())
        self.assertEqual(result["feature_count"], 350)
        self.assertEqual(result["groups"]["small"]["feature_count"], 100)
        self.assertEqual(result["groups"]["medium_only"]["feature_count"], 250)
        self.assertEqual(result["positive_mean_features"], 175)
        self.assertEqual(result["negative_mean_features"], 175)
        self.assertEqual(result["status_counts"], {"valid": 21000})
        self.assertLess(result["fold_rank_stability"]["fold_1_to_5_spearman"], 0)
        self.assertEqual(len(result["top_absolute_mean_ic"]), 15)

    def test_analysis_writer_refuses_overwrite(self) -> None:
        result = analyze_feature_quality(*self._inputs())
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "analysis.json"
            write_analysis(result, path)
            self.assertTrue(path.is_file())
            with self.assertRaises(FileExistsError):
                write_analysis(result, path)


if __name__ == "__main__":
    unittest.main()
