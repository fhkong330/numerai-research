import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.profile_feature_memory import (
    MAX_PROFILE_ROWS,
    load_bounded_training_sample,
    main,
    profile_feature_memory,
    validate_sample_rows,
)


class FeatureMemoryProfileTest(unittest.TestCase):
    def make_data(self, root: Path, rows: int = 12) -> None:
        data = pd.DataFrame(
            {
                "numerai_ticker": [f"ticker_{i}" for i in range(rows)],
                "composite_figi": [f"figi_{i}" for i in range(rows)],
                "date": ["2003-01-03"] * rows,
                "data_type": ["train"] * rows,
                "target_jupiter_60": [0.5] * rows,
                "feature_int": pd.Series(range(rows), dtype="int8"),
                "feature_float": pd.Series(range(rows), dtype="float32"),
                "feature_country": ["US"] * rows,
            }
        )
        data.to_parquet(root / "train.parquet", index=False)
        # Deliberately unusable validation content proves the bounded loader is Train-only.
        pd.DataFrame({"holdout_target_trap": [1]}).to_parquet(
            root / "validation.parquet", index=False
        )
        (root / "features.json").write_text(
            json.dumps(
                {
                    "feature_sets": {
                        "all": ["feature_int", "feature_float", "feature_country"]
                    }
                }
            ),
            encoding="utf-8",
        )

    def test_sample_limit_is_enforced(self) -> None:
        self.assertEqual(validate_sample_rows(100), 100)
        with self.assertRaises(ValueError):
            validate_sample_rows(MAX_PROFILE_ROWS + 1)

    def test_train_only_sample_preserves_dtypes_and_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.make_data(root)
            sample = load_bounded_training_sample(
                root, ["feature_int", "feature_float"], 5
            )
        self.assertEqual(len(sample), 5)
        self.assertEqual(str(sample["feature_int"].dtype), "int8")
        self.assertEqual(str(sample["feature_float"].dtype), "float32")

    def test_report_schema_and_feature_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.make_data(root)
            report = profile_feature_memory(root, "all", 8, fit_lightgbm=False)
        self.assertEqual(report["sample"]["feature_count"], 2)
        self.assertLessEqual(report["sample"]["actual_rows"], 8)
        self.assertEqual(report["sample"]["dtype_conversions"], [])
        self.assertFalse(report["lightgbm_fit"]["executed"])
        self.assertIn("single_feature_matrix_bytes", report["fold5_extrapolation"])
        self.assertEqual(
            report["risk_assessment"]["classification"],
            "High risk / not recommended",
        )

    def test_cli_writes_once_and_never_trains_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.make_data(root)
            output = root / "reports" / "profile.json"
            argv = [
                "--version",
                "v6.0",
                "--feature-set",
                "all",
                "--max-sample-rows",
                "6",
                "--data-dir",
                str(root),
                "--output",
                str(output),
            ]
            main(argv)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertFalse(report["lightgbm_fit"]["executed"])
            with self.assertRaises(FileExistsError):
                main(argv)


if __name__ == "__main__":
    unittest.main()
