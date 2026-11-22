import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from numerai_tools.scoring import neutral_correlation as official_neutral_correlation

from src.evaluation import calculate_v6_ncorr_by_date, summarize_v6_ncorr
from src.scoring_references import load_validation_neutralizers_for_date


class OfficialNCORRTest(unittest.TestCase):
    date = "2019-01-04"

    def make_inputs(self, root: Path, rows: int = 40) -> tuple[pd.DataFrame, Path]:
        x = np.linspace(-1.0, 1.0, rows)
        identity = {
            "date": [self.date] * rows,
            "numerai_ticker": [f"ticker_{i:03d}" for i in range(rows)],
            "composite_figi": [f"figi_{i:03d}" for i in range(rows)],
        }
        reference = pd.DataFrame(
            {
                **identity,
                "neutralizer_column_0": x.astype("float32"),
                "neutralizer_column_1": np.sin(3 * x).astype("float32"),
                "neutralizer_column_2": np.cos(5 * x).astype("float32"),
            }
        )
        path = root / "neutralizer.parquet"
        reference.to_parquet(path, index=False)
        target = np.clip((np.sin(7 * x) + 1) / 2, 0, 1)
        oof = pd.DataFrame(
            {
                **identity,
                "fold_id": [1] * rows,
                "prediction": (np.cos(4 * x) + 1) / 2,
                "target_jupiter_60": target,
            },
            index=pd.Index(range(100, 100 + rows), name="source_row"),
        )
        return oof, path

    def test_matches_official_function_and_uses_every_column(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            oof, path = self.make_inputs(Path(temp_dir))
            neutralizers = load_validation_neutralizers_for_date(
                self.date, oof[["date", "numerai_ticker", "composite_figi"]],
                path, expected_count=3,
            ).reset_index(drop=True)
            predictions = pd.DataFrame({"prediction": oof["prediction"].to_numpy()})
            target = pd.Series(oof["target_jupiter_60"].to_numpy())
            expected = float(
                official_neutral_correlation(
                    predictions, neutralizers, target
                ).iloc[0]
            )
            seen = {}

            def capture(predictions, neutralizers, targets):
                seen["columns"] = neutralizers.columns.tolist()
                return official_neutral_correlation(predictions, neutralizers, targets)

            with patch("src.evaluation.neutral_correlation", side_effect=capture):
                actual = calculate_v6_ncorr_by_date(
                    oof, path, expected_neutralizer_count=3
                )

            self.assertAlmostEqual(actual.loc[0, "ncorr"], expected, delta=1e-12)
            self.assertEqual(
                seen["columns"],
                [f"neutralizer_column_{i}" for i in range(3)],
            )

    def test_alignment_rejects_missing_reordered_and_duplicate_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            oof, path = self.make_inputs(root)
            identity = oof[["date", "numerai_ticker", "composite_figi"]]
            for changed in (
                identity.iloc[:-1],
                identity.iloc[::-1],
            ):
                with self.assertRaisesRegex(ValueError, "exactly align"):
                    load_validation_neutralizers_for_date(
                        self.date, changed, path, expected_count=3
                    )

            missing_path = root / "missing.parquet"
            pq.write_table(pq.read_table(path).slice(0, len(identity) - 1), missing_path)
            with self.assertRaisesRegex(ValueError, "exactly align"):
                load_validation_neutralizers_for_date(
                    self.date, identity, missing_path, expected_count=3
                )

            table = pq.read_table(path)
            duplicate_path = root / "duplicate.parquet"
            pq.write_table(pa.concat_tables([table, table.slice(0, 1)]), duplicate_path)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_validation_neutralizers_for_date(
                    self.date, identity, duplicate_path, expected_count=3
                )

    def test_matching_null_composite_figi_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            oof, path = self.make_inputs(root)
            reference = pd.read_parquet(path)
            reference["composite_figi"] = pd.NA
            reference.to_parquet(path, index=False)
            identity = oof[["date", "numerai_ticker", "composite_figi"]].copy()
            identity["composite_figi"] = pd.NA

            neutralizers = load_validation_neutralizers_for_date(
                self.date, identity, path, expected_count=3
            )

            self.assertEqual(len(neutralizers), len(identity))

    def test_invalid_values_dates_and_degenerate_scores_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            oof, path = self.make_inputs(root)
            invalid = oof.copy()
            invalid.iloc[0, invalid.columns.get_loc("prediction")] = np.inf
            with self.assertRaisesRegex(ValueError, "present and finite"):
                calculate_v6_ncorr_by_date(invalid, path, 3)

            invalid = oof.copy()
            invalid.iloc[0, invalid.columns.get_loc("target_jupiter_60")] = np.nan
            with self.assertRaisesRegex(ValueError, "present and finite"):
                calculate_v6_ncorr_by_date(invalid, path, 3)

            unsupported = oof.copy()
            unsupported["date"] = "2025-08-22"
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                calculate_v6_ncorr_by_date(unsupported, path, 3)

            degenerate = oof.copy()
            degenerate["prediction"] = 0.5
            with self.assertRaisesRegex(ValueError, "invalid or degenerate"):
                calculate_v6_ncorr_by_date(degenerate, path, 3)

    def test_per_date_aggregation_preserves_metric_identity(self) -> None:
        per_date = pd.DataFrame(
            {
                "date": ["2019-01-04", "2019-01-11"],
                "fold_id": [1, 1],
                "ncorr": [0.1, -0.05],
                "observation_count": [100, 120],
            }
        )
        result = summarize_v6_ncorr(per_date)
        self.assertEqual(result["overall"]["metric"], "NCORR")
        self.assertEqual(result["overall"]["api_display_name"], "neutral_corr")
        self.assertEqual(result["overall"]["metric_version"], 1)
        self.assertAlmostEqual(result["overall"]["mean"], 0.025)


if __name__ == "__main__":
    unittest.main()
