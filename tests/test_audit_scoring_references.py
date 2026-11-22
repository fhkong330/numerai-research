import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.audit_scoring_references import parquet_summary, verify_row_alignment


class ScoringReferenceAuditTest(unittest.TestCase):
    def _write(self, path: Path, tickers: list[str], dates: list[str]) -> None:
        pq.write_table(
            pa.table(
                {
                    "numerai_ticker": tickers,
                    "date": dates,
                    "composite_figi": [f"figi_{i}" for i in range(len(tickers))],
                    "neutralizer_column_0": pa.array(
                        [0.1] * len(tickers), type=pa.float32()
                    ),
                }
            ),
            path,
        )

    def test_summary_counts_neutralizer_columns_and_hashes_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "reference.parquet"
            self._write(path, ["a"], ["2019-01-04"])

            summary = parquet_summary(path)

            self.assertEqual(summary["row_count"], 1)
            self.assertEqual(summary["neutralizer_column_count"], 1)
            self.assertEqual(len(summary["sha256"]), 64)

    def test_alignment_compares_identity_without_targets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "validation.parquet"
            second = root / "reference.parquet"
            tickers = ["a", "b"]
            dates = ["2019-01-04", "2019-01-11"]
            self._write(first, tickers, dates)
            self._write(second, tickers, dates)

            result = verify_row_alignment(first, second, batch_size=1)

            self.assertTrue(result["row_order_and_identity_match"])
            self.assertEqual(result["rows_compared"], 2)

    def test_alignment_rejects_mismatched_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "validation.parquet"
            second = root / "reference.parquet"
            self._write(first, ["a"], ["2019-01-04"])
            self._write(second, ["different"], ["2019-01-04"])

            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                verify_row_alignment(first, second)


if __name__ == "__main__":
    unittest.main()
