import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.audit_data import (
    evenly_spaced_indices,
    parquet_schema_summary,
    summarize_date_gaps,
    write_report,
)


class DataAuditHelperTest(unittest.TestCase):
    def test_evenly_spaced_indices_are_bounded(self) -> None:
        self.assertEqual(evenly_spaced_indices(10, 3), [0, 4, 9])
        self.assertEqual(evenly_spaced_indices(3, 5), [0, 1, 2])
        self.assertEqual(evenly_spaced_indices(0, 5), [])

    def test_date_gap_summary_uses_observed_dates(self) -> None:
        summary = summarize_date_gaps(
            ["2026-01-01", "2026-01-08", "2026-01-22"]
        )

        self.assertEqual(summary["gap_count"], 2)
        self.assertEqual(summary["gap_days_counts"], {"7": 1, "14": 1})
        self.assertEqual(summary["median_gap_days"], 10.5)

    def test_report_writer_refuses_to_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "audit" / "audit.json"
            write_report({"dataset_version": "v6.0"}, report_path)

            with self.assertRaises(FileExistsError):
                write_report({"dataset_version": "v6.0"}, report_path)

    def test_schema_summary_classifies_features_and_targets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            parquet_path = Path(temp_dir) / "sample.parquet"
            pq.write_table(
                pa.table(
                    {
                        "date": ["2026-01-01"],
                        "data_type": ["train"],
                        "numerai_ticker": ["row_1"],
                        "feature_integer": pa.array([1], type=pa.int8()),
                        "feature_float": pa.array([0.5], type=pa.float32()),
                        "target": pa.array([0.25], type=pa.float32()),
                    }
                ),
                parquet_path,
            )

            summary = parquet_schema_summary(pq.ParquetFile(parquet_path))

            self.assertEqual(summary["row_count"], 1)
            self.assertEqual(summary["feature_physical_types"], {"FLOAT": 1, "INT32": 1})
            self.assertEqual(summary["target_physical_types"], {"FLOAT": 1})


if __name__ == "__main__":
    unittest.main()
