import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.audit_production_data import (
    build_report,
    derive_training_eligibility,
    fetch_public_metadata,
    snapshot_compatibility,
    summarize_date_targets,
    validate_frozen_strategy,
    write_report,
)


class ProductionDataAuditTest(unittest.TestCase):
    def _summary(self) -> pd.DataFrame:
        dates = pd.date_range("2026-01-02", periods=20, freq="7D")
        frame = pd.DataFrame(
            {
                "date": dates.repeat(2),
                "target_jupiter_60": [0.25, 0.75] * 20,
            }
        )
        return summarize_date_targets([frame])

    def test_observed_date_purge_and_cutoff(self) -> None:
        result = derive_training_eligibility(
            self._summary(), "2026-05-22", purge_dates=4
        )
        self.assertEqual(result["eligible_training_dates"], 16)
        self.assertEqual(result["eligible_training_cutoff"], "2026-04-17")
        self.assertEqual(
            result["purge_dates"],
            ["2026-04-24", "2026-05-01", "2026-05-08", "2026-05-15"],
        )

    def test_non_mature_target_inside_eligible_window_is_rejected(self) -> None:
        summary = self._summary()
        summary.loc[2, ["nonnull_rows", "finite_rows", "fully_mature"]] = [0, 0, False]
        with self.assertRaisesRegex(ValueError, "Non-mature targets"):
            derive_training_eligibility(summary, "2026-05-22", purge_dates=4)

    def test_recent_missing_targets_are_reported_but_excluded_by_purge(self) -> None:
        summary = self._summary()
        summary.loc[18:, ["nonnull_rows", "finite_rows", "fully_mature"]] = [0, 0, False]
        result = derive_training_eligibility(summary, "2026-05-22", purge_dates=4)
        self.assertEqual(result["non_mature_date_count"], 2)
        self.assertEqual(result["eligible_training_dates"], 16)

    def test_frozen_strategy_rejects_feature_identity_change(self) -> None:
        with self.assertRaisesRegex(ValueError, "350 unique"):
            validate_frozen_strategy(["feature_one"])

    def test_snapshot_compatibility_rejects_historical_revision(self) -> None:
        expected = {"train.parquet": "abc", "validation.parquet": "def"}
        actual = {
            "train.parquet": {"sha256": "abc"},
            "validation.parquet": {"sha256": "changed"},
        }
        result = snapshot_compatibility(actual, expected)
        self.assertFalse(result["exact_research_snapshot_match"])
        self.assertFalse(result["historical_ic_reuse_allowed"])

    def test_public_metadata_is_bounded_to_requested_round(self) -> None:
        class FakeAPI:
            def list_datasets(self):
                return ["v6.0/live.parquet", "v5.3/live.parquet"]

            def list_rounds(self, number):
                stamp = pd.Timestamp("2026-11-21T13:00:00Z").to_pydatetime()
                return [{
                    "number": number, "target": "target_jupiter_60",
                    "dataDatestamp": 20261120, "openTime": stamp,
                    "closeTime": stamp, "closeStakingTime": stamp,
                    "scoreTime": stamp, "resolveTime": stamp,
                }]

            def get_current_round(self):
                return 1404

        with patch("src.audit_production_data.NumerAPI", return_value=FakeAPI()):
            result = fetch_public_metadata(1404)
        self.assertEqual(result["v6_catalog"], ["v6.0/live.parquet"])
        self.assertEqual(result["round"]["data_datestamp"], 20261120)

    def test_report_writer_preserves_history_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "audit.json"
            write_report({"audit_version": 1}, path)
            self.assertEqual(json.loads(path.read_text()), {"audit_version": 1})
            with self.assertRaises(FileExistsError):
                write_report({"audit_version": 2}, path)

    def test_as_of_timestamp_requires_timezone(self) -> None:
        with patch("src.audit_production_data.audit_local_dataset", return_value={}):
            with self.assertRaisesRegex(ValueError, "timezone"):
                build_report(Path("unused"), "2026-11-23T12:00:00", 1404, "2026-11-20", False)


if __name__ == "__main__":
    unittest.main()
