import tempfile
import unittest
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.config import (
    PURGE_ERAS,
    TARGET_COLUMN,
    WALK_FORWARD_VALIDATION_ERAS,
    get_research_config,
)
from src.data import (
    build_v6_research_history,
    partition_v6_validation_index,
)
from src.validation import generate_v6_walk_forward_folds, load_v6_fold_dates


def weekly_dates(start: str, periods: int) -> list[str]:
    return pd.date_range(start, periods=periods, freq="7D").strftime(
        "%Y-%m-%d"
    ).tolist()


def identity_frame(dates: list[str], data_type: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "numerai_ticker": [f"ticker_{index}" for index in range(len(dates))],
            "composite_figi": [f"figi_{index}" for index in range(len(dates))],
            "date": dates,
            "data_type": data_type,
        }
    )


class V6ResearchSplitTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = get_research_config("v6.0")
        self.historical_dates = weekly_dates(
            self.config.historical_start,
            self.config.expected_historical_dates,
        )
        self.research_dates = weekly_dates(
            self.config.research_start,
            self.config.expected_research_dates,
        )

    def test_v5_3_compatibility_defaults_are_unchanged(self) -> None:
        legacy = get_research_config("v5.3")
        self.assertEqual(legacy.target_column, "target")
        self.assertEqual(legacy.time_column, "era")
        self.assertEqual(PURGE_ERAS, 16)
        self.assertEqual(TARGET_COLUMN, "target")
        self.assertEqual(WALK_FORWARD_VALIDATION_ERAS, 156)

    def test_exact_v6_fold_boundaries(self) -> None:
        folds = generate_v6_walk_forward_folds(
            self.historical_dates,
            self.research_dates,
        )
        actual = [
            (
                fold.train_dates[-1],
                fold.purge_dates[0],
                fold.purge_dates[-1],
                fold.validation_dates[0],
                fold.validation_dates[-1],
                len(fold.validation_dates),
            )
            for fold in folds
        ]
        self.assertEqual(
            actual,
            [
                ("2018-09-07", "2018-09-14", "2018-12-28", "2019-01-04", "2020-06-26", 78),
                ("2020-03-06", "2020-03-13", "2020-06-26", "2020-07-03", "2021-12-24", 78),
                ("2021-09-03", "2021-09-10", "2021-12-24", "2021-12-31", "2023-06-23", 78),
                ("2023-03-03", "2023-03-10", "2023-06-23", "2023-06-30", "2024-12-20", 78),
                ("2024-08-30", "2024-09-06", "2024-12-20", "2024-12-27", "2025-04-25", 18),
            ],
        )

    def test_expanding_training_and_purge_are_disjoint(self) -> None:
        folds = generate_v6_walk_forward_folds(
            self.historical_dates,
            self.research_dates,
        )
        self.assertEqual([len(fold.train_dates) for fold in folds], [819, 897, 975, 1053, 1131])
        for fold in folds:
            self.assertEqual(len(fold.purge_dates), 16)
            self.assertFalse(set(fold.train_dates) & set(fold.validation_dates))
            self.assertFalse(set(fold.purge_dates) & set(fold.train_dates))
            self.assertFalse(set(fold.purge_dates) & set(fold.validation_dates))

    def test_validation_partitions_are_exact_and_preserve_identity(self) -> None:
        research = identity_frame(self.research_dates, "validation")
        gap = identity_frame(
            weekly_dates(
                self.config.holdout_gap_start,
                self.config.expected_holdout_gap_dates,
            ),
            "validation",
        )
        holdout = identity_frame(
            weekly_dates(
                self.config.holdout_start,
                self.config.expected_holdout_dates,
            ),
            "validation",
        )
        immature = identity_frame(
            weekly_dates(
                self.config.immature_start,
                self.config.expected_immature_dates,
            ),
            "test",
        )
        split = partition_v6_validation_index(
            pd.concat([research, gap, holdout, immature], ignore_index=True)
        )

        self.assertEqual(len(split.research_validation), 330)
        self.assertEqual(len(split.holdout_gap), 16)
        self.assertEqual(len(split.final_holdout), 52)
        self.assertEqual(len(split.immature), 13)
        self.assertIn("numerai_ticker", split.final_holdout)
        self.assertIn("composite_figi", split.final_holdout)
        self.assertNotIn(self.config.target_column, split.final_holdout)

    def test_research_history_rejects_null_targets(self) -> None:
        historical = identity_frame(self.historical_dates, "train")
        research = identity_frame(self.research_dates, "validation")
        historical[self.config.target_column] = 0.5
        research[self.config.target_column] = 0.5
        research.loc[0, self.config.target_column] = None

        with self.assertRaisesRegex(ValueError, "null target_jupiter_60"):
            build_v6_research_history(historical, research)

    def test_research_history_excludes_holdout_and_preserves_identity(self) -> None:
        historical = identity_frame(self.historical_dates, "train")
        research = identity_frame(self.research_dates, "validation")
        historical[self.config.target_column] = 0.5
        research[self.config.target_column] = 0.5

        history = build_v6_research_history(historical, research)

        self.assertEqual(len(history), 1165)
        self.assertEqual(history["date"].min(), self.config.historical_start)
        self.assertEqual(history["date"].max(), self.config.research_end)
        self.assertIn("numerai_ticker", history)
        self.assertIn("composite_figi", history)

    def test_unexpected_date_range_is_rejected(self) -> None:
        changed_dates = [*self.research_dates, "2025-05-02"]
        with self.assertRaisesRegex(ValueError, "Unexpected research validation dates"):
            generate_v6_walk_forward_folds(
                self.historical_dates,
                changed_dates,
            )

    def test_unexpected_schema_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            incomplete = pa.table({"date": [self.config.historical_start]})
            pq.write_table(incomplete, data_dir / "train.parquet")
            pq.write_table(incomplete, data_dir / "validation.parquet")

            with self.assertRaisesRegex(ValueError, "missing required columns"):
                load_v6_fold_dates(data_dir)


if __name__ == "__main__":
    unittest.main()
