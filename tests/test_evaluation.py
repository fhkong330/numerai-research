import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.config import DEFAULT_LIGHTGBM_PARAMS, get_research_config
from src.evaluation import (
    calculate_era_correlations,
    calculate_v6_numerai_corr_by_date,
    combine_v6_metric_summaries,
    create_v6_oof_frame,
    numerai_corr,
    summarize_correlations,
    summarize_v6_oof_corr,
    summarize_v6_ncorr,
    v6_metric_availability,
    validate_v6_oof_predictions,
)
from src.experiment_artifacts import (
    V6ExperimentSpec,
    load_v6_experiment_artifacts,
    save_v6_experiment_artifacts,
)
from src.validation import generate_v6_walk_forward_folds


def weekly_dates(start: str, count: int) -> list[str]:
    return pd.date_range(start, periods=count, freq="7D").strftime(
        "%Y-%m-%d"
    ).tolist()


class V6OOFEvaluationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = get_research_config("v6.0")
        historical = weekly_dates(
            cls.config.historical_start,
            cls.config.expected_historical_dates,
        )
        research = weekly_dates(
            cls.config.research_start,
            cls.config.expected_research_dates,
        )
        cls.folds = generate_v6_walk_forward_folds(historical, research)
        cls.fold_by_date = {
            date: fold.fold for fold in cls.folds for date in fold.validation_dates
        }

    def make_oof(self, rows_per_date: int = 5) -> pd.DataFrame:
        rows = []
        index = []
        row_number = 0
        target_values = [0.0, 0.25, 0.5, 0.75, 1.0]
        for date_number, date in enumerate(sorted(self.fold_by_date)):
            for position in range(rows_per_date):
                target = target_values[position % len(target_values)]
                prediction = target if date_number % 3 else 1 - target
                rows.append(
                    {
                        "date": date,
                        "numerai_ticker": f"ticker_{date_number}_{position}",
                        "composite_figi": f"figi_{date_number}_{position}",
                        "fold_id": self.fold_by_date[date],
                        "prediction": prediction,
                        "target_jupiter_60": target,
                    }
                )
                index.append(row_number + 1000)
                row_number += 1
        return pd.DataFrame(rows, index=pd.Index(index, name="source_row"))

    def test_oof_frame_requires_aligned_series(self) -> None:
        index = pd.Index([10, 11])
        identity = pd.DataFrame(
            {
                "date": ["2019-01-04", "2019-01-04"],
                "numerai_ticker": ["a", "b"],
                "composite_figi": ["fa", "fb"],
            },
            index=index,
        )
        with self.assertRaisesRegex(ValueError, "index does not align"):
            create_v6_oof_frame(
                identity,
                pd.Series([1, 1], index=index),
                pd.Series([0.1, 0.2], index=pd.Index([0, 1])),
                pd.Series([0.25, 0.75], index=index),
            )

    def test_valid_oof_preserves_row_order_and_index(self) -> None:
        oof = self.make_oof()
        validated = validate_v6_oof_predictions(oof, self.folds)
        self.assertTrue(validated.index.equals(oof.index))
        self.assertEqual(validated["numerai_ticker"].tolist(), oof["numerai_ticker"].tolist())

    def test_duplicate_missing_and_nonfinite_predictions_are_rejected(self) -> None:
        duplicate = self.make_oof()
        duplicate.iloc[1, duplicate.columns.get_loc("numerai_ticker")] = duplicate.iloc[0]["numerai_ticker"]
        duplicate.iloc[1, duplicate.columns.get_loc("composite_figi")] = duplicate.iloc[0]["composite_figi"]
        with self.assertRaisesRegex(ValueError, "Duplicate OOF"):
            validate_v6_oof_predictions(duplicate, self.folds)

        for value in (np.nan, np.inf):
            invalid = self.make_oof()
            invalid.iloc[0, invalid.columns.get_loc("prediction")] = value
            with self.assertRaisesRegex(ValueError, "present and finite"):
                validate_v6_oof_predictions(invalid, self.folds)

    def test_null_target_is_rejected(self) -> None:
        oof = self.make_oof()
        oof.iloc[0, oof.columns.get_loc("target_jupiter_60")] = np.nan
        with self.assertRaisesRegex(ValueError, "must not be null"):
            validate_v6_oof_predictions(oof, self.folds)

    def test_holdout_date_and_wrong_fold_are_rejected(self) -> None:
        holdout = self.make_oof()
        holdout.iloc[0, holdout.columns.get_loc("date")] = self.config.holdout_start
        with self.assertRaisesRegex(ValueError, "unexpected"):
            validate_v6_oof_predictions(holdout, self.folds)

        wrong_fold = self.make_oof()
        wrong_fold.iloc[0, wrong_fold.columns.get_loc("fold_id")] = 5
        with self.assertRaisesRegex(ValueError, "fold assignments"):
            validate_v6_oof_predictions(wrong_fold, self.folds)

    def test_numerai_corr_matches_independent_reference_formula(self) -> None:
        prediction = pd.Series([0.1, 0.1, 0.4, 0.8, 0.9], index=range(5))
        target = pd.Series([0.0, 0.25, 0.5, 0.75, 1.0], index=range(5))
        rank = (prediction.rank(method="average") - 0.5) / prediction.count()
        gaussian = stats.norm.ppf(rank)
        transformed_prediction = np.sign(gaussian) * np.abs(gaussian) ** 1.5
        centered = target - target.mean()
        transformed_target = np.sign(centered) * np.abs(centered) ** 1.5
        expected = np.corrcoef(transformed_prediction, transformed_target)[0, 1]

        self.assertAlmostEqual(numerai_corr(prediction, target), expected, places=14)

    def test_legacy_v5_evaluation_functions_remain_compatible(self) -> None:
        predictions = np.tile([0.1, 0.2, 0.5, 0.8, 0.9], 2)
        target = pd.Series(
            [0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 0.75, 0.5, 0.25, 0.0]
        )
        eras = pd.Series(["0001"] * 5 + ["0002"] * 5)

        correlations = calculate_era_correlations(predictions, target, eras)
        summary = summarize_correlations(correlations)

        self.assertEqual(correlations.index.tolist(), ["0001", "0002"])
        self.assertIn("mean_corr", summary)
        self.assertIn("positive_era_rate", summary)

    def test_aggregation_weights_dates_not_folds(self) -> None:
        fold_lengths = [78, 78, 78, 78, 18]
        fold_scores = [0.01, 0.02, 0.03, 0.04, -0.20]
        rows = []
        date_number = 0
        for fold_id, (length, score) in enumerate(
            zip(fold_lengths, fold_scores), start=1
        ):
            for _ in range(length):
                rows.append(
                    {
                        "date": f"{2019 + date_number // 52:04d}-{date_number:03d}",
                        "fold_id": fold_id,
                        "numerai_corr_jupiter_60": score,
                        "observation_count": 10,
                    }
                )
                date_number += 1
        per_date = pd.DataFrame(rows)
        summary = summarize_v6_oof_corr(per_date)
        expected = np.average(fold_scores, weights=fold_lengths)

        self.assertAlmostEqual(summary["overall"]["mean"], expected)
        self.assertNotAlmostEqual(summary["overall"]["mean"], np.mean(fold_scores))
        self.assertEqual([x["date_count"] for x in summary["per_fold"]], fold_lengths)

    def test_metric_availability_does_not_substitute_neutral_scores(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            availability = v6_metric_availability(Path(temp_dir))
        self.assertTrue(availability["NUMERAI_CORR_JUPITER_60"]["available"])
        self.assertTrue(availability["NCORR"]["implemented"])
        self.assertFalse(availability["NCORR"]["available"])
        for metric in ("NMMC", "NBMC"):
            self.assertFalse(availability[metric]["implemented"])
            self.assertFalse(availability[metric]["available"])

    def test_experiment_artifacts_round_trip_and_refuse_overwrite(self) -> None:
        oof = validate_v6_oof_predictions(self.make_oof(), self.folds)
        per_date = calculate_v6_numerai_corr_by_date(oof)
        metrics = summarize_v6_oof_corr(per_date)
        spec = V6ExperimentSpec(
            experiment_id="synthetic_v6_test",
            feature_set_name="medium",
            feature_names=["feature_a", "feature_b"],
            model_family="LightGBM",
            model_parameters=DEFAULT_LIGHTGBM_PARAMS,
            random_seed=42,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            experiment_dir = save_v6_experiment_artifacts(
                spec,
                self.folds,
                oof,
                per_date,
                metrics,
                output_root=Path(temp_dir),
                git_state={"commit": "abc123", "dirty": False},
            )
            loaded = load_v6_experiment_artifacts(experiment_dir)
            metadata_text = json.dumps(loaded["metadata"])

            self.assertEqual(loaded["metadata"]["target"], "target_jupiter_60")
            self.assertEqual(loaded["metadata"]["metric_identity"], "NUMERAI_CORR_JUPITER_60")
            self.assertEqual(len(loaded["oof_predictions"]), len(oof))
            self.assertNotIn(str(Path(temp_dir)), metadata_text)
            with self.assertRaises(FileExistsError):
                save_v6_experiment_artifacts(
                    spec,
                    self.folds,
                    oof,
                    per_date,
                    metrics,
                    output_root=Path(temp_dir),
                )

    def test_experiment_artifacts_record_official_ncorr_identity(self) -> None:
        oof = validate_v6_oof_predictions(self.make_oof(), self.folds)
        per_date = calculate_v6_numerai_corr_by_date(oof)
        per_date["ncorr"] = np.linspace(-0.02, 0.03, len(per_date))
        metrics = combine_v6_metric_summaries(
            summarize_v6_oof_corr(per_date),
            summarize_v6_ncorr(
                per_date[["date", "fold_id", "ncorr", "observation_count"]]
            ),
        )
        spec = V6ExperimentSpec(
            experiment_id="synthetic_ncorr_test",
            feature_set_name="medium",
            feature_names=["feature_a"],
            model_family="LightGBM",
            model_parameters=DEFAULT_LIGHTGBM_PARAMS,
            random_seed=42,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            experiment_dir = save_v6_experiment_artifacts(
                spec,
                self.folds,
                oof,
                per_date,
                metrics,
                output_root=Path(temp_dir),
                git_state={"commit": "abc123", "dirty": False},
            )
            metadata = load_v6_experiment_artifacts(experiment_dir)["metadata"]

        self.assertEqual(metadata["metric_identity"], ["NCORR", "NUMERAI_CORR_JUPITER_60"])
        self.assertEqual(metadata["ncorr_reference"]["metric_version"], 1)
        self.assertEqual(metadata["ncorr_reference"]["numerai_tools_version"], "0.7.2")
        self.assertEqual(metadata["ncorr_reference"]["date_count"], 330)


if __name__ == "__main__":
    unittest.main()
