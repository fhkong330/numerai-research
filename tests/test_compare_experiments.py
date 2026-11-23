import unittest

import pandas as pd

from src.compare_experiments import compare_loaded_v6_experiments


def experiment(experiment_id: str, offset: float = 0.0) -> dict:
    dates = ["2019-01-04", "2019-01-11"]
    oof = pd.DataFrame(
        {
            "date": dates,
            "numerai_ticker": ["a", "b"],
            "composite_figi": [pd.NA, "figi_b"],
            "target_jupiter_60": [0.25, 0.75],
        }
    )
    per_date = pd.DataFrame(
        {
            "date": dates,
            "fold_id": [1, 1],
            "ncorr": [0.01 + offset, 0.03 + offset],
            "numerai_corr_jupiter_60": [0.02 + offset, 0.04 + offset],
        }
    )
    return {
        "metadata": {
            "experiment_id": experiment_id,
            "dataset_version": "v6.0",
            "target": "target_jupiter_60",
            "metric_identity": ["NCORR", "NUMERAI_CORR_JUPITER_60"],
            "ncorr_reference": {"metric_version": 1, "sha256": "abc"},
            "feature_set": "small" if offset else "medium",
            "ordered_feature_names": ["feature_a"],
        },
        "metrics": {"ncorr": {"overall": {"mean": 0.02 + offset}}},
        "oof_predictions": oof,
        "per_date_metrics": per_date,
    }


class ExperimentComparisonTest(unittest.TestCase):
    def test_paired_comparison_uses_candidate_minus_baseline(self) -> None:
        result = compare_loaded_v6_experiments(
            experiment("medium"), experiment("small", 0.005)
        )
        self.assertAlmostEqual(result["paired_ncorr"]["mean_difference"], 0.005)
        self.assertEqual(
            result["paired_ncorr"]["candidate_outperformance_percentage"], 100.0
        )
        self.assertEqual(result["paired_ncorr"]["date_count"], 2)

    def test_rejects_misaligned_oof_identity(self) -> None:
        baseline = experiment("medium")
        candidate = experiment("small", 0.005)
        candidate["oof_predictions"].loc[0, "numerai_ticker"] = "different"
        with self.assertRaisesRegex(ValueError, "identities"):
            compare_loaded_v6_experiments(baseline, candidate)

    def test_rejects_scoring_reference_mismatch(self) -> None:
        baseline = experiment("medium")
        candidate = experiment("small", 0.005)
        candidate["metadata"]["ncorr_reference"] = {"metric_version": 2}
        with self.assertRaisesRegex(ValueError, "references"):
            compare_loaded_v6_experiments(baseline, candidate)


if __name__ == "__main__":
    unittest.main()
