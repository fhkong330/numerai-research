import unittest

import pandas as pd

from src.analyze_ensembles import summarize_candidate


class EnsembleReportingTest(unittest.TestCase):
    def test_candidate_summary_uses_saved_metrics_and_weekly_scores(self):
        loaded = {
            "metrics": {
                "ncorr": {"overall": {"mean": 0.02, "std": 0.01, "sharpe_like": 2.0, "positive_date_percentage": 75.0}},
                "legacy_corr": {"overall": {"mean": 0.03, "std": 0.02, "sharpe_like": 1.5}},
            },
            "per_date_metrics": pd.DataFrame({"ncorr": [-0.02, 0.01, 0.03, 0.06]}),
        }
        result = summarize_candidate("candidate", loaded)
        self.assertEqual(result["candidate_id"], "candidate")
        self.assertEqual(result["worst_weekly_ncorr"], -0.02)
        self.assertAlmostEqual(result["p10_weekly_ncorr"], -0.011)
        self.assertEqual(result["ncorr_mean_std_ratio"], 2.0)

    def test_summary_does_not_require_predictions_or_targets(self):
        loaded = {
            "metrics": {
                "ncorr": {"overall": {"mean": 0.0, "std": 1.0, "sharpe_like": 0.0, "positive_date_percentage": 0.0}},
                "legacy_corr": {"overall": {"mean": 0.0, "std": 1.0, "sharpe_like": 0.0}},
            },
            "per_date_metrics": pd.DataFrame({"ncorr": [0.0]}),
        }
        self.assertEqual(summarize_candidate("x", loaded)["mean_ncorr"], 0.0)


if __name__ == "__main__":
    unittest.main()
