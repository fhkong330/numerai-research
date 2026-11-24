import copy
import unittest

from src.config import DATASET_VERSION, TARGET_COLUMN
from src.v1_policy import load_v1_policy, validate_v1_policy


class FrozenV1PolicyTest(unittest.TestCase):
    def setUp(self):
        self.policy = load_v1_policy()

    def test_exact_components_selection_weights_and_normalization(self):
        self.assertEqual(
            [item["component_id"] for item in self.policy["components"]],
            ["medium_lightgbm", "medium_top250_ic_lightgbm"],
        )
        self.assertEqual(self.policy["components"][0]["feature_count"], 350)
        self.assertEqual(self.policy["components"][1]["selected_feature_count"], 250)
        self.assertEqual(
            self.policy["components"][1]["feature_selection"]["policy_id"],
            "medium_top250_abs_mean_ic_v1",
        )
        self.assertEqual(
            self.policy["ensemble"]["component_weights"],
            {"medium_lightgbm": 0.5, "medium_top250_ic_lightgbm": 0.5},
        )
        self.assertTrue(self.policy["ensemble"]["ridge_excluded"])
        self.assertEqual(
            self.policy["components"][0]["source_experiment_id"],
            "v6_medium_baseline_001",
        )
        self.assertEqual(self.policy["scoring"]["primary"]["metric"], "NCORR")
        self.assertEqual(
            self.policy["scoring"]["diagnostic"]["label"], "Legacy CORR"
        )

    def test_approved_boundaries_and_sealed_state(self):
        holdout = self.policy["final_holdout"]
        self.assertEqual(
            (holdout["gap_start"], holdout["gap_end"], holdout["gap_dates"]),
            ("2025-05-02", "2025-08-15", 16),
        )
        self.assertEqual(
            (holdout["holdout_start"], holdout["holdout_end"], holdout["holdout_dates"]),
            ("2025-08-22", "2026-08-14", 52),
        )
        self.assertEqual(holdout["expected_rows"], 365363)
        self.assertTrue(holdout["targets_sealed"])

    def test_incompatible_configurations_are_rejected(self):
        cases = [
            ("target", "wrong"),
            ("ensemble.component_weights", {"medium_lightgbm": 0.6, "medium_top250_ic_lightgbm": 0.4}),
            ("final_holdout.holdout_start", "2025-08-29"),
            ("components.selection", "wrong_policy"),
        ]
        for field, value in cases:
            invalid = copy.deepcopy(self.policy)
            if field == "target": invalid["target"] = value
            elif field == "ensemble.component_weights": invalid["ensemble"]["component_weights"] = value
            elif field == "final_holdout.holdout_start": invalid["final_holdout"]["holdout_start"] = value
            else: invalid["components"][1]["feature_selection"]["policy_id"] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_v1_policy(invalid)

    def test_historical_v5_defaults_remain_compatible(self):
        self.assertEqual(DATASET_VERSION, "v5.3")
        self.assertEqual(TARGET_COLUMN, "target")


if __name__ == "__main__":
    unittest.main()
