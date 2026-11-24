import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.data import (
    V6HoldoutFeatures,
    V6TrainingData,
    prepare_v6_final_training_data,
    prepare_v6_holdout_features,
)
from src.feature_quality import IC_COLUMNS, build_final_training_ic_summary
from src.feature_selection import select_final_top250
from src.final_holdout import (
    build_fixed_holdout_ensemble,
    main,
    require_scoring_authorization,
    run_final_preparation,
    save_final_preparation,
    save_final_scoring,
    verify_final_preparation,
)


FEATURES = [f"feature_{index:03d}" for index in range(350)]
HOLDOUT_DATES = pd.date_range("2025-08-22", periods=52, freq="7D").strftime("%Y-%m-%d").tolist()


def holdout_frame(reverse=False):
    rows = []
    for date in HOLDOUT_DATES:
        values = [3.0, 2.0, 1.0] if reverse else [1.0, 2.0, 3.0]
        for index, value in enumerate(values):
            rows.append((date, f"ticker_{index}", f"figi_{index}", value))
    return pd.DataFrame(rows, columns=["date", "numerai_ticker", "composite_figi", "prediction"])


class FinalHoldoutTest(unittest.TestCase):
    def test_final_training_cutoff_gap_and_maturity(self):
        historical = pd.DataFrame({"date": ["2018-12-28"], "target_jupiter_60": [0.5], "f": pd.Series([1], dtype="int8")})
        research = pd.DataFrame({"date": ["2025-04-25"], "target_jupiter_60": [0.75], "f": pd.Series([2], dtype="int8")})
        result = prepare_v6_final_training_data(historical, research, ["f"])
        self.assertEqual(result.features["f"].dtype, np.dtype("int8"))
        protected = research.copy(); protected["date"] = "2025-05-02"
        with self.assertRaisesRegex(ValueError, "cutoff"):
            prepare_v6_final_training_data(historical, protected, ["f"])
        null = research.copy(); null["target_jupiter_60"] = np.nan
        with self.assertRaisesRegex(ValueError, "null"):
            prepare_v6_final_training_data(historical, null, ["f"])

    def test_target_free_holdout_and_fixed_rank_ensemble(self):
        medium = holdout_frame(False)
        top250 = holdout_frame(True)
        feature_frame = medium.drop(columns="prediction").assign(f=1)
        prepared = prepare_v6_holdout_features(feature_frame, ["f"])
        self.assertNotIn("target_jupiter_60", prepared.features.columns)
        with self.assertRaisesRegex(ValueError, "forbidden"):
            prepare_v6_holdout_features(feature_frame.assign(target_jupiter_60=0.5), ["f"])
        left, right, ensemble = build_fixed_holdout_ensemble(medium, top250)
        np.testing.assert_allclose(ensemble["prediction"], 0.5)
        self.assertEqual(ensemble["date"].nunique(), 52)

    def test_final_ic_reuse_coverage_and_deterministic_selection(self):
        dates = ["2025-04-18", "2025-04-25"]
        rows = []
        for date in dates:
            for index, feature in enumerate(FEATURES):
                rows.append((date, feature, (index - 175) / 1000, 10, 10, 1.0, "valid"))
        ic = pd.DataFrame(rows, columns=IC_COLUMNS)
        metadata = {
            "dataset_version": "v6.0", "target": "target_jupiter_60",
            "ordered_feature_names": FEATURES,
            "ic_definition": "Per-date cross-sectional Spearman correlation: Pearson correlation of pairwise-complete average ranks.",
        }
        summary = build_final_training_ic_summary(ic.iloc[:350], ic.iloc[350:], FEATURES, dates, metadata)
        manifest_a = select_final_top250(summary, FEATURES)
        manifest_b = select_final_top250(summary.sample(frac=1, random_state=1).set_index("feature").loc[FEATURES].reset_index(), FEATURES)
        self.assertEqual(manifest_a["model_feature_names"], manifest_b["model_feature_names"])
        self.assertEqual(len(manifest_a["model_feature_names"]), 250)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            build_final_training_ic_summary(ic.iloc[:350], ic.iloc[350:-1], FEATURES, dates, metadata)

    def test_preparation_hash_authorization_state_and_overwrite(self):
        medium = holdout_frame(False); top250 = holdout_frame(True)
        left, right, ensemble = build_fixed_holdout_ensemble(medium, top250)
        manifest = {"policy_id": "medium_top250_abs_mean_ic_v1", "selected_feature_count": 250}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); model_a = root / "a.txt"; model_b = root / "b.txt"
            model_a.write_text("fixed-a"); model_b.write_text("fixed-b")
            evaluation = save_final_preparation(
                "synthetic_final", manifest, {"medium": model_a, "top250": model_b},
                left, right, ensemble, training_dates=["2003-01-03", "2025-04-25"],
                output_root=root, git_state={"commit": "x", "dirty": True, "synthetic": True},
                _synthetic_expected_rows=len(ensemble),
            )
            metadata = verify_final_preparation(evaluation)
            with self.assertRaises(PermissionError):
                require_scoring_authorization(evaluation, False)
            authorized = require_scoring_authorization(evaluation, True)
            metrics = pd.DataFrame({"date": HOLDOUT_DATES, "ncorr": np.zeros(52)})
            save_final_scoring(evaluation, authorized, metrics, {"ncorr": {}}, {"components": 2})
            with self.assertRaises(FileExistsError):
                require_scoring_authorization(evaluation, True)
            with self.assertRaises(FileExistsError):
                save_final_preparation(
                    "synthetic_final", manifest, {"medium": model_a, "top250": model_b},
                    left, right, ensemble, training_dates=["2025-04-25"], output_root=root,
                    git_state={"synthetic": True}, _synthetic_expected_rows=len(ensemble),
                )
            self.assertEqual(metadata["target_columns_accessed"], [])

    def test_altered_prediction_hash_is_rejected(self):
        medium = holdout_frame(False); top250 = holdout_frame(True)
        left, right, ensemble = build_fixed_holdout_ensemble(medium, top250)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); a=root/"a.txt"; b=root/"b.txt"; a.write_text("a"); b.write_text("b")
            evaluation = save_final_preparation(
                "tamper", {"policy_id":"medium_top250_abs_mean_ic_v1","selected_feature_count":250},
                {"medium":a,"top250":b}, left,right,ensemble,
                training_dates=["2025-04-25"], output_root=root,
                git_state={"synthetic":True}, _synthetic_expected_rows=len(ensemble),
            )
            path=evaluation/"preparation"/"ensemble_predictions.parquet"
            changed=pd.read_parquet(path); changed.loc[0,"prediction"]=0.1; changed.to_parquet(path,index=False)
            with self.assertRaisesRegex(ValueError,"hash mismatch"):
                verify_final_preparation(evaluation)

    def test_cli_confirmations_are_distinct(self):
        with self.assertRaisesRegex(SystemExit, "preparation requires"):
            main(["prepare","--version","v6.0","--evaluation-id","x"])
        with self.assertRaisesRegex(SystemExit, "scoring requires"):
            main(["score","--version","v6.0","--evaluation-id","x"])
        with self.assertRaisesRegex(SystemExit, "scoring remains intentionally disabled"):
            main([
                "score", "--version", "v6.0", "--evaluation-id", "x", "--run",
                "--authorize-holdout-target-access", "--confirm-final-scoring",
            ])

    def test_production_preparation_orchestrates_two_target_free_models(self):
        rng = np.random.default_rng(42)
        train_features = pd.DataFrame(
            rng.normal(size=(40, 350)).astype(np.float32), columns=FEATURES
        )
        target = pd.Series(rng.normal(size=40), index=train_features.index)
        identities = pd.DataFrame(
            [
                (date, f"ticker_{row}", f"figi_{row}")
                for date in HOLDOUT_DATES
                for row in range(2)
            ],
            columns=["date", "numerai_ticker", "composite_figi"],
        )
        holdout_features = pd.DataFrame(
            rng.normal(size=(len(identities), 350)).astype(np.float32), columns=FEATURES
        )
        summary = pd.DataFrame(
            {
                "feature": FEATURES,
                "train_start": "2003-01-03",
                "train_end": "2025-04-25",
                "training_dates": 1165,
                "valid_dates": 1165,
                "mean_ic": np.linspace(-0.02, 0.02, 350),
            }
        )
        summary["absolute_mean_ic"] = summary["mean_ic"].abs()
        provenance = {
            "existing_dates": 1131,
            "additional_dates": 34,
            "combined_dates": 1165,
            "holdout_target_columns_accessed": [],
        }
        captured = {}

        def fake_save(*args, **kwargs):
            captured["model_keys"] = set(args[2])
            captured["medium"] = args[3]
            captured["top250"] = args[4]
            captured["ensemble"] = args[5]
            captured["details"] = kwargs["preparation_details"]
            destination = Path(kwargs["output_root"]) / "v6.0" / "final_holdout" / args[0]
            preparation = destination / "preparation"
            preparation.mkdir(parents=True)
            (preparation / "metadata.json").write_text(
                '{"preparation_details": {}}', encoding="utf-8"
            )
            return destination

        with tempfile.TemporaryDirectory() as temp, patch(
            "src.final_holdout.resolve_v6_numeric_feature_set", return_value=FEATURES
        ), patch(
            "src.final_holdout.extend_final_training_ic",
            return_value=(pd.DataFrame(), pd.DataFrame(), summary, provenance),
        ), patch(
            "src.final_holdout.load_v6_final_training_data",
            return_value=V6TrainingData(train_features, target),
        ), patch(
            "src.final_holdout.load_v6_final_holdout_features",
            return_value=V6HoldoutFeatures(holdout_features, identities),
        ), patch(
            "src.final_holdout.verify_observed_final_training_dates",
            return_value=(
                pd.date_range("2003-01-03", periods=1165, freq="7D")
                .strftime("%Y-%m-%d")
                .tolist(),
                len(train_features),
            ),
        ), patch("src.final_holdout.save_final_preparation", side_effect=fake_save):
            destination = run_final_preparation(
                "synthetic_orchestration", data_dir=Path(temp), output_root=Path(temp)
            )
        self.assertTrue(destination.name == "synthetic_orchestration")
        self.assertEqual(captured["model_keys"], {"medium", "top250"})
        self.assertEqual(len(captured["medium"]), len(identities))
        self.assertEqual(len(captured["top250"]), len(identities))
        self.assertEqual(len(captured["ensemble"]), len(identities))
        self.assertNotIn("target_jupiter_60", captured["medium"].columns)
        self.assertEqual(captured["details"]["ic"]["additional_dates"], 34)


if __name__ == "__main__":
    unittest.main()
