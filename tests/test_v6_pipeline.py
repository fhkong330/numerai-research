import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.experiment_artifacts import load_v6_experiment_artifacts
from src.models import load_streaming_ridge, predict_streaming_ridge
from src.pipeline import (
    _synthetic_v6_smoke_inputs,
    execute_v6_walk_forward,
    main,
)


class V6PipelineIntegrationTest(unittest.TestCase):
    def test_synthetic_end_to_end_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fixture_dir = root / "fixture"
            fixture_dir.mkdir()
            historical, research, features, folds, reference = (
                _synthetic_v6_smoke_inputs(fixture_dir)
            )
            experiment_dir = execute_v6_walk_forward(
                historical,
                research,
                features,
                folds,
                "synthetic_pipeline_test",
                model_params={"n_estimators": 3, "n_jobs": 1, "verbose": -1},
                neutralizer_path=reference,
                expected_neutralizer_count=2,
                output_root=root / "outputs",
                run_kind="synthetic_smoke",
            )

            loaded = load_v6_experiment_artifacts(experiment_dir)
            oof = loaded["oof_predictions"]
            per_date = loaded["per_date_metrics"]
            metadata = loaded["metadata"]
            fold_by_date = {
                date: fold.fold for fold in folds for date in fold.validation_dates
            }

            self.assertTrue(all(len(fold.purge_dates) == 16 for fold in folds))
            self.assertEqual(
                oof["date"].map(fold_by_date).tolist(), oof["fold_id"].tolist()
            )
            self.assertEqual(set(oof["date"]), set(fold_by_date))
            self.assertEqual(len(per_date), 330)
            self.assertIn("ncorr", per_date)
            self.assertIn("numerai_corr_jupiter_60", per_date)
            self.assertEqual(len(loaded["metrics"]["ncorr"]["per_fold"]), 5)
            self.assertEqual(metadata["run_kind"], "synthetic_smoke")
            self.assertNotIn("feature_selection", metadata)
            self.assertNotIn("feature_manifests", loaded)
            self.assertEqual(len(list((experiment_dir / "models").glob("*.txt"))), 5)
            self.assertTrue((experiment_dir / "COMPLETED.json").is_file())

    def test_full_cli_requires_explicit_confirmation(self) -> None:
        with self.assertRaisesRegex(SystemExit, "requires both"):
            main(
                [
                    "--version",
                    "v6.0",
                    "--experiment-id",
                    "must_not_run",
                    "--run",
                ]
            )

    def test_synthetic_ridge_pipeline_saves_reloadable_fold_models(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fixture_dir = root / "fixture"
            fixture_dir.mkdir()
            historical, research, features, folds, reference = (
                _synthetic_v6_smoke_inputs(fixture_dir)
            )
            experiment_dir = execute_v6_walk_forward(
                historical,
                research,
                features,
                folds,
                "synthetic_ridge_test",
                model_params={"batch_size": 256},
                neutralizer_path=reference,
                expected_neutralizer_count=2,
                output_root=root / "outputs",
                run_kind="synthetic_smoke",
                model_config_name="ridge_baseline",
            )
            loaded = load_v6_experiment_artifacts(experiment_dir)
            self.assertEqual(loaded["metadata"]["model_family"], "Ridge")
            self.assertEqual(
                loaded["metadata"]["model_artifact_format"], "json"
            )
            self.assertEqual(
                len(list((experiment_dir / "models").glob("fold_*.json"))), 5
            )
            self.assertEqual(len(loaded["oof_predictions"]), 330 * 8)
            restored = load_streaming_ridge(
                experiment_dir / "models" / "fold_1.json"
            )
            restored_predictions = predict_streaming_ridge(
                restored, research.loc[:, features].head(8)
            )
            self.assertEqual(restored.feature_names, tuple(features))
            self.assertTrue(restored_predictions.notna().all())

    def test_smoke_cli_cannot_call_real_pipeline(self) -> None:
        with patch("src.pipeline.run_v6_smoke_pipeline", return_value=Path("smoke")) as smoke:
            with patch("src.pipeline.run_v6_walk_forward_pipeline") as real:
                main(
                    [
                        "--version",
                        "v6.0",
                        "--experiment-id",
                        "safe_smoke",
                        "--smoke",
                    ]
                )
        smoke.assert_called_once()
        real.assert_not_called()

    def test_ridge_cli_selects_existing_real_pipeline_contract(self) -> None:
        with patch(
            "src.pipeline.run_v6_walk_forward_pipeline", return_value=Path("ridge")
        ) as real:
            main(
                [
                    "--version",
                    "v6.0",
                    "--feature-set",
                    "medium",
                    "--model-config",
                    "ridge_baseline",
                    "--experiment-id",
                    "ridge_future",
                    "--run",
                    "--confirm-full-run",
                ]
            )
        self.assertEqual(real.call_args.kwargs["model_config_name"], "ridge_baseline")
        self.assertIsNone(real.call_args.kwargs["selection_policy_id"])

    def test_incomplete_experiment_is_not_loadable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            incomplete = Path(temp_dir) / "incomplete"
            incomplete.mkdir()
            with self.assertRaisesRegex(FileNotFoundError, "COMPLETED"):
                load_v6_experiment_artifacts(incomplete)


if __name__ == "__main__":
    unittest.main()
