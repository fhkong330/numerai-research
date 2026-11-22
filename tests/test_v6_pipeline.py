import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.data import build_v6_research_history
from src.experiment_artifacts import load_v6_experiment_artifacts
from src.models import train_validated_lightgbm as official_train_interface
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
            history = build_v6_research_history(historical, research)
            date_by_index = history["date"]
            fitted_dates = []

            def capture_training(model, X_train, y_train, feature_names):
                fitted_dates.append(set(date_by_index.loc[X_train.index]))
                return official_train_interface(
                    model, X_train, y_train, feature_names
                )

            with patch(
                "src.pipeline.train_validated_lightgbm",
                side_effect=capture_training,
            ):
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

            self.assertEqual(len(fitted_dates), 5)
            for trained, fold in zip(fitted_dates, folds):
                self.assertTrue(trained.isdisjoint(fold.validation_dates))
                self.assertEqual(len(fold.purge_dates), 16)
            self.assertEqual(
                oof["date"].map(fold_by_date).tolist(), oof["fold_id"].tolist()
            )
            self.assertEqual(set(oof["date"]), set(fold_by_date))
            self.assertEqual(len(per_date), 330)
            self.assertIn("ncorr", per_date)
            self.assertIn("numerai_corr_jupiter_60", per_date)
            self.assertEqual(len(loaded["metrics"]["ncorr"]["per_fold"]), 5)
            self.assertEqual(metadata["run_kind"], "synthetic_smoke")
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

    def test_incomplete_experiment_is_not_loadable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            incomplete = Path(temp_dir) / "incomplete"
            incomplete.mkdir()
            with self.assertRaisesRegex(FileNotFoundError, "COMPLETED"):
                load_v6_experiment_artifacts(incomplete)


if __name__ == "__main__":
    unittest.main()
