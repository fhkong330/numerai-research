import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.production_training import (
    _fit_component,
    choose_ic_plan,
    select_production_top250,
    snapshot_compatibility,
    verify_production_artifacts,
    write_json_once,
    append_progress,
)


class ProductionTrainingTest(unittest.TestCase):
    def test_snapshot_revision_controls_ic_reuse(self) -> None:
        files = {
            "features.json": {"sha256": "wrong"},
            "train.parquet": {"sha256": "wrong"},
            "validation.parquet": {"sha256": "wrong"},
        }
        self.assertEqual(choose_ic_plan(files), "full_recompute")

    def test_top250_is_deterministic_and_restores_official_order(self) -> None:
        features = [f"feature_{index:03d}" for index in range(350)]
        summary = pd.DataFrame(
            {
                "feature": features,
                "train_start": "2020-01-03",
                "train_end": "2026-07-24",
                "training_dates": 300,
                "valid_dates": 300,
                "mean_ic": np.linspace(-0.35, 0.349, 350),
            }
        )
        summary["absolute_mean_ic"] = summary["mean_ic"].abs()
        first = select_production_top250(summary, features)
        second = select_production_top250(summary, features)
        self.assertEqual(first, second)
        self.assertEqual(len(first["model_feature_names"]), 250)
        self.assertEqual(
            first["model_feature_names"],
            [name for name in features if name in set(first["model_feature_names"])],
        )

    def test_model_persistence_and_reload(self) -> None:
        rng = np.random.default_rng(42)
        features = ["feature_a", "feature_b"]
        frame = pd.DataFrame(rng.normal(size=(100, 2)).astype(np.float32), columns=features)
        target = pd.Series(rng.normal(size=100).astype(np.float32))
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "model.txt"
            diagnostics = _fit_component(frame, target, features, path)
            self.assertEqual(diagnostics["trees"], 300)
            self.assertEqual(diagnostics["reload_max_absolute_prediction_difference"], 0.0)

    def test_json_artifact_is_non_overwriting(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "record.json"
            write_json_once(path, {"state": "one"})
            with self.assertRaises(FileExistsError):
                write_json_once(path, {"state": "two"})

    def test_progress_log_is_durable_json_lines(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "progress.jsonl"
            append_progress(path, "first", count=1)
            append_progress(path, "second", count=2)
            events = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([event["stage"] for event in events], ["first", "second"])
            self.assertEqual(events[-1]["count"], 2)

    def test_incomplete_artifact_directory_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "incomplete"):
                verify_production_artifacts(Path(temp_dir))


if __name__ == "__main__":
    unittest.main()
