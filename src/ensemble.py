"""Fixed, target-free rank ensembles for completed v6 OOF experiments."""

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import get_research_config
from src.evaluation import (
    calculate_v6_ncorr_by_date,
    calculate_v6_numerai_corr_by_date,
    combine_v6_metric_summaries,
    summarize_v6_ncorr,
    summarize_v6_oof_corr,
)
from src.experiment_artifacts import current_git_state, load_v6_experiment_artifacts
from src.paths import OUTPUT_ROOT


IDENTITY_COLUMNS = ["date", "numerai_ticker", "composite_figi", "fold_id"]


@dataclass(frozen=True)
class EnsemblePolicy:
    policy_id: str
    components: tuple[tuple[str, float], ...]

    @property
    def weights(self) -> dict[str, float]:
        return dict(self.components)


APPROVED_ENSEMBLE_POLICIES = {
    "medium_top250_rank_equal_v1": EnsemblePolicy(
        "medium_top250_rank_equal_v1",
        (("v6_medium_baseline_001", 0.50), ("v6_medium_top250_ic_002", 0.50)),
    ),
    "medium_top250_ridge_rank_442_v1": EnsemblePolicy(
        "medium_top250_ridge_rank_442_v1",
        (
            ("v6_medium_baseline_001", 0.40),
            ("v6_medium_top250_ic_002", 0.40),
            ("v6_medium_ridge_003", 0.20),
        ),
    ),
}


def validate_weights(weights: dict[str, float]) -> None:
    if not weights or len(weights) != len(set(weights)):
        raise ValueError("Ensemble weights must identify unique components.")
    values = np.asarray(list(weights.values()), dtype=np.float64)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Ensemble weights must be finite and positive.")
    if not math.isclose(float(values.sum()), 1.0, abs_tol=1e-12):
        raise ValueError("Ensemble weights must sum to 1.0.")


def rank_normalize_by_date(frame: pd.DataFrame, prediction_column: str) -> pd.Series:
    """Return (average rank - 0.5) / n independently within each date."""
    required = {"date", prediction_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Rank normalization is missing columns: {missing}")
    values = frame[prediction_column]
    if not pd.api.types.is_numeric_dtype(values):
        raise ValueError("Predictions must be numeric.")
    numeric = values.to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("Predictions must be present and finite.")
    if frame["date"].isna().any():
        raise ValueError("Dates must be present for rank normalization.")
    counts = frame.groupby("date", sort=False)[prediction_column].transform("size")
    ranks = frame.groupby("date", sort=False)[prediction_column].rank(method="average")
    return ((ranks - 0.5) / counts).astype(np.float64)


def _canonical_component(artifact: dict, experiment_id: str) -> pd.DataFrame:
    frame = artifact["oof_predictions"]
    missing = sorted({*IDENTITY_COLUMNS, "prediction"} - set(frame.columns))
    if missing:
        raise ValueError(f"{experiment_id} OOF is missing columns: {missing}")
    selected = frame.loc[:, [*IDENTITY_COLUMNS, "prediction"]].copy()
    if selected[IDENTITY_COLUMNS].isna().any(axis=None):
        # composite_figi is legitimately nullable in v6; all other identities are not.
        if selected[["date", "numerai_ticker", "fold_id"]].isna().any(axis=None):
            raise ValueError(f"{experiment_id} contains missing required identities.")
    if selected.duplicated(IDENTITY_COLUMNS).any():
        raise ValueError(f"{experiment_id} contains duplicate OOF identities.")
    if not pd.api.types.is_numeric_dtype(selected["prediction"]):
        raise ValueError(f"{experiment_id} predictions must be numeric.")
    if not np.isfinite(selected["prediction"].to_numpy(dtype=np.float64)).all():
        raise ValueError(f"{experiment_id} predictions must be present and finite.")
    return selected.sort_values(IDENTITY_COLUMNS, na_position="first").reset_index(drop=True)


def _validate_metadata(artifacts: dict[str, dict]) -> dict:
    first_id = next(iter(artifacts))
    baseline = artifacts[first_id]["metadata"]
    required = ("dataset_version", "target", "folds", "metric_identity", "ncorr_reference")
    for experiment_id, artifact in artifacts.items():
        metadata = artifact["metadata"]
        if metadata.get("experiment_id") != experiment_id:
            raise ValueError(f"Source experiment identity mismatch: {experiment_id}")
        for field in required:
            if metadata.get(field) != baseline.get(field):
                raise ValueError(f"Source experiment metadata mismatch: {field}")
    if baseline.get("dataset_version") != "v6.0" or baseline.get("target") != "target_jupiter_60":
        raise ValueError("Ensembles require the approved v6.0 Jupiter-60 research identity.")
    return baseline


def _validate_approved_dates(frame: pd.DataFrame, folds: list[dict]) -> None:
    config = get_research_config("v6.0")
    if frame["date"].nunique() != config.expected_research_dates:
        raise ValueError("Ensemble dates do not contain all approved Research Validation dates.")
    dates = frame["date"].astype(str)
    if dates.min() != config.research_start or dates.max() != config.research_end:
        raise ValueError("Ensemble contains dates outside approved Research Validation.")
    expected_fold = {}
    for fold in folds:
        fold_id = int(fold["fold"])
        selected = sorted(d for d in dates.unique() if fold["validation_start"] <= d <= fold["validation_end"])
        if len(selected) != int(fold["validation_dates"]):
            raise ValueError("Ensemble dates do not match approved fold boundaries.")
        expected_fold.update({date: fold_id for date in selected})
    actual = frame["fold_id"].to_numpy(dtype=np.int64)
    mapped = dates.map(expected_fold)
    if mapped.isna().any() or not np.array_equal(actual, mapped.to_numpy(dtype=np.int64)):
        raise ValueError("Ensemble fold assignments do not match approved dates.")


def construct_rank_ensemble(
    artifacts: dict[str, dict], weights: dict[str, float]
) -> tuple[pd.DataFrame, dict]:
    """Construct a target-independent rank ensemble from completed artifacts."""
    validate_weights(weights)
    if tuple(artifacts) != tuple(weights):
        raise ValueError("Artifact order and identities must exactly match ensemble weights.")
    metadata = _validate_metadata(artifacts)
    components = {key: _canonical_component(value, key) for key, value in artifacts.items()}
    first_id = next(iter(components))
    identity = components[first_id][IDENTITY_COLUMNS]
    result = identity.copy()
    diagnostics = {}
    weighted = np.zeros(len(identity), dtype=np.float64)
    for experiment_id, weight in weights.items():
        component = components[experiment_id]
        if not component[IDENTITY_COLUMNS].equals(identity):
            raise ValueError(f"Source OOF identities do not match: {experiment_id}")
        normalized = rank_normalize_by_date(component, "prediction")
        weighted += weight * normalized.to_numpy()
        diagnostics[experiment_id] = {
            "weight": weight,
            "constant_date_count": int(
                component.groupby("date")["prediction"].nunique(dropna=False).eq(1).sum()
            ),
        }
    result["prediction"] = weighted
    _validate_approved_dates(result, metadata["folds"])
    return result, diagnostics


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_ensemble_artifacts(
    ensemble_id: str,
    policy: EnsemblePolicy,
    predictions: pd.DataFrame,
    source_root: Path,
    output_root: Path = OUTPUT_ROOT,
    per_date_metrics: pd.DataFrame | None = None,
    metrics: dict | None = None,
    git_state: dict | None = None,
) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", ensemble_id):
        raise ValueError("Invalid ensemble ID.")
    destination = output_root / "v6.0" / "ensembles" / ensemble_id
    if destination.exists():
        raise FileExistsError(f"Ensemble artifacts already exist: {destination}")
    required = {*IDENTITY_COLUMNS, "prediction"}
    if sorted(required - set(predictions.columns)):
        raise ValueError("Ensemble predictions are incomplete.")
    if predictions.duplicated(IDENTITY_COLUMNS).any():
        raise ValueError("Ensemble predictions contain duplicate identities.")
    if not np.isfinite(predictions["prediction"].to_numpy(dtype=np.float64)).all():
        raise ValueError("Ensemble predictions must be finite.")
    if (per_date_metrics is None) != (metrics is None):
        raise ValueError("Scored ensembles require both per-date and aggregate metrics.")
    source_artifacts = {}
    reference_metadata = None
    for experiment_id, weight in policy.components:
        experiment_dir = source_root / experiment_id
        with (experiment_dir / "metadata.json").open(encoding="utf-8") as file:
            component_metadata = json.load(file)
        if reference_metadata is None:
            reference_metadata = component_metadata
        source_artifacts[experiment_id] = {
            "weight": weight,
            "metadata_sha256": _sha256(experiment_dir / "metadata.json"),
            "oof_predictions_sha256": _sha256(experiment_dir / "oof_predictions.parquet"),
        }
    metadata = {
        "ensemble_id": ensemble_id,
        "artifact_type": "ensemble",
        "dataset_version": "v6.0",
        "target": "target_jupiter_60",
        "experiment_id": ensemble_id,
        "feature_set": "rank_ensemble",
        "ordered_feature_names": [],
        "folds": reference_metadata.get("folds"),
        "metric_identity": reference_metadata.get("metric_identity"),
        "ncorr_reference": reference_metadata.get("ncorr_reference"),
        "policy_id": policy.policy_id,
        "component_experiment_ids": [name for name, _ in policy.components],
        "component_weights": policy.weights,
        "normalization": "per-date average rank normalized as (rank - 0.5) / n",
        "identity_columns": IDENTITY_COLUMNS,
        "target_columns_accessed_for_construction": [],
        "date_start": str(predictions["date"].min()),
        "date_end": str(predictions["date"].max()),
        "date_count": int(predictions["date"].nunique()),
        "row_count": int(len(predictions)),
        "contains_trained_models": False,
        "scored": metrics is not None,
        "source_artifacts": source_artifacts,
        "git": git_state or current_git_state(),
        "artifacts": {"predictions": "ensemble_predictions.parquet"},
    }
    destination.mkdir(parents=True, exist_ok=False)
    predictions.to_parquet(destination / "ensemble_predictions.parquet", index=False)
    if metrics is not None:
        per_date_metrics.to_parquet(destination / "per_date_metrics.parquet", index=False)
        with (destination / "metrics.json").open("x", encoding="utf-8") as file:
            json.dump(metrics, file, indent=2, allow_nan=False)
        metadata["artifacts"].update(
            {"per_date_metrics": "per_date_metrics.parquet", "metrics": "metrics.json"}
        )
    with (destination / "metadata.json").open("x", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, allow_nan=False)
    with (destination / "COMPLETED.json").open("x", encoding="utf-8") as file:
        json.dump({"status": "complete"}, file, indent=2)
    return destination


def load_ensemble_artifacts(path: Path, require_scored: bool = False) -> dict:
    required = ("COMPLETED.json", "metadata.json", "ensemble_predictions.parquet")
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Ensemble artifacts are missing: {missing}")
    with (path / "metadata.json").open(encoding="utf-8") as file:
        metadata = json.load(file)
    result = {
        "metadata": metadata,
        "predictions": pd.read_parquet(path / "ensemble_predictions.parquet"),
    }
    if metadata.get("scored"):
        with (path / "metrics.json").open(encoding="utf-8") as file:
            result["metrics"] = json.load(file)
        result["per_date_metrics"] = pd.read_parquet(path / "per_date_metrics.parquet")
    elif require_scored:
        raise ValueError("Ensemble has not been scored.")
    return result


def score_ensemble_predictions(predictions: pd.DataFrame, target: pd.Series) -> tuple[pd.DataFrame, dict]:
    """Use the existing approved scoring functions; targets enter only here."""
    if not target.index.equals(predictions.index):
        raise ValueError("Target index does not align with ensemble predictions.")
    scored = predictions.copy()
    scored["target_jupiter_60"] = target
    legacy = calculate_v6_numerai_corr_by_date(scored)
    ncorr = calculate_v6_ncorr_by_date(scored)
    per_date = legacy.merge(ncorr, on=["date", "fold_id", "observation_count"], validate="one_to_one")
    metrics = combine_v6_metric_summaries(
        summarize_v6_oof_corr(per_date), summarize_v6_ncorr(per_date)
    )
    return per_date, metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--policy", required=True, choices=sorted(APPROVED_ENSEMBLE_POLICIES))
    parser.add_argument("--ensemble-id", required=True)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--confirm-real-scoring", action="store_true")
    args = parser.parse_args(argv)
    if args.version != "v6.0":
        raise SystemExit("Fixed rank ensembles currently support only v6.0.")
    if not args.run or not args.confirm_real_scoring:
        raise SystemExit("Real ensemble scoring requires --run --confirm-real-scoring.")
    policy = APPROVED_ENSEMBLE_POLICIES[args.policy]
    experiment_root = OUTPUT_ROOT / "v6.0" / "experiments"
    artifacts = {
        experiment_id: load_v6_experiment_artifacts(experiment_root / experiment_id)
        for experiment_id, _ in policy.components
    }
    predictions, _ = construct_rank_ensemble(artifacts, policy.weights)
    # Target access is deliberately isolated from target-free construction.
    first = artifacts[policy.components[0][0]]["oof_predictions"]
    canonical = first.sort_values(IDENTITY_COLUMNS, na_position="first").reset_index(drop=True)
    per_date, metrics = score_ensemble_predictions(predictions, canonical["target_jupiter_60"])
    path = save_ensemble_artifacts(
        args.ensemble_id, policy, predictions, experiment_root,
        per_date_metrics=per_date, metrics=metrics,
    )
    print(path)


if __name__ == "__main__":
    main()
