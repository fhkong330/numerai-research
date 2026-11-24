import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from importlib.metadata import version

from src.config import get_research_config
from src.evaluation import validate_v6_oof_predictions
from src.paths import OUTPUT_ROOT
from src.scoring_references import V6_VALIDATION_NEUTRALIZER_SHA256
from src.validation import V6WalkForwardFold


@dataclass(frozen=True)
class V6ExperimentSpec:
    experiment_id: str
    feature_set_name: str
    feature_names: list[str]
    model_family: str
    model_parameters: dict
    random_seed: int
    run_kind: str = "research"
    feature_selection_policy_id: str | None = None
    source_feature_quality_analysis_id: str | None = None
    fold_feature_manifests: tuple[dict, ...] | None = None
    model_artifact_extension: str = "txt"


def current_git_state(project_root: Path | None = None) -> dict:
    working_dir = project_root or Path(__file__).resolve().parent.parent
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=working_dir,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=working_dir,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _validate_experiment_id(experiment_id: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", experiment_id):
        raise ValueError(
            "Experiment ID must contain only letters, numbers, '.', '_' or '-'."
        )


def save_v6_experiment_artifacts(
    spec: V6ExperimentSpec,
    folds: list[V6WalkForwardFold],
    oof_predictions: pd.DataFrame,
    per_date_metrics: pd.DataFrame,
    metrics: dict,
    output_root: Path = OUTPUT_ROOT,
    git_state: dict | None = None,
    model_files: dict[int, Path] | None = None,
) -> Path:
    config = get_research_config("v6.0")
    _validate_experiment_id(spec.experiment_id)
    if not spec.feature_names or len(spec.feature_names) != len(
        set(spec.feature_names)
    ):
        raise ValueError("Experiment feature names must be nonempty and unique.")
    validated_oof = validate_v6_oof_predictions(oof_predictions, folds)
    expected_dates = set(validated_oof[config.time_column].astype(str))
    required_metric_columns = {
        config.time_column,
        "fold_id",
        "numerai_corr_jupiter_60",
        "observation_count",
    }
    missing_metric_columns = sorted(
        required_metric_columns - set(per_date_metrics.columns)
    )
    if missing_metric_columns:
        raise ValueError(
            f"Per-date metrics are missing columns: {missing_metric_columns}"
        )
    if per_date_metrics[config.time_column].duplicated().any():
        raise ValueError("Per-date metrics contain duplicate dates.")
    if set(per_date_metrics[config.time_column].astype(str)) != expected_dates:
        raise ValueError("Per-date metrics do not cover the OOF research dates.")
    scores = per_date_metrics["numerai_corr_jupiter_60"]
    if not pd.api.types.is_numeric_dtype(scores) or not np.isfinite(scores).all():
        raise ValueError("Per-date metrics contain invalid scores.")
    has_ncorr = "ncorr" in per_date_metrics.columns
    legacy_summary = metrics.get("legacy_corr", metrics)
    if legacy_summary.get("overall", {}).get("metric") != "NUMERAI_CORR_JUPITER_60":
        raise ValueError("Artifact legacy metric identity must be NUMERAI_CORR_JUPITER_60.")
    if has_ncorr:
        ncorr_scores = per_date_metrics["ncorr"]
        if not pd.api.types.is_numeric_dtype(ncorr_scores) or not np.isfinite(
            ncorr_scores
        ).all():
            raise ValueError("Per-date metrics contain invalid NCORR scores.")
        ncorr_summary = metrics.get("ncorr", {})
        if ncorr_summary.get("overall", {}).get("metric") != "NCORR":
            raise ValueError("Artifact NCORR metric identity must be NCORR.")

    experiment_dir = (
        output_root / config.dataset_version / "experiments" / spec.experiment_id
    )
    if experiment_dir.exists():
        raise FileExistsError(f"Experiment already exists: {experiment_dir}")

    fold_summaries = [fold.summary() for fold in folds]
    feature_manifests = spec.fold_feature_manifests
    if feature_manifests is not None:
        if not spec.feature_selection_policy_id or not spec.source_feature_quality_analysis_id:
            raise ValueError("Fold feature manifests require selection identities.")
        if len(feature_manifests) != len(folds):
            raise ValueError("Expected one feature manifest per fold.")
        for fold, manifest in zip(folds, feature_manifests):
            if manifest.get("fold_id") != fold.fold:
                raise ValueError("Feature manifest fold identity does not match.")
            if manifest.get("policy_id") != spec.feature_selection_policy_id:
                raise ValueError("Feature manifest selection policy does not match.")
            selected = manifest.get("model_feature_names", [])
            if len(selected) != 250 or len(selected) != len(set(selected)):
                raise ValueError("Each feature manifest must select exactly 250 features.")
            if not set(selected) <= set(spec.feature_names):
                raise ValueError("Feature manifest contains features outside the source set.")
            if manifest.get("train_end") != fold.train_dates[-1]:
                raise ValueError("Feature manifest training cutoff does not match its fold.")
    metadata = {
        "experiment_id": spec.experiment_id,
        "dataset_version": config.dataset_version,
        "target": config.target_column,
        "feature_set": spec.feature_set_name,
        "ordered_feature_names": spec.feature_names,
        "model_family": spec.model_family,
        "model_parameters": spec.model_parameters,
        "random_seed": spec.random_seed,
        "run_kind": spec.run_kind,
        "model_artifact_format": spec.model_artifact_extension,
        "validation_method": "expanding_walk_forward",
        "purge_dates": config.purge_dates,
        "folds": fold_summaries,
        "metric_identity": (
            ["NCORR", "NUMERAI_CORR_JUPITER_60"]
            if has_ncorr
            else "NUMERAI_CORR_JUPITER_60"
        ),
        "metric_note": (
            "NCORR is official neutral_corr v1; NUMERAI_CORR_JUPITER_60 is "
            "a separate legacy research diagnostic."
            if has_ncorr
            else "Official numerai_corr formula used as a research diagnostic; "
            "this is not NCORR, NMMC, or NBMC."
        ),
        "git": git_state if git_state is not None else current_git_state(),
        "artifacts": {
            "oof_predictions": "oof_predictions.parquet",
            "per_date_metrics": "per_date_metrics.parquet",
            "metrics": "metrics.json",
        },
    }
    if feature_manifests is not None:
        metadata["feature_selection"] = {
            "policy_id": spec.feature_selection_policy_id,
            "source_feature_quality_analysis_id": spec.source_feature_quality_analysis_id,
            "selected_feature_count_per_fold": 250,
            "source_feature_count": len(spec.feature_names),
        }
    if has_ncorr:
        metadata["ncorr_reference"] = {
            "api_display_name": "neutral_corr",
            "metric_version": 1,
            "numerai_tools_version": version("numerai-tools"),
            "neutralizer_dataset_version": "v6.0",
            "neutralizer_sha256": V6_VALIDATION_NEUTRALIZER_SHA256,
            "date_start": str(per_date_metrics[config.time_column].min()),
            "date_end": str(per_date_metrics[config.time_column].max()),
            "date_count": int(len(per_date_metrics)),
            "dates": sorted(per_date_metrics[config.time_column].astype(str).tolist()),
        }

    experiment_dir.mkdir(parents=True, exist_ok=False)
    try:
        validated_oof.to_parquet(
            experiment_dir / "oof_predictions.parquet",
            index=True,
        )
        per_date_metrics.to_parquet(
            experiment_dir / "per_date_metrics.parquet",
            index=False,
        )
        if model_files:
            models_dir = experiment_dir / "models"
            models_dir.mkdir()
            for fold_id, source in sorted(model_files.items()):
                if not source.is_file():
                    raise FileNotFoundError(f"Fold model file not found: {source}")
                if source.suffix.lstrip(".") != spec.model_artifact_extension:
                    raise ValueError("Fold model artifact extension does not match metadata.")
                shutil.copy2(
                    source,
                    models_dir / f"fold_{fold_id}.{spec.model_artifact_extension}",
                )
            metadata["artifacts"]["models"] = "models/"
        if feature_manifests is not None:
            manifests_dir = experiment_dir / "feature_manifests"
            manifests_dir.mkdir()
            for manifest in feature_manifests:
                path = manifests_dir / f"fold_{manifest['fold_id']}.json"
                manifest_payload = {
                    **manifest,
                    "model_parameters": spec.model_parameters,
                    "random_seed": spec.random_seed,
                    "git": metadata["git"],
                }
                with path.open("x", encoding="utf-8") as file:
                    json.dump(manifest_payload, file, indent=2, allow_nan=False)
            metadata["artifacts"]["feature_manifests"] = "feature_manifests/"
        with (experiment_dir / "metadata.json").open("x", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2, allow_nan=False)
        with (experiment_dir / "metrics.json").open("x", encoding="utf-8") as file:
            json.dump(metrics, file, indent=2, allow_nan=False)
        with (experiment_dir / "COMPLETED.json").open("x", encoding="utf-8") as file:
            json.dump({"status": "complete"}, file, indent=2)
    except Exception:
        # Keep the incomplete directory for diagnosis, but loaders will reject it.
        raise
    return experiment_dir


def load_v6_experiment_artifacts(experiment_dir: Path) -> dict:
    required = (
        "COMPLETED.json",
        "metadata.json",
        "metrics.json",
        "oof_predictions.parquet",
        "per_date_metrics.parquet",
    )
    missing = [name for name in required if not (experiment_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Experiment artifacts are missing: {missing}")
    with (experiment_dir / "metadata.json").open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    with (experiment_dir / "metrics.json").open("r", encoding="utf-8") as file:
        metrics = json.load(file)
    result = {
        "metadata": metadata,
        "metrics": metrics,
        "oof_predictions": pd.read_parquet(
            experiment_dir / "oof_predictions.parquet"
        ),
        "per_date_metrics": pd.read_parquet(
            experiment_dir / "per_date_metrics.parquet"
        ),
    }
    manifest_directory = metadata.get("artifacts", {}).get("feature_manifests")
    if manifest_directory:
        manifest_root = experiment_dir / manifest_directory
        paths = [manifest_root / f"fold_{fold_id}.json" for fold_id in range(1, 6)]
        missing_manifests = [path.name for path in paths if not path.is_file()]
        if missing_manifests:
            raise FileNotFoundError(
                f"Experiment feature manifests are missing: {missing_manifests}"
            )
        manifests = []
        for path in paths:
            with path.open("r", encoding="utf-8") as file:
                manifests.append(json.load(file))
        result["feature_manifests"] = manifests
    return result
