"""Sealed two-phase final-holdout artifacts for the frozen v6 V1 policy."""

import argparse
import gc
import hashlib
import json
import re
import shutil
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import psutil
import pyarrow.parquet as pq

from src.config import DEFAULT_LIGHTGBM_PARAMS, get_research_config
from src.data import (
    load_v6_final_holdout_features,
    load_v6_final_training_data,
    resolve_v6_numeric_feature_set,
)
from src.ensemble import rank_normalize_by_date
from src.experiment_artifacts import current_git_state
from src.evaluation import (
    calculate_v6_ncorr_by_date,
    calculate_v6_numerai_corr_by_date,
    combine_v6_metric_summaries,
    summarize_v6_ncorr,
    summarize_v6_oof_corr,
)
from src.feature_quality import (
    build_final_training_ic_summary,
    calculate_cross_sectional_spearman_ic,
    load_feature_quality_artifacts,
)
from src.feature_selection import select_final_top250
from src.models import (
    create_lightgbm_model,
    predict_validated_lightgbm,
    train_validated_lightgbm,
)
from src.paths import OUTPUT_ROOT
from src.paths import resolve_data_dir
from src.scoring_references import (
    V6_NEUTRALIZER_COLUMN_COUNT,
    V6_VALIDATION_NEUTRALIZER_SHA256,
    load_final_holdout_neutralizers_for_date,
    ordered_neutralizer_columns,
    validation_neutralizer_path,
)
from src.v1_policy import V1_POLICY_PATH, load_v1_policy


IDENTITY_COLUMNS = ["date", "numerai_ticker", "composite_figi"]
PREPARATION_COMPLETE = "preparation_complete"
SCORING_COMPLETE = "scoring_complete"
SOURCE_IC_ANALYSIS_ID = "v6_medium_ic_001"
EXPECTED_EXISTING_IC_DATES = 1_131
EXPECTED_ADDITIONAL_IC_DATES = 34
EXPECTED_FINAL_IC_DATES = 1_165
PREPARATION_SEAL_PATH = (
    Path(__file__).resolve().parent.parent
    / "docs"
    / "manifests"
    / "v6.0_v1_final_001_preparation_seal.json"
)
PREPARATION_SEAL_SHA256 = (
    "dbb1d968687125abd915be88e9b864c0c005e324e14f053f92d470c854b9a297"
)
RESEARCH_ENSEMBLE_ID = "v6_medium_top250_rank_equal_001"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_holdout_predictions(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    required = {*IDENTITY_COLUMNS, "prediction"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{name} predictions are missing columns: {missing}")
    selected = frame.loc[:, [*IDENTITY_COLUMNS, "prediction"]].copy()
    if selected[["date", "numerai_ticker"]].isna().any(axis=None):
        raise ValueError(f"{name} contains missing required identities.")
    if selected.duplicated(IDENTITY_COLUMNS).any():
        raise ValueError(f"{name} contains duplicate identities.")
    if not np.isfinite(selected["prediction"].to_numpy(dtype=np.float64)).all():
        raise ValueError(f"{name} predictions must be finite.")
    config = get_research_config("v6.0")
    dates = selected["date"].astype(str)
    if dates.nunique() != config.expected_holdout_dates or dates.min() != config.holdout_start or dates.max() != config.holdout_end:
        raise ValueError(f"{name} dates do not match the frozen holdout.")
    return selected.sort_values(IDENTITY_COLUMNS, na_position="first").reset_index(drop=True)


def build_fixed_holdout_ensemble(
    medium: pd.DataFrame, top250: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Canonicalize two components and apply the frozen 50/50 rank policy."""
    left = canonical_holdout_predictions(medium, "Medium")
    right = canonical_holdout_predictions(top250, "Top-250")
    if not left[IDENTITY_COLUMNS].equals(right[IDENTITY_COLUMNS]):
        raise ValueError("Holdout component identities do not match exactly.")
    left["prediction"] = rank_normalize_by_date(left, "prediction")
    right["prediction"] = rank_normalize_by_date(right, "prediction")
    ensemble = left[IDENTITY_COLUMNS].copy()
    ensemble["prediction"] = 0.5 * left["prediction"] + 0.5 * right["prediction"]
    return left, right, ensemble


def _validate_evaluation_id(evaluation_id: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", evaluation_id):
        raise ValueError("Invalid final evaluation ID.")


def save_final_preparation(
    evaluation_id: str,
    feature_manifest: dict,
    model_files: dict[str, Path],
    medium_predictions: pd.DataFrame,
    top250_predictions: pd.DataFrame,
    ensemble_predictions: pd.DataFrame,
    *,
    training_dates: list[str],
    output_root: Path = OUTPUT_ROOT,
    git_state: dict | None = None,
    policy_path: Path = V1_POLICY_PATH,
    additional_artifacts: dict[str, Path] | None = None,
    preparation_details: dict | None = None,
    _synthetic_expected_rows: int | None = None,
) -> Path:
    _validate_evaluation_id(evaluation_id)
    policy = load_v1_policy(policy_path)
    destination = output_root / "v6.0" / "final_holdout" / evaluation_id
    if destination.exists():
        raise FileExistsError(f"Final evaluation already exists: {destination}")
    if feature_manifest.get("policy_id") != "medium_top250_abs_mean_ic_v1" or feature_manifest.get("selected_feature_count") != 250:
        raise ValueError("Final feature manifest does not match frozen V1.")
    config = get_research_config("v6.0")
    if not training_dates or max(training_dates) != config.research_end or any(date >= config.holdout_gap_start for date in training_dates):
        raise ValueError("Final preparation training dates are incompatible.")
    medium = canonical_holdout_predictions(medium_predictions, "Medium")
    top250 = canonical_holdout_predictions(top250_predictions, "Top-250")
    ensemble = canonical_holdout_predictions(ensemble_predictions, "Ensemble")
    if not medium[IDENTITY_COLUMNS].equals(top250[IDENTITY_COLUMNS]) or not medium[IDENTITY_COLUMNS].equals(ensemble[IDENTITY_COLUMNS]):
        raise ValueError("Final preparation prediction identities do not match.")
    expected_rows = policy["final_holdout"]["expected_rows"]
    if _synthetic_expected_rows is not None:
        if not (git_state or {}).get("synthetic"):
            raise ValueError("Synthetic row override is test-only.")
        expected_rows = _synthetic_expected_rows
    if len(ensemble) != expected_rows:
        raise ValueError("Final preparation does not contain 365,363 identities.")
    if set(model_files) != {"medium", "top250"} or any(not path.is_file() for path in model_files.values()):
        raise ValueError("Final preparation requires exactly two saved models.")

    preparation = destination / "preparation"
    models = preparation / "models"
    models.mkdir(parents=True, exist_ok=False)
    shutil.copy2(policy_path, preparation / "frozen_policy.json")
    with (preparation / "feature_manifest.json").open("x", encoding="utf-8") as file:
        json.dump(feature_manifest, file, indent=2, allow_nan=False)
    for name, source in model_files.items():
        shutil.copy2(source, models / f"{name}.txt")
    for name, frame in (("medium", medium), ("top250", top250), ("ensemble", ensemble)):
        frame.to_parquet(preparation / f"{name}_predictions.parquet", index=False)
    artifact_names = [
        "frozen_policy.json", "feature_manifest.json",
        "models/medium.txt", "models/top250.txt",
        "medium_predictions.parquet", "top250_predictions.parquet", "ensemble_predictions.parquet",
    ]
    for name, source in (additional_artifacts or {}).items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Additional artifact paths must be relative and contained.")
        destination_path = preparation / name
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination_path)
        artifact_names.append(name)
    hashes = {name: sha256_file(preparation / name) for name in artifact_names}
    metadata = {
        "evaluation_id": evaluation_id,
        "state": PREPARATION_COMPLETE,
        "policy_id": policy["policy_id"],
        "policy_sha256": hashes["frozen_policy.json"],
        "training_start": min(training_dates),
        "training_end": max(training_dates),
        "training_dates": len(set(training_dates)),
        "holdout_start": config.holdout_start,
        "holdout_end": config.holdout_end,
        "holdout_dates": config.expected_holdout_dates,
        "holdout_rows": len(ensemble),
        "target_columns_accessed": [],
        "artifact_sha256": hashes,
        "preparation_details": preparation_details or {},
        "git": git_state or current_git_state(),
    }
    with (preparation / "metadata.json").open("x", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, allow_nan=False)
    with (preparation / "PREPARATION_COMPLETED.json").open("x", encoding="utf-8") as file:
        json.dump({"state": PREPARATION_COMPLETE}, file, indent=2)
    return destination


def approved_final_training_dates() -> list[str]:
    """Return the frozen observed weekly index without reading any target column."""
    config = get_research_config("v6.0")
    return pd.date_range(
        config.historical_start, config.research_end, freq="7D"
    ).strftime("%Y-%m-%d").tolist()


def verify_observed_final_training_dates(data_dir: Path) -> tuple[list[str], int]:
    """Verify exact observed dates and row count using target-free reads."""
    config = get_research_config("v6.0")
    train = pq.read_table(data_dir / "train.parquet", columns=["date"]).to_pandas()
    validation = pq.read_table(
        data_dir / "validation.parquet",
        columns=["date"],
        filters=[("date", "<=", config.research_end)],
    ).to_pandas()
    observed = sorted(
        pd.concat([train["date"], validation["date"]], ignore_index=True)
        .astype(str)
        .unique()
    )
    approved = approved_final_training_dates()
    if observed != approved:
        raise ValueError("Observed final-training dates changed from the frozen index.")
    return observed, len(train) + len(validation)


def extend_final_training_ic(
    features: list[str],
    *,
    data_dir: Path,
    source_analysis_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Reuse verified IC rows and calculate only missing eligible training dates."""
    source = load_feature_quality_artifacts(source_analysis_dir)
    metadata = source["metadata"]
    existing = source["per_date_ic"]
    existing_dates = sorted(existing["date"].astype(str).unique())
    if (
        len(existing_dates) != EXPECTED_EXISTING_IC_DATES
        or existing_dates[-1] != "2024-08-30"
    ):
        raise ValueError("Existing IC coverage does not match the approved source artifact.")
    eligible_dates = approved_final_training_dates()
    missing_dates = sorted(set(eligible_dates) - set(existing_dates))
    if (
        len(eligible_dates) != EXPECTED_FINAL_IC_DATES
        or len(missing_dates) != EXPECTED_ADDITIONAL_IC_DATES
        or missing_dates[0] != "2024-09-06"
        or missing_dates[-1] != "2025-04-25"
    ):
        raise ValueError("Final IC extension date index changed unexpectedly.")
    pieces = []
    columns = ["date", "target_jupiter_60", *features]
    started = time.perf_counter()
    for date in missing_dates:
        frame = pd.read_parquet(
            data_dir / "validation.parquet",
            columns=columns,
            filters=[("date", "=", date)],
        )
        if frame.empty or set(frame["date"].astype(str)) != {date}:
            raise ValueError(f"Missing final IC training date: {date}")
        if frame["target_jupiter_60"].isna().any():
            raise ValueError(f"Final IC training target is immature on {date}.")
        pieces.append(calculate_cross_sectional_spearman_ic(frame, features))
    additional = pd.concat(pieces, ignore_index=True)
    summary = build_final_training_ic_summary(
        existing, additional, features, eligible_dates, metadata
    )
    combined = pd.concat([existing, additional], ignore_index=True).sort_values(
        ["date", "feature"], kind="stable"
    )
    provenance = {
        "source_analysis_id": metadata.get("analysis_id"),
        "source_completion_marker": "COMPLETED.json",
        "source_artifact_sha256": {
            name: sha256_file(source_analysis_dir / name)
            for name in ("COMPLETED.json", "metadata.json", "per_date_ic.parquet")
        },
        "source_validation_file": str(data_dir.name + "/validation.parquet"),
        "source_validation_bytes": (data_dir / "validation.parquet").stat().st_size,
        "dataset_version": "v6.0",
        "target": "target_jupiter_60",
        "ic_definition": metadata.get("ic_definition"),
        "missingness_policy": metadata.get("missingness_policy"),
        "feature_count": len(features),
        "existing_dates": len(existing_dates),
        "existing_start": existing_dates[0],
        "existing_end": existing_dates[-1],
        "additional_dates": len(missing_dates),
        "additional_start": missing_dates[0],
        "additional_end": missing_dates[-1],
        "combined_dates": len(eligible_dates),
        "combined_start": eligible_dates[0],
        "combined_end": eligible_dates[-1],
        "additional_runtime_seconds": time.perf_counter() - started,
        "holdout_target_columns_accessed": [],
    }
    return additional, combined, summary, provenance


def _train_final_component(
    training_features: pd.DataFrame,
    training_target: pd.Series,
    holdout_features: pd.DataFrame,
    feature_names: list[str],
    model_path: Path,
) -> tuple[pd.Series, dict]:
    started = time.perf_counter()
    model = create_lightgbm_model(DEFAULT_LIGHTGBM_PARAMS)
    model = train_validated_lightgbm(
        model,
        training_features.loc[:, feature_names],
        training_target,
        feature_names,
    )
    training_seconds = time.perf_counter() - started
    predictions = predict_validated_lightgbm(
        model, holdout_features.loc[:, feature_names]
    )
    model.booster_.save_model(str(model_path))
    reloaded = lgb.Booster(model_file=str(model_path))
    reloaded_predictions = reloaded.predict(holdout_features.loc[:, feature_names])
    if not np.allclose(
        predictions.to_numpy(), reloaded_predictions, atol=1e-12, rtol=0
    ):
        raise ValueError("Reloaded final model predictions changed.")
    diagnostics = {
        "feature_count": len(feature_names),
        "training_rows": len(training_features),
        "training_seconds": training_seconds,
        "trees": int(model.booster_.num_trees()),
        "model_bytes": model_path.stat().st_size,
        "reload_max_absolute_prediction_difference": float(
            np.max(np.abs(predictions.to_numpy() - reloaded_predictions))
        ),
    }
    del model, reloaded, reloaded_predictions
    gc.collect()
    return predictions, diagnostics


def run_final_preparation(
    evaluation_id: str,
    *,
    data_dir: Path | None = None,
    output_root: Path = OUTPUT_ROOT,
    source_ic_analysis_id: str = SOURCE_IC_ANALYSIS_ID,
) -> Path:
    """Execute the one frozen, target-free v6 final preparation."""
    started = time.perf_counter()
    data_dir = data_dir or resolve_data_dir("v6.0")
    evaluation_dir = output_root / "v6.0" / "final_holdout" / evaluation_id
    if evaluation_dir.exists():
        raise FileExistsError(f"Final evaluation already exists: {evaluation_dir}")
    policy = load_v1_policy()
    features = resolve_v6_numeric_feature_set("medium", data_dir)
    if len(features) != 350:
        raise ValueError("Frozen final preparation requires 350 Medium features.")
    source_dir = output_root / "v6.0" / "feature_quality" / source_ic_analysis_id
    additional, combined, summary, ic_provenance = extend_final_training_ic(
        features, data_dir=data_dir, source_analysis_dir=source_dir
    )
    feature_manifest = select_final_top250(summary, features)
    feature_manifest.update(
        {
            "source_ic_analysis_id": source_ic_analysis_id,
            "source_ic_existing_dates": EXPECTED_EXISTING_IC_DATES,
            "source_ic_additional_dates": EXPECTED_ADDITIONAL_IC_DATES,
            "source_ic_combined_dates": EXPECTED_FINAL_IC_DATES,
            "git": current_git_state(),
            "frozen_policy_id": policy["policy_id"],
        }
    )
    top250_features = feature_manifest["model_feature_names"]
    training_dates, expected_training_rows = verify_observed_final_training_dates(data_dir)
    training = load_v6_final_training_data(features, data_dir)
    holdout = load_v6_final_holdout_features(features, data_dir)
    if len(training.features) != expected_training_rows:
        raise ValueError("Final-training row count changed during feature loading.")
    training_details = {
        "start": training_dates[0],
        "end": training_dates[-1],
        "dates": len(training_dates),
        "rows": len(training.features),
        "target": "target_jupiter_60",
    }
    with tempfile.TemporaryDirectory(prefix="numerai-v6-final-") as temporary:
        temp = Path(temporary)
        additional_path = temp / "ic_extension.parquet"
        combined_path = temp / "combined_per_date_ic.parquet"
        summary_path = temp / "final_ic_summary.parquet"
        provenance_path = temp / "ic_provenance.json"
        additional.to_parquet(additional_path, index=False)
        combined.to_parquet(combined_path, index=False)
        summary.to_parquet(summary_path, index=False)
        provenance_path.write_text(
            json.dumps(ic_provenance, indent=2, allow_nan=False), encoding="utf-8"
        )
        medium_model_path = temp / "medium.txt"
        top250_model_path = temp / "top250.txt"
        medium_values, medium_diagnostics = _train_final_component(
            training.features, training.target, holdout.features, features, medium_model_path
        )
        medium_predictions = holdout.identity.copy()
        medium_predictions["prediction"] = medium_values.to_numpy()
        del medium_values
        gc.collect()
        top250_values, top250_diagnostics = _train_final_component(
            training.features,
            training.target,
            holdout.features,
            top250_features,
            top250_model_path,
        )
        top250_predictions = holdout.identity.copy()
        top250_predictions["prediction"] = top250_values.to_numpy()
        del top250_values, training
        gc.collect()
        _medium_ranked, _top250_ranked, ensemble = build_fixed_holdout_ensemble(
            medium_predictions, top250_predictions
        )
        details = {
            "training": training_details,
            "ic": ic_provenance,
            "models": {"medium": medium_diagnostics, "top250": top250_diagnostics},
            "ensemble": {
                "normalization": "per-date average rank normalized as (rank - 0.5) / n",
                "weights": {"medium": 0.5, "top250": 0.5},
            },
            "runtime_before_artifact_persistence_seconds": time.perf_counter() - started,
        }
        destination = save_final_preparation(
            evaluation_id,
            feature_manifest,
            {"medium": medium_model_path, "top250": top250_model_path},
            medium_predictions,
            top250_predictions,
            ensemble,
            training_dates=training_dates,
            output_root=output_root,
            additional_artifacts={
                "feature_quality/ic_extension.parquet": additional_path,
                "feature_quality/combined_per_date_ic.parquet": combined_path,
                "feature_quality/final_ic_summary.parquet": summary_path,
                "feature_quality/ic_provenance.json": provenance_path,
            },
            preparation_details=details,
        )
    return destination


def save_preparation_failure(
    evaluation_id: str,
    error: BaseException,
    *,
    output_root: Path = OUTPUT_ROOT,
) -> Path:
    """Preserve a non-retriable diagnostic without implying preparation success."""
    _validate_evaluation_id(evaluation_id)
    evaluation_dir = output_root / "v6.0" / "final_holdout" / evaluation_id
    if evaluation_dir.exists():
        raise FileExistsError(f"Cannot overwrite final evaluation state: {evaluation_dir}")
    evaluation_dir.mkdir(parents=True, exist_ok=False)
    payload = {
        "state": "preparation_failed",
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": traceback.format_exc(),
        "target_columns_accessed": [],
        "scoring_enabled": False,
        "git": current_git_state(),
    }
    with (evaluation_dir / "PREPARATION_FAILED.json").open("x", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, allow_nan=False)
    return evaluation_dir


def verify_final_preparation(evaluation_dir: Path) -> dict:
    preparation = evaluation_dir / "preparation"
    required = ("PREPARATION_COMPLETED.json", "metadata.json")
    if any(not (preparation / name).is_file() for name in required):
        raise FileNotFoundError("Final preparation is incomplete.")
    with (preparation / "metadata.json").open(encoding="utf-8") as file:
        metadata = json.load(file)
    if metadata.get("state") != PREPARATION_COMPLETE:
        raise ValueError("Final preparation state is invalid.")
    for name, expected in metadata.get("artifact_sha256", {}).items():
        path = preparation / name
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Final preparation artifact hash mismatch: {name}")
    policy = json.loads((preparation / "frozen_policy.json").read_text(encoding="utf-8"))
    if policy.get("policy_id") != metadata.get("policy_id"):
        raise ValueError("Frozen policy identity changed after preparation.")
    predictions = pd.read_parquet(preparation / "ensemble_predictions.parquet")
    canonical = canonical_holdout_predictions(predictions, "Ensemble")
    if len(canonical) != metadata.get("holdout_rows"):
        raise ValueError("Prepared holdout row identity count changed.")
    return metadata


def verify_preparation_seal(
    evaluation_dir: Path,
    seal_path: Path = PREPARATION_SEAL_PATH,
) -> tuple[dict, dict]:
    """Verify the committed seal and every immutable preparation identity."""
    if sha256_file(seal_path) != PREPARATION_SEAL_SHA256:
        raise ValueError("Committed preparation seal hash changed.")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if (
        seal.get("evaluation_id") != evaluation_dir.name
        or seal.get("policy_id") != "numerai_classic_v6_v1"
        or seal.get("frozen_policy_sha256")
        != "4ccddd457707232feb7391389b4a387e57dcd1871af898aeefd82a1f8fa7381e"
    ):
        raise ValueError("Preparation seal identity is incompatible.")
    preparation = evaluation_dir / "preparation"
    metadata = verify_final_preparation(evaluation_dir)
    if sha256_file(preparation / "metadata.json") != seal["preparation_metadata"]["sha256"]:
        raise ValueError("Preparation metadata does not match the committed seal.")
    if sha256_file(preparation / "PREPARATION_COMPLETED.json") != seal["preparation_completion_marker"]["sha256"]:
        raise ValueError("Preparation completion marker does not match the seal.")
    expected_hashes = seal.get("preparation_artifact_sha256", {})
    if len(expected_hashes) != 11 or expected_hashes != metadata.get("artifact_sha256"):
        raise ValueError("Preparation artifact hash inventory differs from the seal.")
    for name, expected in expected_hashes.items():
        if sha256_file(preparation / name) != expected:
            raise ValueError(f"Sealed preparation artifact changed: {name}")
    if metadata.get("target_columns_accessed") != [] or seal.get("target_columns_accessed") != []:
        raise ValueError("Preparation target-access record is not empty.")
    if seal.get("status") != {
        "preparation": "prepared",
        "scoring": "unscored",
        "holdout_targets": "sealed",
    }:
        raise ValueError("Preparation seal no longer records the pre-scoring state.")
    return seal, metadata


def require_scoring_authorization(evaluation_dir: Path, authorized: bool) -> dict:
    if not authorized:
        raise PermissionError("Explicit holdout target-access authorization is required.")
    if (evaluation_dir / "scoring" / "FINAL_EVALUATION_COMPLETED.json").exists():
        raise FileExistsError("Final evaluation has already been scored.")
    return verify_final_preparation(evaluation_dir)


def load_authorized_holdout_targets(
    evaluation_dir: Path,
    *,
    authorized: bool,
    data_dir: Path | None = None,
) -> pd.DataFrame:
    """Access holdout targets only after immutable preparation verification."""
    require_scoring_authorization(evaluation_dir, authorized)
    config = get_research_config("v6.0")
    data_dir = data_dir or resolve_data_dir("v6.0")
    columns = [*IDENTITY_COLUMNS, config.target_column]
    frame = pq.read_table(
        data_dir / "validation.parquet",
        columns=columns,
        filters=[
            ("date", ">=", config.holdout_start),
            ("date", "<=", config.holdout_end),
        ],
    ).to_pandas()
    if len(frame) != 365_363 or frame["date"].nunique() != 52:
        raise ValueError("Authorized holdout targets have unexpected coverage.")
    expected_dates = set(
        pd.date_range(config.holdout_start, config.holdout_end, freq="7D").strftime("%Y-%m-%d")
    )
    if set(frame["date"].astype(str)) != expected_dates:
        raise ValueError("Authorized holdout targets contain unsupported dates.")
    if frame.duplicated(IDENTITY_COLUMNS).any():
        raise ValueError("Authorized holdout targets contain duplicate identities.")
    if frame[config.target_column].isna().any() or not np.isfinite(frame[config.target_column]).all():
        raise ValueError("Authorized holdout targets are incomplete or nonfinite.")
    return frame.sort_values(IDENTITY_COLUMNS, na_position="first").reset_index(drop=True)


def score_authorized_final_predictions(
    predictions: pd.DataFrame,
    targets: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Score immutable holdout predictions with existing official formulas."""
    canonical = canonical_holdout_predictions(predictions, "Ensemble")
    target_identity = targets.loc[:, IDENTITY_COLUMNS].reset_index(drop=True)
    if not canonical[IDENTITY_COLUMNS].equals(target_identity):
        raise ValueError("Authorized holdout targets do not align with predictions.")
    scored = canonical.copy()
    scored["fold_id"] = 1
    scored["target_jupiter_60"] = targets["target_jupiter_60"].to_numpy()
    legacy = calculate_v6_numerai_corr_by_date(scored)
    ncorr = calculate_v6_ncorr_by_date(
        scored, reference_loader=load_final_holdout_neutralizers_for_date
    )
    per_date = legacy.merge(ncorr, on=["date", "fold_id", "observation_count"], validate="one_to_one")
    metrics = combine_v6_metric_summaries(
        summarize_v6_oof_corr(per_date), summarize_v6_ncorr(per_date)
    )
    scores = per_date["ncorr"]
    metrics["final_reporting"] = {
        "worst_weekly_ncorr": float(scores.min()),
        "p10_weekly_ncorr": float(scores.quantile(0.10)),
        "automatic_model_selection": False,
    }
    return per_date, metrics


def summarize_final_candidate(per_date: pd.DataFrame) -> dict:
    """Create the predeclared final summaries from already-scored weekly rows."""
    scores = per_date["ncorr"].astype(float)
    legacy = per_date["numerai_corr_jupiter_60"].astype(float)
    ncorr_std = float(scores.std())
    legacy_std = float(legacy.std())
    return {
        "scored_dates": int(len(per_date)),
        "scored_identities": int(per_date["observation_count"].sum()),
        "mean_ncorr": float(scores.mean()),
        "ncorr_std": ncorr_std,
        "ncorr_mean_std_ratio": float(scores.mean() / ncorr_std),
        "positive_ncorr_date_percentage": float((scores > 0).mean() * 100),
        "median_weekly_ncorr": float(scores.median()),
        "worst_weekly_ncorr": float(scores.min()),
        "best_weekly_ncorr": float(scores.max()),
        "p10_weekly_ncorr": float(scores.quantile(0.10)),
        "mean_legacy_corr": float(legacy.mean()),
        "legacy_corr_std": legacy_std,
        "legacy_corr_mean_std_ratio": float(legacy.mean() / legacy_std),
    }


def paired_weekly_diagnostics(
    primary: pd.DataFrame,
    component: pd.DataFrame,
    label: str,
) -> dict:
    joined = primary[["date", "ncorr"]].merge(
        component[["date", "ncorr"]], on="date", suffixes=("_primary", "_component"), validate="one_to_one"
    )
    difference = joined["ncorr_primary"] - joined["ncorr_component"]
    return {
        "comparison": label,
        "date_count": int(len(difference)),
        "mean_difference": float(difference.mean()),
        "median_difference": float(difference.median()),
        "std_difference": float(difference.std()),
        "outperforming_date_percentage": float((difference > 0).mean() * 100),
        "p10_difference": float(difference.quantile(0.10)),
        "p90_difference": float(difference.quantile(0.90)),
        "largest_improvement": float(difference.max()),
        "largest_decline": float(difference.min()),
    }


def temporal_final_summary(per_date: pd.DataFrame) -> dict:
    dated = per_date.assign(year=per_date["date"].str.slice(0, 4).astype(int))
    periods = []
    for year, group in dated.groupby("year", sort=True):
        summary = summarize_final_candidate(group)
        summary["year"] = int(year)
        periods.append(summary)
    best = per_date.loc[per_date["ncorr"].idxmax()]
    worst = per_date.loc[per_date["ncorr"].idxmin()]
    return {
        "per_year": periods,
        "best_week": {"date": str(best["date"]), "ncorr": float(best["ncorr"])},
        "worst_week": {"date": str(worst["date"]), "ncorr": float(worst["ncorr"])},
    }


def compare_final_to_research(
    final_summary: dict,
    research_metrics_path: Path,
    research_per_date_path: Path,
) -> dict:
    research = json.loads(research_metrics_path.read_text(encoding="utf-8"))
    research_dates = pd.read_parquet(research_per_date_path)
    ncorr = research["ncorr"]["overall"]
    legacy = research["legacy_corr"]["overall"]
    research_summary = {
        "scored_dates": int(ncorr["date_count"]),
        "scored_identities": int(ncorr["observation_count"]),
        "mean_ncorr": float(ncorr["mean"]),
        "ncorr_std": float(ncorr["std"]),
        "ncorr_mean_std_ratio": float(ncorr["sharpe_like"]),
        "positive_ncorr_date_percentage": float(ncorr["positive_date_percentage"]),
        "worst_weekly_ncorr": float(research_dates["ncorr"].min()),
        "p10_weekly_ncorr": float(research_dates["ncorr"].quantile(0.10)),
        "mean_legacy_corr": float(legacy["mean"]),
    }
    keys = (
        "mean_ncorr", "ncorr_std", "ncorr_mean_std_ratio",
        "positive_ncorr_date_percentage", "worst_weekly_ncorr",
        "p10_weekly_ncorr", "mean_legacy_corr",
    )
    return {
        "research_experiment_id": RESEARCH_ENSEMBLE_ID,
        "research_scope": "Research Validation OOF model-selection evidence",
        "final_scope": "Independent fixed-model final holdout evidence",
        "research": research_summary,
        "final": final_summary,
        "absolute_difference_final_minus_research": {
            key: float(final_summary[key] - research_summary[key]) for key in keys
        },
        "relative_mean_ncorr_difference": float(
            final_summary["mean_ncorr"] / research_summary["mean_ncorr"] - 1
        ),
        "periods_are_not_exchangeable": True,
        "automatic_pass_fail_threshold": None,
    }


def save_final_scoring(
    evaluation_dir: Path,
    preparation_metadata: dict,
    per_date_metrics: pd.DataFrame,
    metrics: dict,
    component_diagnostics: dict,
) -> Path:
    verify_final_preparation(evaluation_dir)
    scoring = evaluation_dir / "scoring"
    if scoring.exists():
        raise FileExistsError("Final scoring artifacts already exist.")
    scoring.mkdir()
    per_date_metrics.to_parquet(scoring / "per_date_metrics.parquet", index=False)
    payload = {
        "state": SCORING_COMPLETE,
        "preparation_policy_sha256": preparation_metadata["policy_sha256"],
        "metrics": metrics,
        "component_diagnostics": component_diagnostics,
        "reporting_contract": [
            "mean_ncorr", "ncorr_std", "ncorr_mean_std_ratio",
            "positive_date_percentage", "worst_weekly_ncorr", "p10_weekly_ncorr",
            "legacy_corr", "component_diagnostics", "identity_integrity",
        ],
        "automatic_model_selection": False,
    }
    with (scoring / "metrics.json").open("x", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, allow_nan=False)
    with (scoring / "FINAL_EVALUATION_COMPLETED.json").open("x", encoding="utf-8") as file:
        json.dump({"state": SCORING_COMPLETE}, file, indent=2)
    return scoring


def verify_final_scoring_reference(data_dir: Path | None = None) -> dict:
    path = validation_neutralizer_path(data_dir)
    actual_hash = sha256_file(path)
    if actual_hash != V6_VALIDATION_NEUTRALIZER_SHA256:
        raise ValueError("Official v6 neutralizer hash changed.")
    columns = ordered_neutralizer_columns(path, V6_NEUTRALIZER_COLUMN_COUNT)
    return {
        "path": "data/v6.0/validation_neutralizer.parquet",
        "sha256": actual_hash,
        "column_count": len(columns),
        "columns_start": columns[0],
        "columns_end": columns[-1],
        "numerai_tools_version": "0.7.2",
        "primary_metric": "NCORR",
        "api_display_name": "neutral_corr",
        "metric_version": 1,
        "diagnostic_metric": "Legacy CORR",
        "nmmc_calculated": False,
        "nbmc_calculated": False,
    }


def _write_json_once(path: Path, payload: dict) -> None:
    with path.open("x", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, allow_nan=False)


def persist_final_evaluation(
    evaluation_dir: Path,
    candidate_per_date: dict[str, pd.DataFrame],
    aggregate: dict,
    paired: dict,
    temporal: dict,
    research_comparison: dict,
    scoring_reference: dict,
    runtime: dict,
    preparation_seal: dict,
) -> Path:
    """Persist already-calculated results and write the completion marker last."""
    scoring = evaluation_dir / "scoring"
    required_state = {
        "SCORING_STARTED.json", "AUTHORIZATION.json", "TARGET_ACCESS.json"
    }
    if not scoring.is_dir() or not required_state.issubset({p.name for p in scoring.iterdir()}):
        raise ValueError("Final scoring state records are incomplete.")
    if (scoring / "FINAL_EVALUATION_COMPLETED.json").exists():
        raise FileExistsError("Final evaluation already completed.")
    combined = pd.concat(
        [frame.assign(candidate=name) for name, frame in candidate_per_date.items()],
        ignore_index=True,
    )
    combined.to_parquet(scoring / "per_date_metrics.parquet", index=False)
    _write_json_once(scoring / "aggregate_metrics.json", aggregate)
    _write_json_once(scoring / "paired_diagnostics.json", paired)
    _write_json_once(scoring / "temporal_summaries.json", temporal)
    _write_json_once(scoring / "research_oof_comparison.json", research_comparison)
    metadata = {
        "evaluation_id": evaluation_dir.name,
        "state": SCORING_COMPLETE,
        "primary_candidate": "ensemble",
        "diagnostic_candidates": ["medium", "top250"],
        "policy_id": preparation_seal["policy_id"],
        "preparation_seal_sha256": PREPARATION_SEAL_SHA256,
        "scoring_reference": scoring_reference,
        "runtime": runtime,
        "target_values_persisted": False,
        "automatic_model_reselection": False,
        "git": current_git_state(),
    }
    _write_json_once(scoring / "scoring_metadata.json", metadata)
    artifact_names = [
        "SCORING_STARTED.json", "AUTHORIZATION.json", "TARGET_ACCESS.json",
        "per_date_metrics.parquet", "aggregate_metrics.json",
        "paired_diagnostics.json", "temporal_summaries.json",
        "research_oof_comparison.json", "scoring_metadata.json",
    ]
    hashes = {name: sha256_file(scoring / name) for name in artifact_names}
    _write_json_once(scoring / "artifact_hashes.json", hashes)
    hashes["artifact_hashes.json"] = sha256_file(scoring / "artifact_hashes.json")
    _write_json_once(
        scoring / "FINAL_EVALUATION_COMPLETED.json",
        {
            "state": SCORING_COMPLETE,
            "completed_at_utc": utc_now(),
            "artifact_sha256": hashes,
        },
    )
    return scoring


def load_final_evaluation_report(evaluation_dir: Path) -> dict:
    """Load verified saved metrics without reading targets or recalculating scores."""
    scoring = evaluation_dir / "scoring"
    marker_path = scoring / "FINAL_EVALUATION_COMPLETED.json"
    if not marker_path.is_file():
        raise FileNotFoundError("Final evaluation is not complete.")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("state") != SCORING_COMPLETE:
        raise ValueError("Final evaluation completion state is invalid.")
    for name, expected in marker.get("artifact_sha256", {}).items():
        if sha256_file(scoring / name) != expected:
            raise ValueError(f"Final scoring artifact hash mismatch: {name}")
    return {
        "aggregate": json.loads((scoring / "aggregate_metrics.json").read_text()),
        "paired": json.loads((scoring / "paired_diagnostics.json").read_text()),
        "temporal": json.loads((scoring / "temporal_summaries.json").read_text()),
        "research_comparison": json.loads(
            (scoring / "research_oof_comparison.json").read_text()
        ),
        "metadata": json.loads((scoring / "scoring_metadata.json").read_text()),
        "per_date": pd.read_parquet(scoring / "per_date_metrics.parquet"),
        "completion": marker,
    }


def run_final_scoring(
    evaluation_id: str,
    *,
    authorized: bool,
    confirmed: bool,
    data_dir: Path | None = None,
    output_root: Path = OUTPUT_ROOT,
    seal_path: Path = PREPARATION_SEAL_PATH,
    target_loader=None,
    score_function=None,
    reference_verifier=None,
) -> Path:
    """Execute the one authorized scoring transaction from sealed predictions."""
    if not authorized or not confirmed:
        raise PermissionError("Both target-access authorization flags are required.")
    evaluation_dir = output_root / "v6.0" / "final_holdout" / evaluation_id
    scoring = evaluation_dir / "scoring"
    if scoring.exists():
        raise FileExistsError("A final scoring attempt already exists; retries are forbidden.")
    seal, preparation_metadata = verify_preparation_seal(evaluation_dir, seal_path)
    preparation = evaluation_dir / "preparation"
    predictions = {
        name: canonical_holdout_predictions(
            pd.read_parquet(preparation / f"{name}_predictions.parquet"), name
        )
        for name in ("medium", "top250", "ensemble")
    }
    identity = predictions["ensemble"][IDENTITY_COLUMNS]
    if any(not frame[IDENTITY_COLUMNS].equals(identity) for frame in predictions.values()):
        raise ValueError("Sealed candidate identities do not match.")
    _, _, rebuilt = build_fixed_holdout_ensemble(
        predictions["medium"], predictions["top250"]
    )
    if not np.array_equal(
        rebuilt["prediction"].to_numpy(), predictions["ensemble"]["prediction"].to_numpy()
    ):
        raise ValueError("Sealed ensemble does not match the frozen 50/50 policy.")
    reference_verifier = reference_verifier or verify_final_scoring_reference
    scoring_reference = reference_verifier(data_dir)
    if (
        scoring_reference.get("sha256") != V6_VALIDATION_NEUTRALIZER_SHA256
        or scoring_reference.get("column_count") != 300
    ):
        raise ValueError("Official scoring reference identity is incompatible.")
    scoring.mkdir(exist_ok=False)
    started_at = utc_now()
    _write_json_once(
        scoring / "SCORING_STARTED.json",
        {
            "state": "scoring_started",
            "started_at_utc": started_at,
            "evaluation_id": evaluation_id,
            "preparation_seal_sha256": PREPARATION_SEAL_SHA256,
            "preparation_metadata_sha256": seal["preparation_metadata"]["sha256"],
            "target_accessed": False,
        },
    )
    _write_json_once(
        scoring / "AUTHORIZATION.json",
        {
            "authorized_at_utc": utc_now(),
            "run": True,
            "authorize_holdout_target_access": True,
            "confirm_final_scoring": True,
            "scope": "target_jupiter_60 for the 52 frozen final-holdout dates only",
        },
    )
    started = time.perf_counter()
    rss_start = psutil.Process().memory_info().rss
    target_loader = target_loader or load_authorized_holdout_targets
    score_function = score_function or score_authorized_final_predictions
    try:
        targets = target_loader(
            evaluation_dir, authorized=True, data_dir=data_dir
        )
        if not predictions["ensemble"][IDENTITY_COLUMNS].equals(
            targets[IDENTITY_COLUMNS].reset_index(drop=True)
        ):
            raise ValueError("Authorized targets do not match sealed prediction identities.")
        _write_json_once(
            scoring / "TARGET_ACCESS.json",
            {
                "first_authorized_access_at_utc": utc_now(),
                "columns": [*IDENTITY_COLUMNS, "target_jupiter_60"],
                "date_start": "2025-08-22",
                "date_end": "2026-08-14",
                "date_count": 52,
                "row_count": 365363,
                "other_target_columns_accessed": [],
            },
        )
        candidate_per_date = {}
        for name in ("medium", "top250", "ensemble"):
            candidate_per_date[name], _ = score_function(predictions[name], targets)
        del targets
        gc.collect()
        aggregate = {
            name: summarize_final_candidate(frame)
            for name, frame in candidate_per_date.items()
        }
        paired = {
            "ensemble_minus_medium": paired_weekly_diagnostics(
                candidate_per_date["ensemble"], candidate_per_date["medium"],
                "Frozen V1 Ensemble minus Medium LightGBM",
            ),
            "ensemble_minus_top250": paired_weekly_diagnostics(
                candidate_per_date["ensemble"], candidate_per_date["top250"],
                "Frozen V1 Ensemble minus Top-250 LightGBM",
            ),
        }
        temporal = {
            name: temporal_final_summary(frame)
            for name, frame in candidate_per_date.items()
        }
        research_dir = output_root / "v6.0" / "ensembles" / RESEARCH_ENSEMBLE_ID
        research_comparison = compare_final_to_research(
            aggregate["ensemble"],
            research_dir / "metrics.json",
            research_dir / "per_date_metrics.parquet",
        )
        runtime = {
            "scoring_seconds_before_artifact_persistence": time.perf_counter() - started,
            "rss_bytes_start": rss_start,
            "rss_bytes_end": psutil.Process().memory_info().rss,
            "peak_rss_measured": False,
        }
        destination = persist_final_evaluation(
            evaluation_dir, candidate_per_date, aggregate, paired, temporal,
            research_comparison, scoring_reference, runtime, seal,
        )
    except Exception as error:
        if not (scoring / "SCORING_FAILED.json").exists():
            _write_json_once(
                scoring / "SCORING_FAILED.json",
                {
                    "state": "scoring_failed",
                    "failed_at_utc": utc_now(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                    "retry_forbidden": True,
                },
            )
        raise
    if verify_final_preparation(evaluation_dir) != preparation_metadata:
        raise ValueError("Preparation metadata changed during scoring.")
    load_final_evaluation_report(evaluation_dir)
    return destination


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--version", required=True)
    prepare.add_argument("--evaluation-id", required=True)
    prepare.add_argument("--run", action="store_true")
    prepare.add_argument("--confirm-final-preparation", action="store_true")
    score = subparsers.add_parser("score")
    score.add_argument("--version", required=True)
    score.add_argument("--evaluation-id", required=True)
    score.add_argument("--run", action="store_true")
    score.add_argument("--authorize-holdout-target-access", action="store_true")
    score.add_argument("--confirm-final-scoring", action="store_true")
    report = subparsers.add_parser("report")
    report.add_argument("--version", required=True)
    report.add_argument("--evaluation-id", required=True)
    args = parser.parse_args(argv)
    if args.version != "v6.0":
        raise SystemExit("Final holdout supports only v6.0.")
    if args.command == "prepare" and (not args.run or not args.confirm_final_preparation):
        raise SystemExit("Final preparation requires --run --confirm-final-preparation.")
    if args.command == "prepare":
        try:
            print(run_final_preparation(args.evaluation_id))
        except Exception as error:
            save_preparation_failure(args.evaluation_id, error)
            raise
        return
    if args.command == "report":
        evaluation_dir = OUTPUT_ROOT / "v6.0" / "final_holdout" / args.evaluation_id
        saved = load_final_evaluation_report(evaluation_dir)
        print(json.dumps(saved["aggregate"], indent=2, allow_nan=False))
        return
    if not args.run or not args.authorize_holdout_target_access or not args.confirm_final_scoring:
        raise SystemExit("Final scoring requires explicit run, target access, and scoring confirmation.")
    print(
        run_final_scoring(
            args.evaluation_id,
            authorized=args.authorize_holdout_target_access,
            confirmed=args.confirm_final_scoring,
        )
    )


if __name__ == "__main__":
    main()
