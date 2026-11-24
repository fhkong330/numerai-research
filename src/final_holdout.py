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
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
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
from src.scoring_references import load_final_holdout_neutralizers_for_date
from src.v1_policy import V1_POLICY_PATH, load_v1_policy


IDENTITY_COLUMNS = ["date", "numerai_ticker", "composite_figi"]
PREPARATION_COMPLETE = "preparation_complete"
SCORING_COMPLETE = "scoring_complete"
SOURCE_IC_ANALYSIS_ID = "v6_medium_ic_001"
EXPECTED_EXISTING_IC_DATES = 1_131
EXPECTED_ADDITIONAL_IC_DATES = 34
EXPECTED_FINAL_IC_DATES = 1_165


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    if not args.run or not args.authorize_holdout_target_access or not args.confirm_final_scoring:
        raise SystemExit("Final scoring requires explicit run, target access, and scoring confirmation.")
    raise SystemExit("Real final scoring remains intentionally disabled.")


if __name__ == "__main__":
    main()
