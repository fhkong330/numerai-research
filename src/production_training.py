"""Refresh a v6 snapshot and train the two frozen V1 production components."""

import argparse
import gc
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from numerapi import NumerAPI

from src.audit_production_data import (
    TARGET,
    audit_local_dataset,
    derive_training_eligibility,
    fetch_public_metadata,
    sha256_file,
    snapshot_compatibility,
    summarize_date_targets,
)
from src.config import DEFAULT_LIGHTGBM_PARAMS
from src.data import resolve_v6_numeric_feature_set
from src.feature_quality import VALID_IC_STATUS, calculate_cross_sectional_spearman_ic
from src.feature_selection import POLICY_ID, SELECTED_FEATURE_COUNT, feature_list_sha256
from src.models import create_lightgbm_model, train_validated_lightgbm
from src.paths import DATA_ROOT, OUTPUT_ROOT, PROJECT_ROOT
from src.v1_policy import V1_POLICY_ID, V1_POLICY_PATH, load_v1_policy


MODEL_VERSION = "v6_v1_production_001"
RECOVERY_MODEL_VERSION = "v6_v1_production_002"
ALLOWED_MODEL_VERSIONS = (MODEL_VERSION, RECOVERY_MODEL_VERSION)
SNAPSHOT_FILES = ("features.json", "train.parquet", "validation.parquet")
FINAL_IC_PATH = (
    OUTPUT_ROOT
    / "v6.0"
    / "final_holdout"
    / "v6_v1_final_001"
    / "preparation"
    / "feature_quality"
    / "combined_per_date_ic.parquet"
)
FINAL_IC_SHA256 = "0ee40d632e3c3a27cfb188d1e3e9d0d30f9b0564df86cc4fae01ca0f7a3de952"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_once(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, allow_nan=False)


def append_progress(path: Path, stage: str, **details) -> None:
    """Persist and display a durable progress event for long production runs."""
    event = {"timestamp_utc": utc_now(), "stage": stage, **details}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(event, allow_nan=False) + "\n")
    print(json.dumps(event, allow_nan=False), flush=True)


def _round_identity(metadata: dict) -> dict:
    round_info = metadata["round"]
    return {
        key: round_info[key]
        for key in ("number", "target", "data_datestamp", "open_time", "close_time")
    }


def refresh_snapshot(
    snapshot_id: str,
    round_number: int,
    *,
    data_root: Path = DATA_ROOT,
    api_factory=NumerAPI,
    metadata_fetcher=fetch_public_metadata,
) -> Path:
    destination = data_root / "v6.0" / "snapshots" / snapshot_id
    if destination.exists():
        raise FileExistsError(f"Snapshot already exists: {destination}")
    destination.mkdir(parents=True)
    write_json_once(
        destination / "REFRESH_STARTED.json",
        {"snapshot_id": snapshot_id, "round_number": round_number, "started_at_utc": utc_now()},
    )
    try:
        before = metadata_fetcher(round_number)
        api = api_factory()
        catalog = set(api.list_datasets())
        downloads = {}
        for name in SNAPSHOT_FILES:
            remote = f"v6.0/{name}"
            if remote not in catalog:
                raise ValueError(f"Official dataset is unavailable: {remote}")
            path = destination / name
            started = utc_now()
            api.download_dataset(remote, str(path))
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"Snapshot download is incomplete: {remote}")
            downloads[name] = {
                "official_path": remote,
                "download_started_at_utc": started,
                "download_completed_at_utc": utc_now(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        after = metadata_fetcher(round_number)
        if _round_identity(before) != _round_identity(after):
            raise ValueError("Public round metadata changed during snapshot refresh.")
        prediction_date = str(before["round"]["data_datestamp"])
        prediction_date = f"{prediction_date[:4]}-{prediction_date[4:6]}-{prediction_date[6:]}"
        local = audit_local_dataset(destination, prediction_date)
        for name, item in downloads.items():
            if local["files"][name]["sha256"] != item["sha256"]:
                raise ValueError(f"Downloaded snapshot hash changed during audit: {name}")
        manifest = {
            "snapshot_id": snapshot_id,
            "dataset_version": "v6.0",
            "state": "complete",
            "completed_at_utc": utc_now(),
            "public_metadata_before": before,
            "public_metadata_after": after,
            "downloads": downloads,
            "local_audit": local,
        }
        write_json_once(destination / "snapshot_manifest.json", manifest)
        write_json_once(
            destination / "SNAPSHOT_COMPLETED.json",
            {
                "state": "complete",
                "snapshot_manifest_sha256": sha256_file(destination / "snapshot_manifest.json"),
            },
        )
        return destination
    except Exception as error:
        write_json_once(
            destination / "REFRESH_FAILED.json",
            {"failed_at_utc": utc_now(), "error_type": type(error).__name__, "error": str(error)},
        )
        raise


def verify_snapshot(snapshot_dir: Path) -> dict:
    manifest_path = snapshot_dir / "snapshot_manifest.json"
    marker_path = snapshot_dir / "SNAPSHOT_COMPLETED.json"
    if not manifest_path.is_file() or not marker_path.is_file():
        raise ValueError("Snapshot is not complete.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("snapshot_manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("Snapshot manifest hash changed.")
    for name in SNAPSHOT_FILES:
        expected = manifest["downloads"][name]["sha256"]
        if sha256_file(snapshot_dir / name) != expected:
            raise ValueError(f"Snapshot file hash changed: {name}")
    if _round_identity(manifest["public_metadata_before"]) != _round_identity(
        manifest["public_metadata_after"]
    ):
        raise ValueError("Snapshot round identities are inconsistent.")
    return manifest


def choose_ic_plan(snapshot_files: dict) -> str:
    return "reuse_and_extend" if snapshot_compatibility(snapshot_files)["historical_ic_reuse_allowed"] else "full_recompute"


def _eligible_dates(snapshot_dir: Path, prediction_date: str) -> tuple[list[str], dict]:
    frames = [
        pq.read_table(snapshot_dir / name, columns=["date", TARGET]).to_pandas()
        for name in ("train.parquet", "validation.parquet")
    ]
    summary = summarize_date_targets(frames)
    eligibility = derive_training_eligibility(summary, prediction_date)
    dates = summary.loc[
        pd.to_datetime(summary["date"])
        <= pd.Timestamp(eligibility["eligible_training_cutoff"]),
        "date",
    ].dt.strftime("%Y-%m-%d").tolist()
    if len(dates) != eligibility["eligible_training_dates"]:
        raise ValueError("Eligible date index and audit count differ.")
    return dates, eligibility


def _calculate_ic_dates(
    snapshot_dir: Path,
    dates: list[str],
    features: list[str],
    *,
    progress_path: Path | None = None,
) -> pd.DataFrame:
    pieces = []
    columns = ["date", TARGET, *features]
    for index, date in enumerate(dates, 1):
        filename = "train.parquet" if date <= "2018-12-28" else "validation.parquet"
        frame = pd.read_parquet(
            snapshot_dir / filename, columns=columns, filters=[("date", "=", date)]
        )
        if frame.empty or set(frame["date"].astype(str)) != {date}:
            raise ValueError(f"Snapshot lacks eligible IC date {date}.")
        if frame[TARGET].isna().any() or not np.isfinite(frame[TARGET]).all():
            raise ValueError(f"Target is not mature and finite on {date}.")
        pieces.append(calculate_cross_sectional_spearman_ic(frame, features))
        if progress_path is not None and (index == 1 or index % 25 == 0 or index == len(dates)):
            append_progress(
                progress_path,
                "ic_computation",
                completed_dates=index,
                total_dates=len(dates),
                latest_date=date,
            )
    return pd.concat(pieces, ignore_index=True)


def summarize_production_ic(
    per_date_ic: pd.DataFrame, features: list[str], eligible_dates: list[str]
) -> pd.DataFrame:
    expected = {(date, feature) for date in eligible_dates for feature in features}
    actual = set(zip(per_date_ic["date"].astype(str), per_date_ic["feature"].astype(str)))
    if actual != expected or per_date_ic.duplicated(["date", "feature"]).any():
        raise ValueError("Production IC coverage is incomplete or inconsistent.")
    valid = per_date_ic.loc[per_date_ic["status"].eq(VALID_IC_STATUS)]
    rows = []
    for feature in features:
        values = valid.loc[valid["feature"].eq(feature), "ic"].astype(float)
        mean = float(values.mean()) if len(values) else np.nan
        rows.append(
            {
                "feature": feature,
                "train_start": eligible_dates[0],
                "train_end": eligible_dates[-1],
                "training_dates": len(eligible_dates),
                "valid_dates": int(len(values)),
                "mean_ic": mean,
                "absolute_mean_ic": abs(mean) if np.isfinite(mean) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def select_production_top250(summary: pd.DataFrame, features: list[str]) -> dict:
    if len(features) != 350 or summary["feature"].tolist() != features:
        raise ValueError("Production selection requires exact ordered Medium features.")
    frame = summary.copy()
    frame["eligible"] = (
        frame["valid_dates"].gt(0)
        & np.isfinite(frame["mean_ic"])
        & np.isfinite(frame["absolute_mean_ic"])
    )
    if not np.allclose(
        frame.loc[frame["eligible"], "absolute_mean_ic"],
        frame.loc[frame["eligible"], "mean_ic"].abs(),
        atol=1e-15,
        rtol=0,
    ):
        raise ValueError("Production IC statistics are inconsistent.")
    ranked = frame.sort_values(
        ["eligible", "absolute_mean_ic", "feature"],
        ascending=[False, False, True],
        na_position="last",
        kind="stable",
    ).head(SELECTED_FEATURE_COUNT)
    if not ranked["eligible"].all():
        raise ValueError("Fewer than 250 production features have valid IC.")
    ranked_names = ranked["feature"].tolist()
    selected = set(ranked_names)
    model_features = [feature for feature in features if feature in selected]
    return {
        "policy_id": POLICY_ID,
        "selected_feature_count": 250,
        "source_feature_count": 350,
        "train_start": str(summary.iloc[0]["train_start"]),
        "train_end": str(summary.iloc[0]["train_end"]),
        "training_dates": int(summary.iloc[0]["training_dates"]),
        "selected_ranked": [
            {
                "selection_rank": rank,
                "feature": row.feature,
                "mean_ic": float(row.mean_ic),
                "absolute_mean_ic": float(row.absolute_mean_ic),
            }
            for rank, row in enumerate(ranked.itertuples(), 1)
        ],
        "model_feature_names": model_features,
        "source_feature_list_sha256": feature_list_sha256(features),
        "selected_model_features_sha256": feature_list_sha256(model_features),
    }


def _load_training(snapshot_dir: Path, features: list[str], cutoff: str) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    columns = ["date", TARGET, *features]
    train = pd.read_parquet(snapshot_dir / "train.parquet", columns=columns)
    validation = pd.read_parquet(
        snapshot_dir / "validation.parquet",
        columns=columns,
        filters=[("date", "<=", cutoff)],
    )
    combined = pd.concat([train, validation], ignore_index=True)
    dates = sorted(combined["date"].astype(str).unique())
    if dates[-1] != cutoff or combined[TARGET].isna().any() or not np.isfinite(combined[TARGET]).all():
        raise ValueError("Production training frame violates cutoff or maturity.")
    target = combined.pop(TARGET)
    combined.pop("date")
    return combined, target, dates


def _fit_component(
    features: pd.DataFrame,
    target: pd.Series,
    feature_names: list[str],
    path: Path,
) -> dict:
    started = time.perf_counter()
    matrix = features.loc[:, feature_names]
    model = train_validated_lightgbm(
        create_lightgbm_model(DEFAULT_LIGHTGBM_PARAMS), matrix, target, feature_names
    )
    model.booster_.save_model(str(path))
    reloaded = lgb.Booster(model_file=str(path))
    sample = matrix.iloc[: min(10_000, len(matrix))]
    original_predictions = model.predict(sample)
    reloaded_predictions = reloaded.predict(sample)
    difference = float(np.max(np.abs(original_predictions - reloaded_predictions)))
    if difference > 1e-12:
        raise ValueError("Reloaded production model predictions changed.")
    result = {
        "feature_count": len(feature_names),
        "training_rows": len(features),
        "training_seconds": time.perf_counter() - started,
        "trees": int(model.booster_.num_trees()),
        "model_bytes": path.stat().st_size,
        "model_sha256": sha256_file(path),
        "reload_check_rows": len(sample),
        "reload_max_absolute_prediction_difference": difference,
    }
    del model, reloaded, original_predictions, reloaded_predictions, sample, matrix
    gc.collect()
    return result


def train_production(
    snapshot_dir: Path,
    model_version: str,
    *,
    output_root: Path = OUTPUT_ROOT,
) -> Path:
    if model_version not in ALLOWED_MODEL_VERSIONS:
        raise ValueError(f"Only predeclared model versions are allowed: {ALLOWED_MODEL_VERSIONS}")
    destination = output_root / "v6.0" / "production" / "models" / model_version
    if destination.exists():
        raise FileExistsError(f"Production model version already exists: {destination}")
    destination.mkdir(parents=True)
    write_json_once(destination / "TRAINING_STARTED.json", {"started_at_utc": utc_now()})
    progress_path = destination / "progress.jsonl"
    append_progress(progress_path, "training_started", model_version=model_version)
    started = time.perf_counter()
    try:
        snapshot_manifest = verify_snapshot(snapshot_dir)
        append_progress(progress_path, "snapshot_verified", snapshot_id=snapshot_manifest["snapshot_id"])
        policy = load_v1_policy()
        if policy["policy_id"] != V1_POLICY_ID or sha256_file(V1_POLICY_PATH) != "4ccddd457707232feb7391389b4a387e57dcd1871af898aeefd82a1f8fa7381e":
            raise ValueError("Frozen V1 policy identity changed.")
        prediction_date = str(snapshot_manifest["public_metadata_after"]["round"]["data_datestamp"])
        prediction_date = f"{prediction_date[:4]}-{prediction_date[4:6]}-{prediction_date[6:]}"
        local = audit_local_dataset(snapshot_dir, prediction_date)
        eligible_dates, eligibility = _eligible_dates(snapshot_dir, prediction_date)
        features = resolve_v6_numeric_feature_set("medium", snapshot_dir)
        if len(features) != 350:
            raise ValueError("Frozen production Medium must contain 350 features.")
        plan = choose_ic_plan(local["files"])
        append_progress(progress_path, "ic_plan_verified", plan=plan, eligible_dates=len(eligible_dates))
        ic_started = time.perf_counter()
        if plan == "reuse_and_extend":
            if sha256_file(FINAL_IC_PATH) != FINAL_IC_SHA256:
                raise ValueError("Reusable final IC artifact hash changed.")
            reused = pd.read_parquet(FINAL_IC_PATH)
            reused_dates = sorted(reused["date"].astype(str).unique())
            if len(reused_dates) != 1165 or reused_dates[-1] != "2025-04-25":
                raise ValueError("Reusable final IC coverage changed.")
            missing_dates = [date for date in eligible_dates if date not in set(reused_dates)]
            extension = _calculate_ic_dates(
                snapshot_dir, missing_dates, features, progress_path=progress_path
            )
            combined_ic = pd.concat([reused, extension], ignore_index=True)
        else:
            reused_dates = []
            missing_dates = eligible_dates
            extension = _calculate_ic_dates(
                snapshot_dir, eligible_dates, features, progress_path=progress_path
            )
            combined_ic = extension.copy()
        combined_ic = combined_ic.sort_values(["date", "feature"], kind="stable")
        summary = summarize_production_ic(combined_ic, features, eligible_dates)
        manifest = select_production_top250(summary, features)
        append_progress(progress_path, "feature_selection_completed", selected_features=250)
        holdout_manifest = json.loads(
            (OUTPUT_ROOT / "v6.0" / "final_holdout" / "v6_v1_final_001" / "preparation" / "feature_manifest.json").read_text()
        )
        prior = set(holdout_manifest["model_feature_names"])
        current = set(manifest["model_feature_names"])
        manifest["final_holdout_overlap"] = {
            "shared": len(prior & current),
            "added": sorted(current - prior),
            "removed": sorted(prior - current),
            "jaccard": len(prior & current) / len(prior | current),
            "descriptive_only": True,
        }
        ic_dir = destination / "feature_quality"
        model_dir = destination / "models"
        ic_dir.mkdir()
        model_dir.mkdir()
        extension.to_parquet(ic_dir / "ic_extension.parquet", index=False)
        combined_ic.to_parquet(ic_dir / "combined_per_date_ic.parquet", index=False)
        summary.to_parquet(ic_dir / "ic_summary.parquet", index=False)
        ic_provenance = {
            "plan": plan,
            "source_final_ic_sha256": FINAL_IC_SHA256 if plan == "reuse_and_extend" else None,
            "reused_dates": len(reused_dates),
            "computed_dates": len(missing_dates),
            "combined_dates": len(eligible_dates),
            "feature_date_observations": len(combined_ic),
            "runtime_seconds": time.perf_counter() - ic_started,
            "snapshot_hashes": {name: item["sha256"] for name, item in local["files"].items()},
            "target": TARGET,
            "definition": "Per-date cross-sectional Spearman correlation: Pearson correlation of pairwise-complete average ranks.",
        }
        write_json_once(ic_dir / "provenance.json", ic_provenance)
        write_json_once(destination / "selected_features.json", manifest)
        shutil.copy2(snapshot_dir / "snapshot_manifest.json", destination / "source_snapshot_manifest.json")
        features_frame, target, training_dates = _load_training(
            snapshot_dir, features, eligibility["eligible_training_cutoff"]
        )
        if len(features_frame) != eligibility["eligible_training_rows"] or training_dates != eligible_dates:
            raise ValueError("Loaded training coverage differs from audited eligibility.")
        append_progress(progress_path, "medium_model_training_started", training_rows=len(features_frame))
        medium_diagnostics = _fit_component(
            features_frame, target, features, model_dir / "medium.txt"
        )
        append_progress(progress_path, "medium_model_reload_verified", **medium_diagnostics)
        append_progress(progress_path, "top250_model_training_started", training_rows=len(features_frame))
        top250_diagnostics = _fit_component(
            features_frame,
            target,
            manifest["model_feature_names"],
            model_dir / "top250.txt",
        )
        append_progress(progress_path, "top250_model_reload_verified", **top250_diagnostics)
        del features_frame, target
        gc.collect()
        configuration = {
            "model_version": model_version,
            "dataset_version": "v6.0",
            "target": TARGET,
            "frozen_policy_id": V1_POLICY_ID,
            "frozen_policy_sha256": sha256_file(V1_POLICY_PATH),
            "snapshot_id": snapshot_manifest["snapshot_id"],
            "production_as_of_timestamp": snapshot_manifest["completed_at_utc"],
            "prediction_date": prediction_date,
            "eligibility": eligibility,
            "lightgbm_parameters": DEFAULT_LIGHTGBM_PARAMS,
            "ensemble": {"weights": {"medium": 0.5, "top250": 0.5}, "normalization": "per-date average rank normalized as (rank - 0.5) / n"},
        }
        write_json_once(destination / "production_config.json", configuration)
        diagnostics = {
            "medium": medium_diagnostics,
            "top250": top250_diagnostics,
            "total_runtime_seconds_before_completion": time.perf_counter() - started,
        }
        write_json_once(destination / "training_diagnostics.json", diagnostics)
        append_progress(progress_path, "artifact_persistence_started")
        artifact_names = [
            "TRAINING_STARTED.json", "production_config.json", "source_snapshot_manifest.json",
            "selected_features.json", "training_diagnostics.json", "models/medium.txt",
            "models/top250.txt", "feature_quality/ic_extension.parquet",
            "feature_quality/combined_per_date_ic.parquet", "feature_quality/ic_summary.parquet",
            "feature_quality/provenance.json", "progress.jsonl",
        ]
        hashes = {name: sha256_file(destination / name) for name in artifact_names}
        write_json_once(destination / "artifact_hashes.json", hashes)
        write_json_once(
            destination / "PRODUCTION_TRAINING_COMPLETED.json",
            {"state": "complete", "completed_at_utc": utc_now(), "artifact_sha256": hashes},
        )
        verify_production_artifacts(destination)
        return destination
    except BaseException as error:
        append_progress(
            progress_path,
            "training_failed",
            error_type=type(error).__name__,
            error=str(error),
        )
        write_json_once(
            destination / "TRAINING_FAILED.json",
            {"failed_at_utc": utc_now(), "error_type": type(error).__name__, "error": str(error)},
        )
        raise


def verify_production_artifacts(destination: Path) -> dict:
    marker_path = destination / "PRODUCTION_TRAINING_COMPLETED.json"
    if not marker_path.is_file():
        raise ValueError("Production training is incomplete.")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    for name, digest in marker["artifact_sha256"].items():
        if sha256_file(destination / name) != digest:
            raise ValueError(f"Production artifact hash changed: {name}")
    config = json.loads((destination / "production_config.json").read_text())
    if config["frozen_policy_sha256"] != sha256_file(V1_POLICY_PATH):
        raise ValueError("Production artifact policy identity changed.")
    for name in ("medium.txt", "top250.txt"):
        model = lgb.Booster(model_file=str(destination / "models" / name))
        if model.num_trees() != DEFAULT_LIGHTGBM_PARAMS["n_estimators"]:
            raise ValueError(f"Unexpected production model tree count: {name}")
    return {"config": config, "completion": marker}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    refresh = subparsers.add_parser("refresh")
    refresh.add_argument("--snapshot-id", required=True)
    refresh.add_argument("--round-number", required=True, type=int)
    refresh.add_argument("--run", action="store_true")
    refresh.add_argument("--confirm-snapshot-refresh", action="store_true")
    train = subparsers.add_parser("train")
    train.add_argument("--snapshot-id", required=True)
    train.add_argument("--model-version", required=True)
    train.add_argument("--run", action="store_true")
    train.add_argument("--confirm-production-training", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "refresh":
        if not args.run or not args.confirm_snapshot_refresh:
            raise PermissionError("Snapshot refresh requires both explicit confirmation flags.")
        print(refresh_snapshot(args.snapshot_id, args.round_number))
        return
    if not args.run or not args.confirm_production_training:
        raise PermissionError("Production training requires both explicit confirmation flags.")
    snapshot_dir = DATA_ROOT / "v6.0" / "snapshots" / args.snapshot_id
    print(train_production(snapshot_dir, args.model_version))


if __name__ == "__main__":
    main()
