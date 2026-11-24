"""Download an official v6 live snapshot and run frozen V1 inference."""

import argparse
import importlib.metadata
import json
import platform
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from numerapi import NumerAPI

from src.audit_production_data import fetch_public_metadata, sha256_file
from src.paths import DATA_ROOT, OUTPUT_ROOT
from src.production_training import verify_production_artifacts, write_json_once


LIVE_DATASET_PATH = "v6.0/live.parquet"
FROZEN_WEIGHTS = {"medium": 0.5, "top250": 0.5}
MODEL_VERSION = "v6_v1_production_002"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class LiveModelArtifacts:
    medium_model: object
    top250_model: object
    medium_features: tuple[str, ...]
    top250_features: tuple[str, ...]
    model_version: str
    model_hashes: dict[str, str]


def _round_identity(metadata: dict) -> dict:
    return {
        key: metadata["round"][key]
        for key in ("number", "target", "data_datestamp", "open_time", "close_time")
    }


def current_public_metadata(api_factory=NumerAPI) -> dict:
    round_number = api_factory().get_current_round()
    return fetch_public_metadata(round_number)


def download_live_snapshot(
    snapshot_id: str,
    *,
    data_root: Path = DATA_ROOT,
    api_factory=NumerAPI,
    metadata_fetcher=current_public_metadata,
) -> Path:
    before = metadata_fetcher()
    round_number = int(before["round"]["number"])
    destination = data_root / "v6.0" / "live_snapshots" / f"round_{round_number}" / snapshot_id
    if destination.exists():
        raise FileExistsError(f"Live snapshot already exists: {destination}")
    destination.mkdir(parents=True)
    write_json_once(
        destination / "DOWNLOAD_STARTED.json",
        {"started_at_utc": utc_now(), "snapshot_id": snapshot_id, "round": round_number},
    )
    try:
        api = api_factory()
        if LIVE_DATASET_PATH not in set(api.list_datasets()):
            raise ValueError(f"Official live dataset is unavailable: {LIVE_DATASET_PATH}")
        live_path = destination / "live.parquet"
        download_started = utc_now()
        api.download_dataset(LIVE_DATASET_PATH, str(live_path))
        if not live_path.is_file() or live_path.stat().st_size == 0:
            raise ValueError("Official live download is incomplete.")
        after = metadata_fetcher()
        if _round_identity(before) != _round_identity(after):
            raise ValueError("Public round identity changed during live download.")
        parquet = pq.ParquetFile(live_path)
        manifest = {
            "state": "complete",
            "snapshot_id": snapshot_id,
            "official_dataset_path": LIVE_DATASET_PATH,
            "download_started_at_utc": download_started,
            "download_completed_at_utc": utc_now(),
            "public_metadata_before": before,
            "public_metadata_after": after,
            "file": {
                "relative_path": "live.parquet",
                "size_bytes": live_path.stat().st_size,
                "sha256": sha256_file(live_path),
                "rows": parquet.metadata.num_rows,
                "columns": parquet.metadata.num_columns,
                "row_groups": parquet.metadata.num_row_groups,
            },
        }
        write_json_once(destination / "live_snapshot_manifest.json", manifest)
        write_json_once(
            destination / "LIVE_SNAPSHOT_COMPLETED.json",
            {
                "state": "complete",
                "manifest_sha256": sha256_file(destination / "live_snapshot_manifest.json"),
            },
        )
        return destination
    except BaseException as error:
        write_json_once(
            destination / "DOWNLOAD_FAILED.json",
            {"failed_at_utc": utc_now(), "error_type": type(error).__name__, "error": str(error)},
        )
        raise


def verify_live_snapshot(snapshot_dir: Path) -> dict:
    manifest_path = snapshot_dir / "live_snapshot_manifest.json"
    marker_path = snapshot_dir / "LIVE_SNAPSHOT_COMPLETED.json"
    if not manifest_path.is_file() or not marker_path.is_file():
        raise ValueError("Live snapshot is incomplete.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("Live snapshot manifest hash changed.")
    live_path = snapshot_dir / manifest["file"]["relative_path"]
    if sha256_file(live_path) != manifest["file"]["sha256"]:
        raise ValueError("Live dataset hash changed.")
    if _round_identity(manifest["public_metadata_before"]) != _round_identity(
        manifest["public_metadata_after"]
    ):
        raise ValueError("Live snapshot round identity is inconsistent.")
    return manifest


def load_model_artifacts(model_dir: Path) -> LiveModelArtifacts:
    verified = verify_production_artifacts(model_dir)
    config = verified["config"]
    if config["model_version"] != MODEL_VERSION:
        raise ValueError(f"Unsupported production model version: {config['model_version']}")
    source = json.loads((model_dir / "source_snapshot_manifest.json").read_text(encoding="utf-8"))
    selected = json.loads((model_dir / "selected_features.json").read_text(encoding="utf-8"))
    medium = tuple(source["local_audit"]["schema"]["medium_features"])
    top250 = tuple(selected["model_feature_names"])
    if len(medium) != 350 or len(top250) != 250 or not set(top250).issubset(medium):
        raise ValueError("Production feature manifests are incompatible.")
    hashes = json.loads((model_dir / "artifact_hashes.json").read_text(encoding="utf-8"))
    return LiveModelArtifacts(
        medium_model=lgb.Booster(model_file=str(model_dir / "models" / "medium.txt")),
        top250_model=lgb.Booster(model_file=str(model_dir / "models" / "top250.txt")),
        medium_features=medium,
        top250_features=top250,
        model_version=config["model_version"],
        model_hashes={
            "medium": hashes["models/medium.txt"],
            "top250": hashes["models/top250.txt"],
        },
    )


def rank_normalize(values: pd.Series, groups: pd.Series | None = None) -> pd.Series:
    if not pd.api.types.is_numeric_dtype(values) or not np.isfinite(
        values.to_numpy(dtype=np.float64)
    ).all():
        raise ValueError("Predictions must be numeric and finite before ranking.")
    if groups is None:
        ranks = values.rank(method="average")
        return (ranks - 0.5) / len(values)
    ranks = values.groupby(groups, sort=False).rank(method="average")
    counts = values.groupby(groups, sort=False).transform("size")
    return (ranks - 0.5) / counts


def predict_live(frame: pd.DataFrame, artifacts: LiveModelArtifacts) -> dict[str, pd.DataFrame]:
    """Pure, network-independent frozen V1 prediction core."""
    if "numerai_ticker" not in frame.columns:
        raise ValueError("Live data is missing numerai_ticker.")
    identity_columns = ["numerai_ticker"] + (["date"] if "date" in frame.columns else [])
    if frame[identity_columns].isna().any(axis=None) or frame.duplicated(identity_columns).any():
        raise ValueError("Live identities are missing or duplicated.")
    missing = sorted(set(artifacts.medium_features) - set(frame.columns))
    if missing:
        raise ValueError(f"Live data is missing required Medium features: {missing[:5]}")
    for feature in artifacts.medium_features:
        if not pd.api.types.is_numeric_dtype(frame[feature]):
            raise ValueError(f"Live feature must be numeric: {feature}")
    medium_raw = np.asarray(
        artifacts.medium_model.predict(frame.loc[:, list(artifacts.medium_features)]), dtype=np.float64
    )
    top250_raw = np.asarray(
        artifacts.top250_model.predict(frame.loc[:, list(artifacts.top250_features)]), dtype=np.float64
    )
    if len(medium_raw) != len(frame) or len(top250_raw) != len(frame):
        raise ValueError("Model output length differs from live input.")
    groups = frame["date"] if "date" in frame.columns else None
    outputs = {}
    normalized = {}
    for name, raw in (("medium", medium_raw), ("top250", top250_raw)):
        if not np.isfinite(raw).all():
            raise ValueError(f"{name} produced missing or non-finite predictions.")
        result = frame.loc[:, identity_columns].copy()
        result["raw_prediction"] = raw
        result["prediction"] = rank_normalize(pd.Series(raw, index=frame.index), groups).to_numpy()
        outputs[name] = result
        normalized[name] = result["prediction"].to_numpy(dtype=np.float64)
    ensemble = frame.loc[:, identity_columns].copy()
    ensemble["prediction"] = (
        FROZEN_WEIGHTS["medium"] * normalized["medium"]
        + FROZEN_WEIGHTS["top250"] * normalized["top250"]
    )
    if not np.isfinite(ensemble["prediction"]).all() or not ensemble["prediction"].between(
        0.0, 1.0, inclusive="both"
    ).all():
        raise ValueError("Frozen ensemble predictions are outside the accepted range.")
    outputs["ensemble"] = ensemble
    return outputs


def audit_live_frame(
    frame: pd.DataFrame,
    artifacts: LiveModelArtifacts,
    *,
    available_columns: list[str] | None = None,
    physical_types: dict[str, str] | None = None,
) -> dict:
    required = list(artifacts.medium_features)
    available_columns = available_columns or frame.columns.tolist()
    missing_counts = frame[required].isna().sum()
    return {
        "rows": len(frame),
        "columns": len(frame.columns),
        "identity_column": "numerai_ticker",
        "unique_tickers": int(frame["numerai_ticker"].nunique()),
        "duplicate_identities": int(frame.duplicated(["numerai_ticker"]).sum()),
        "date_values": sorted(frame["date"].astype(str).unique()) if "date" in frame else [],
        "available_feature_columns": sum(
            column.startswith("feature_") for column in available_columns
        ),
        "available_feature_physical_types": physical_types or {},
        "medium_required": len(artifacts.medium_features),
        "top250_required": len(artifacts.top250_features),
        "medium_missing_columns": sorted(set(artifacts.medium_features) - set(frame.columns)),
        "top250_missing_columns": sorted(set(artifacts.top250_features) - set(frame.columns)),
        "required_feature_dtypes": {
            str(dtype): int(sum(frame[column].dtype == dtype for column in required))
            for dtype in sorted({frame[column].dtype for column in required}, key=str)
        },
        "required_missing_values": int(missing_counts.sum()),
        "required_features_with_missing": int(missing_counts.gt(0).sum()),
        "maximum_required_feature_missing": int(missing_counts.max()),
    }


def _distribution(values: pd.Series) -> dict:
    return {
        "min": float(values.min()),
        "p10": float(values.quantile(0.1)),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "p90": float(values.quantile(0.9)),
        "max": float(values.max()),
        "std": float(values.std(ddof=0)),
    }


def persist_live_predictions(
    outputs: dict[str, pd.DataFrame],
    live_manifest: dict,
    model_dir: Path,
    model_artifacts: LiveModelArtifacts,
    *,
    output_root: Path = OUTPUT_ROOT,
    live_audit: dict | None = None,
    inference_seconds: float | None = None,
) -> Path:
    round_number = int(live_manifest["public_metadata_after"]["round"]["number"])
    destination = (
        output_root
        / "v6.0"
        / "production"
        / "predictions"
        / f"round_{round_number}"
        / model_artifacts.model_version
    )
    if destination.exists():
        raise FileExistsError(f"Live prediction artifacts already exist: {destination}")
    destination.mkdir(parents=True)
    started = time.perf_counter()
    for name in ("medium", "top250", "ensemble"):
        outputs[name].to_parquet(destination / f"{name}_predictions.parquet", index=False)
    submission = outputs["ensemble"].loc[:, ["numerai_ticker", "prediction"]]
    if submission["numerai_ticker"].duplicated().any():
        raise ValueError("Submission contains duplicate numerai_ticker values.")
    submission.to_csv(destination / "submission.csv", index=False)
    round_trip = pd.read_csv(destination / "submission.csv", dtype={"numerai_ticker": str})
    if round_trip.columns.tolist() != ["numerai_ticker", "prediction"]:
        raise ValueError("Submission CSV schema changed on read-back.")
    if not np.array_equal(
        round_trip["numerai_ticker"].to_numpy(dtype=str),
        submission["numerai_ticker"].to_numpy(dtype=str),
    ):
        raise ValueError("Submission ticker identity changed on read-back.")
    if not np.allclose(
        round_trip["prediction"].to_numpy(dtype=np.float64),
        submission["prediction"].to_numpy(dtype=np.float64),
        atol=1e-15,
        rtol=0,
    ):
        raise ValueError("Submission predictions changed on read-back.")
    diagnostics = {
        "rows": len(submission),
        "component_distributions": {
            name: _distribution(outputs[name]["prediction"])
            for name in ("medium", "top250", "ensemble")
        },
        "medium_top250_pearson": float(
            outputs["medium"]["prediction"].corr(outputs["top250"]["prediction"], method="pearson")
        ),
        "medium_top250_spearman": float(
            outputs["medium"]["prediction"].corr(outputs["top250"]["prediction"], method="spearman")
        ),
        "deterministic_formula": "0.5 * medium_average_rank + 0.5 * top250_average_rank",
    }
    runtime = {"artifact_persistence_seconds": time.perf_counter() - started}
    versions = {}
    for package in ("lightgbm", "numpy", "pandas", "cloudpickle"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    metadata = {
        "state": "complete",
        "round": round_number,
        "data_datestamp": live_manifest["public_metadata_after"]["round"]["data_datestamp"],
        "official_dataset_path": LIVE_DATASET_PATH,
        "live_dataset_sha256": live_manifest["file"]["sha256"],
        "live_snapshot_id": live_manifest["snapshot_id"],
        "model_version": model_artifacts.model_version,
        "model_completion_sha256": sha256_file(model_dir / "PRODUCTION_TRAINING_COMPLETED.json"),
        "model_hashes": model_artifacts.model_hashes,
        "weights": FROZEN_WEIGHTS,
        "normalization": "per-cross-section average rank normalized as (rank - 0.5) / n",
        "runtime": runtime,
        "environment": {"python": platform.python_version(), **versions},
    }
    write_json_once(destination / "round_metadata.json", metadata)
    write_json_once(destination / "prediction_diagnostics.json", diagnostics)
    write_json_once(destination / "live_data_audit.json", live_audit or {})
    write_json_once(
        destination / "execution_runtime.json",
        {
            "prediction_seconds_before_persistence": inference_seconds,
            "artifact_persistence_seconds": runtime["artifact_persistence_seconds"],
        },
    )
    names = [
        "medium_predictions.parquet",
        "top250_predictions.parquet",
        "ensemble_predictions.parquet",
        "submission.csv",
        "round_metadata.json",
        "prediction_diagnostics.json",
        "live_data_audit.json",
        "execution_runtime.json",
    ]
    hashes = {name: sha256_file(destination / name) for name in names}
    write_json_once(destination / "artifact_hashes.json", hashes)
    write_json_once(
        destination / "LIVE_INFERENCE_COMPLETED.json",
        {"state": "complete", "completed_at_utc": utc_now(), "artifact_sha256": hashes},
    )
    return destination


def verify_prediction_artifacts(destination: Path) -> dict:
    marker_path = destination / "LIVE_INFERENCE_COMPLETED.json"
    if not marker_path.is_file():
        raise ValueError("Live inference artifacts are incomplete.")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    for name, expected in marker["artifact_sha256"].items():
        if sha256_file(destination / name) != expected:
            raise ValueError(f"Live prediction artifact hash changed: {name}")
    return marker


def finalize_incomplete_live_inference(
    live_snapshot: Path,
    model_dir: Path,
    destination: Path,
) -> Path:
    """Seal already-written predictions after a persistence-only failure."""
    if (destination / "LIVE_INFERENCE_COMPLETED.json").exists():
        raise FileExistsError("Live inference is already complete.")
    required_files = {
        "medium": destination / "medium_predictions.parquet",
        "top250": destination / "top250_predictions.parquet",
        "ensemble": destination / "ensemble_predictions.parquet",
    }
    if any(not path.is_file() for path in required_files.values()):
        raise ValueError("Incomplete live inference lacks saved prediction files.")
    live_manifest = verify_live_snapshot(live_snapshot)
    artifacts = load_model_artifacts(model_dir)
    outputs = {name: pd.read_parquet(path) for name, path in required_files.items()}
    identity = outputs["ensemble"][["numerai_ticker", "date"]]
    for name, output in outputs.items():
        if output.duplicated(["numerai_ticker", "date"]).any():
            raise ValueError(f"Saved {name} predictions contain duplicate identities.")
        if not output[["numerai_ticker", "date"]].equals(identity):
            raise ValueError(f"Saved {name} prediction identities do not align.")
        if not np.isfinite(output["prediction"].to_numpy(dtype=np.float64)).all():
            raise ValueError(f"Saved {name} predictions are non-finite.")
    expected = (
        FROZEN_WEIGHTS["medium"] * outputs["medium"]["prediction"].to_numpy()
        + FROZEN_WEIGHTS["top250"] * outputs["top250"]["prediction"].to_numpy()
    )
    if not np.allclose(outputs["ensemble"]["prediction"], expected, atol=1e-15, rtol=0):
        raise ValueError("Saved ensemble does not match frozen weights.")
    submission = pd.read_csv(destination / "submission.csv", dtype={"numerai_ticker": str})
    if not np.array_equal(
        submission["numerai_ticker"].to_numpy(dtype=str),
        identity["numerai_ticker"].to_numpy(dtype=str),
    ) or not np.allclose(
        submission["prediction"].to_numpy(dtype=np.float64),
        outputs["ensemble"]["prediction"].to_numpy(dtype=np.float64),
        atol=1e-15,
        rtol=0,
    ):
        raise ValueError("Saved submission does not round-trip to the frozen ensemble.")
    schema = pq.read_schema(live_snapshot / "live.parquet")
    live = pd.read_parquet(
        live_snapshot / "live.parquet",
        columns=["numerai_ticker", "date", *artifacts.medium_features],
    )
    feature_types = {}
    for field in schema:
        if field.name.startswith("feature_"):
            feature_types[str(field.type)] = feature_types.get(str(field.type), 0) + 1
    audit = audit_live_frame(
        live,
        artifacts,
        available_columns=schema.names,
        physical_types=feature_types,
    )
    diagnostics = {
        "rows": len(identity),
        "component_distributions": {
            name: _distribution(outputs[name]["prediction"])
            for name in ("medium", "top250", "ensemble")
        },
        "medium_top250_pearson": float(
            outputs["medium"]["prediction"].corr(outputs["top250"]["prediction"], method="pearson")
        ),
        "medium_top250_spearman": float(
            outputs["medium"]["prediction"].corr(outputs["top250"]["prediction"], method="spearman")
        ),
        "deterministic_formula": "0.5 * medium_average_rank + 0.5 * top250_average_rank",
    }
    versions = {}
    for package in ("lightgbm", "numpy", "pandas", "cloudpickle"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    metadata = {
        "state": "complete",
        "round": int(live_manifest["public_metadata_after"]["round"]["number"]),
        "data_datestamp": live_manifest["public_metadata_after"]["round"]["data_datestamp"],
        "official_dataset_path": LIVE_DATASET_PATH,
        "live_dataset_sha256": live_manifest["file"]["sha256"],
        "live_snapshot_id": live_manifest["snapshot_id"],
        "model_version": artifacts.model_version,
        "model_completion_sha256": sha256_file(model_dir / "PRODUCTION_TRAINING_COMPLETED.json"),
        "model_hashes": artifacts.model_hashes,
        "weights": FROZEN_WEIGHTS,
        "normalization": "per-cross-section average rank normalized as (rank - 0.5) / n",
        "environment": {"python": platform.python_version(), **versions},
    }
    write_json_once(destination / "round_metadata.json", metadata)
    write_json_once(destination / "prediction_diagnostics.json", diagnostics)
    write_json_once(destination / "live_data_audit.json", audit)
    write_json_once(
        destination / "execution_runtime.json",
        {"initial_execution_status": "predictions_saved_before_csv_dtype_validation_failure"},
    )
    write_json_once(
        destination / "PERSISTENCE_RECOVERY.json",
        {
            "recovered_at_utc": utc_now(),
            "reason": "CSV read-back ticker dtype differed while values remained identical",
            "predictions_regenerated": False,
        },
    )
    names = [
        "medium_predictions.parquet", "top250_predictions.parquet",
        "ensemble_predictions.parquet", "submission.csv", "round_metadata.json",
        "prediction_diagnostics.json", "live_data_audit.json", "execution_runtime.json",
        "PERSISTENCE_RECOVERY.json",
    ]
    hashes = {name: sha256_file(destination / name) for name in names}
    write_json_once(destination / "artifact_hashes.json", hashes)
    write_json_once(
        destination / "LIVE_INFERENCE_COMPLETED.json",
        {"state": "complete", "completed_at_utc": utc_now(), "artifact_sha256": hashes},
    )
    verify_prediction_artifacts(destination)
    return destination


def run_live_inference(live_snapshot: Path, model_dir: Path) -> Path:
    started = time.perf_counter()
    live_manifest = verify_live_snapshot(live_snapshot)
    artifacts = load_model_artifacts(model_dir)
    schema = pq.read_schema(live_snapshot / "live.parquet")
    identity = ["numerai_ticker"] + (["date"] if "date" in schema.names else [])
    live = pd.read_parquet(
        live_snapshot / "live.parquet",
        columns=[*identity, *artifacts.medium_features],
    )
    feature_types = {}
    for field in schema:
        if field.name.startswith("feature_"):
            feature_types[str(field.type)] = feature_types.get(str(field.type), 0) + 1
    audit = audit_live_frame(
        live,
        artifacts,
        available_columns=schema.names,
        physical_types=feature_types,
    )
    if audit["duplicate_identities"] or audit["medium_missing_columns"]:
        raise ValueError("Official live data is incompatible with production models.")
    first = predict_live(live, artifacts)
    shuffled = live.sample(frac=1.0, random_state=42)
    second = predict_live(shuffled, artifacts)
    for name in first:
        left = first[name].sort_values("numerai_ticker").reset_index(drop=True)
        right = second[name].sort_values("numerai_ticker").reset_index(drop=True)
        columns = [column for column in left.columns if column != "date"]
        if not left[columns].equals(right[columns]):
            raise ValueError(f"Identity-aligned predictions changed after shuffling: {name}")
    destination = persist_live_predictions(
        first,
        live_manifest,
        model_dir,
        artifacts,
        live_audit=audit,
        inference_seconds=time.perf_counter() - started,
    )
    verify_prediction_artifacts(destination)
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    download = subparsers.add_parser("download")
    download.add_argument("--snapshot-id", required=True)
    download.add_argument("--run", action="store_true")
    download.add_argument("--confirm-live-download", action="store_true")
    predict = subparsers.add_parser("predict")
    predict.add_argument("--live-snapshot", required=True, type=Path)
    predict.add_argument("--model-version", required=True)
    predict.add_argument("--run", action="store_true")
    predict.add_argument("--confirm-live-inference", action="store_true")
    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--live-snapshot", required=True, type=Path)
    finalize.add_argument("--model-version", required=True)
    finalize.add_argument("--run", action="store_true")
    finalize.add_argument("--confirm-persistence-recovery", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "download":
        if not args.run or not args.confirm_live_download:
            raise SystemExit("Live download requires --run and --confirm-live-download.")
        print(download_live_snapshot(args.snapshot_id))
        return
    if args.command == "finalize":
        if not args.run or not args.confirm_persistence_recovery:
            raise SystemExit("Persistence recovery requires explicit confirmation.")
        model_dir = OUTPUT_ROOT / "v6.0" / "production" / "models" / args.model_version
        manifest = verify_live_snapshot(args.live_snapshot)
        round_number = manifest["public_metadata_after"]["round"]["number"]
        destination = OUTPUT_ROOT / "v6.0" / "production" / "predictions" / f"round_{round_number}" / args.model_version
        print(finalize_incomplete_live_inference(args.live_snapshot, model_dir, destination))
        return
    if not args.run or not args.confirm_live_inference:
        raise SystemExit("Live inference requires --run and --confirm-live-inference.")
    if args.model_version != MODEL_VERSION:
        raise SystemExit(f"Only model version {MODEL_VERSION} is supported.")
    model_dir = OUTPUT_ROOT / "v6.0" / "production" / "models" / args.model_version
    print(run_live_inference(args.live_snapshot, model_dir))


if __name__ == "__main__":
    main()
