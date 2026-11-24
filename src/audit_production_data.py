"""Audit v6 production-training eligibility without training or live inference."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from numerapi import NumerAPI

from src.config import DEFAULT_LIGHTGBM_PARAMS, RANDOM_STATE
from src.feature_selection import POLICY_ID, SELECTED_FEATURE_COUNT
from src.paths import DATA_ROOT, OUTPUT_ROOT, validate_dataset_version
from src.v1_policy import (
    MEDIUM_FEATURE_LIST_SHA256,
    V1_POLICY_ID,
    V1_POLICY_PATH,
    load_v1_policy,
)


TARGET = "target_jupiter_60"
PURGE_DATES = 16
OFFICIAL_SOURCES = {
    "data": "https://docs.numer.ai/numerai-tournament/data",
    "submissions": "https://docs.numer.ai/numerai-tournament/submissions",
    "model_uploads": "https://docs.numer.ai/numerai-tournament/submissions/model-uploads",
    "api_keys": "https://docs.numer.ai/numerai-tournament/api",
}
RESEARCH_SNAPSHOT_SHA256 = {
    "features.json": "ee6c00bc9d5cb78aaab986630138512945301a3c588865ad61da5d0dea5ff634",
    "train.parquet": "ac81951d1ff5925475e015d83d1252483e9c84d60f6963b2fdae7a197210bca0",
    "validation.parquet": "2f4ee8b9a0a9767286663316d1a89b477d03160f6720d472cd132532a0d52a81",
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_date_targets(frames: list[pd.DataFrame]) -> pd.DataFrame:
    required = {"date", TARGET}
    if not frames or any(not required.issubset(frame.columns) for frame in frames):
        raise ValueError(f"Each frame must contain {sorted(required)}.")
    combined = pd.concat(
        [frame.loc[:, ["date", TARGET]] for frame in frames], ignore_index=True
    )
    combined["date"] = pd.to_datetime(combined["date"], errors="raise")
    if combined["date"].isna().any():
        raise ValueError("Production audit dates must be non-null.")

    def finite_count(values: pd.Series) -> int:
        numeric = pd.to_numeric(values.dropna(), errors="coerce")
        return int(np.isfinite(numeric).sum())

    summary = (
        combined.groupby("date", sort=True)[TARGET]
        .agg(rows="size", nonnull_rows="count", finite_rows=finite_count)
        .reset_index()
    )
    summary["fully_mature"] = (
        (summary["nonnull_rows"] == summary["rows"])
        & (summary["finite_rows"] == summary["rows"])
    )
    return summary


def derive_training_eligibility(
    date_summary: pd.DataFrame,
    prediction_date: str,
    purge_dates: int = PURGE_DATES,
) -> dict:
    required = {"date", "rows", "nonnull_rows", "finite_rows", "fully_mature"}
    if not required.issubset(date_summary.columns):
        raise ValueError(f"Date summary is missing {sorted(required - set(date_summary))}.")
    if purge_dates <= 0:
        raise ValueError("Observed-date purge must be positive.")

    summary = date_summary.copy()
    summary["date"] = pd.to_datetime(summary["date"], errors="raise")
    summary = summary.sort_values("date", kind="stable").reset_index(drop=True)
    if summary["date"].duplicated().any():
        raise ValueError("Date summary contains duplicate dates.")
    prediction = pd.Timestamp(prediction_date)
    prior = summary.loc[summary["date"] < prediction]
    if len(prior) < purge_dates + 1:
        raise ValueError("Insufficient observed history for the requested purge.")

    purge = prior.tail(purge_dates)
    eligible_window = prior.loc[prior["date"] < purge.iloc[0]["date"]]
    if eligible_window.empty:
        raise ValueError("Observed-date purge leaves no training history.")
    if not eligible_window["fully_mature"].all():
        bad = eligible_window.loc[~eligible_window["fully_mature"], "date"]
        raise ValueError(
            "Non-mature targets occur inside the eligible historical window: "
            + ", ".join(bad.dt.strftime("%Y-%m-%d"))
        )

    mature = summary.loc[summary["fully_mature"]]
    if mature.empty:
        raise ValueError("No fully mature target date is available.")
    excluded = summary.loc[summary["date"] > eligible_window.iloc[-1]["date"]]
    non_mature = summary.loc[~summary["fully_mature"]]
    return {
        "prediction_date": prediction.strftime("%Y-%m-%d"),
        "purge_convention": "last 16 sorted observed historical dates before prediction date",
        "purge_date_count": int(purge_dates),
        "purge_dates": purge["date"].dt.strftime("%Y-%m-%d").tolist(),
        "purge_rows": int(purge["rows"].sum()),
        "latest_fully_mature_target_date": mature.iloc[-1]["date"].strftime("%Y-%m-%d"),
        "eligible_training_start": eligible_window.iloc[0]["date"].strftime("%Y-%m-%d"),
        "eligible_training_cutoff": eligible_window.iloc[-1]["date"].strftime("%Y-%m-%d"),
        "eligible_training_dates": int(len(eligible_window)),
        "eligible_training_rows": int(eligible_window["rows"].sum()),
        "excluded_dates_after_cutoff": int(len(excluded)),
        "excluded_rows_after_cutoff": int(excluded["rows"].sum()),
        "non_mature_dates": non_mature["date"].dt.strftime("%Y-%m-%d").tolist(),
        "non_mature_date_count": int(len(non_mature)),
        "non_mature_rows": int(non_mature["rows"].sum()),
    }


def validate_frozen_strategy(medium_features: list[str]) -> dict:
    policy = load_v1_policy()
    if len(medium_features) != 350 or len(set(medium_features)) != 350:
        raise ValueError("Official Medium must contain 350 unique features.")
    feature_hash = hashlib.sha256("\n".join(medium_features).encode()).hexdigest()
    if feature_hash != MEDIUM_FEATURE_LIST_SHA256:
        raise ValueError("Official Medium feature identity changed.")
    if policy["policy_id"] != V1_POLICY_ID:
        raise ValueError("Frozen production strategy identity changed.")
    return {
        "policy_id": V1_POLICY_ID,
        "policy_sha256": sha256_file(V1_POLICY_PATH),
        "medium_feature_count": len(medium_features),
        "medium_feature_list_sha256": feature_hash,
        "top250_policy_id": POLICY_ID,
        "top250_feature_count": SELECTED_FEATURE_COUNT,
        "lightgbm_parameters": DEFAULT_LIGHTGBM_PARAMS,
        "random_seed": RANDOM_STATE,
        "missing_value_handling": "LightGBM native NaN handling",
        "ensemble_weights": {"medium_lightgbm": 0.5, "top250_lightgbm": 0.5},
        "normalization": "per-date average rank normalized as (rank - 0.5) / n",
    }


def snapshot_compatibility(files: dict, expected: dict = RESEARCH_SNAPSHOT_SHA256) -> dict:
    actual = {name: item.get("sha256") for name, item in files.items()}
    matches = {name: actual.get(name) == digest for name, digest in expected.items()}
    return {
        "expected_sha256": expected,
        "matches_by_file": matches,
        "exact_research_snapshot_match": all(matches.values()),
        "historical_ic_reuse_allowed": all(matches.values()),
    }


def fetch_public_metadata(round_number: int) -> dict:
    api = NumerAPI()
    catalog = sorted(path for path in api.list_datasets() if path.startswith("v6.0/"))
    rounds = api.list_rounds(number=round_number)
    if len(rounds) != 1 or rounds[0]["number"] != round_number:
        raise ValueError(f"Public API did not return exactly round {round_number}.")
    round_info = rounds[0]
    return {
        "numerapi_version": __import__("numerapi").__version__,
        "current_round_at_query": api.get_current_round(),
        "requested_round": round_number,
        "round": {
            "number": round_info["number"],
            "target": round_info["target"],
            "data_datestamp": round_info["dataDatestamp"],
            "open_time": round_info["openTime"].isoformat(),
            "close_time": round_info["closeTime"].isoformat(),
            "close_staking_time": round_info["closeStakingTime"].isoformat(),
            "score_time": round_info["scoreTime"].isoformat(),
            "resolve_time": round_info["resolveTime"].isoformat(),
        },
        "v6_catalog": catalog,
    }


def audit_local_dataset(data_dir: Path, prediction_date: str) -> dict:
    features_path = data_dir / "features.json"
    train_path = data_dir / "train.parquet"
    validation_path = data_dir / "validation.parquet"
    for path in (features_path, train_path, validation_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    metadata = json.loads(features_path.read_text(encoding="utf-8"))
    medium_with_country = metadata["feature_sets"]["medium"]
    medium = [name for name in medium_with_country if name != "feature_country"]
    strategy = validate_frozen_strategy(medium)
    frames = [
        pq.read_table(path, columns=["date", TARGET]).to_pandas()
        for path in (train_path, validation_path)
    ]
    date_summary = summarize_date_targets(frames)
    eligibility = derive_training_eligibility(date_summary, prediction_date)
    files = {}
    for path in (features_path, train_path, validation_path):
        item = {
            "path": path.as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        if path.suffix == ".parquet":
            parquet = pq.ParquetFile(path)
            item.update(
                {
                    "rows": parquet.metadata.num_rows,
                    "columns": parquet.metadata.num_columns,
                    "row_groups": parquet.metadata.num_row_groups,
                }
            )
        files[path.name] = item

    existing_ic_dates = 1165
    additional_dates = eligibility["eligible_training_dates"] - existing_ic_dates
    if additional_dates < 0:
        raise ValueError("Production eligibility predates the sealed final IC coverage.")
    compatibility = snapshot_compatibility(files)
    return {
        "files": files,
        "schema": {
            "total_feature_columns": sum(
                name.startswith("feature_")
                for name in pq.ParquetFile(train_path).schema.names
            ),
            "medium_features": medium,
            "medium_feature_count": len(medium),
            "medium_includes_excluded_text_country": "feature_country" in medium_with_country,
            "targets_in_metadata": metadata["targets"],
        },
        "coverage": {
            "start": date_summary.iloc[0]["date"].strftime("%Y-%m-%d"),
            "end": date_summary.iloc[-1]["date"].strftime("%Y-%m-%d"),
            "observed_dates": int(len(date_summary)),
            "rows": int(date_summary["rows"].sum()),
        },
        "eligibility": eligibility,
        "ic_extension": {
            "existing_dates": existing_ic_dates,
            "existing_cutoff": "2025-04-25",
            "existing_feature_date_observations": 407750,
            "additional_dates_required": additional_dates,
            "additional_feature_date_observations": additional_dates * len(medium),
            "production_total_dates": eligibility["eligible_training_dates"],
            "production_feature_date_observations": eligibility["eligible_training_dates"] * len(medium),
            "reuse_requires_exact_snapshot_match": True,
            "historical_revisions_require_full_recomputation": True,
        },
        "snapshot_compatibility": compatibility,
        "frozen_strategy": strategy,
    }


def build_report(
    data_dir: Path,
    as_of_timestamp: str,
    round_number: int,
    prediction_date: str,
    query_public_api: bool,
) -> dict:
    as_of = datetime.fromisoformat(as_of_timestamp.replace("Z", "+00:00"))
    if as_of.tzinfo is None:
        raise ValueError("Production as-of timestamp must include a timezone.")
    report = {
        "audit_version": 1,
        "dataset_version": "v6.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "production_as_of_timestamp": as_of.isoformat(),
        "intended_round": round_number,
        "intended_prediction_date": prediction_date,
        "verified_official_sources": OFFICIAL_SOURCES,
        "official_requirements": {
            "supported_classic_version": "v6.0",
            "historical_paths": ["v6.0/train.parquet", "v6.0/validation.parquet", "v6.0/features.json"],
            "live_path": "v6.0/live.parquet",
            "live_path_is_round_specific": False,
            "round_identification": "NumerAPI.get_current_round plus public round metadata and dataDatestamp",
            "submission_schema": ["numerai_ticker", "prediction"],
            "prediction_range": "ranked floats in [0, 1]",
            "authentication": "NUMERAI_PUBLIC_ID and NUMERAI_SECRET_KEY with least-privilege upload scope",
            "historical_update_frequency_documented": False,
            "target_availability": "recent 60-day validation targets can remain null until mature",
            "hosted_model": {
                "format": "cloudpickle .pkl containing predict(live_features, live_benchmark_models)",
                "runtime": "matching Numerai Predict Python and dependency environment",
                "default_resources": "1 CPU, 4GB RAM, 10-minute runtime",
                "internet_access": False,
                "manual_api_upload_after_enablement": False,
            },
        },
        "local": audit_local_dataset(data_dir, prediction_date),
        "freshness_assessment": {
            "catalog_paths_verified": True,
            "public_catalog_exposes_content_hashes_or_sizes": False,
            "current_byte_identity_provable_without_refresh": False,
            "required_before_training": "download train, validation, and features into a new versioned snapshot; never overwrite the research snapshot",
            "snapshot_path_design": "data/v6.0/snapshots/<as_of>_round_<round_number>/",
        },
        "production_configuration_design": {
            "inherits_frozen_strategy": V1_POLICY_ID,
            "runtime_fields": [
                "training_as_of_timestamp",
                "eligible_training_cutoff",
                "source_data_snapshot_identity",
                "production_model_version",
                "training_artifact_identity",
            ],
        },
        "artifact_design": {
            "models": "outputs/v6.0/production/models/<model_version>/",
            "round_predictions": "outputs/v6.0/production/predictions/round_<round_number>/",
            "model_contents": [
                "production_config.json", "training_snapshot.json", "training_coverage.json",
                "ic_provenance.json", "selected_features.json", "models/medium.txt",
                "models/top250.txt", "model_hashes.json", "training_diagnostics.json",
                "PRODUCTION_TRAINING_COMPLETED.json",
            ],
        },
    }
    if query_public_api:
        report["official_api"] = fetch_public_metadata(round_number)
    return report


def write_report(report: dict, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Production audit already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--analysis-id", required=True)
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--round-number", required=True, type=int)
    parser.add_argument("--prediction-date", required=True)
    parser.add_argument("--query-public-api", action="store_true")
    parser.add_argument("--run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_dataset_version(args.version)
    if args.version != "v6.0":
        raise ValueError("Production audit currently supports only v6.0.")
    if not args.run:
        raise PermissionError("Production audit requires explicit --run confirmation.")
    report = build_report(
        DATA_ROOT / args.version,
        args.as_of,
        args.round_number,
        args.prediction_date,
        args.query_public_api,
    )
    path = OUTPUT_ROOT / args.version / "production_audit" / args.analysis_id / "audit.json"
    write_report(report, path)
    print(path)


if __name__ == "__main__":
    main()
