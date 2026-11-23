"""Bounded Train-only memory preflight for Numerai feature sets."""

import argparse
import json
import threading
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import psutil
import pyarrow.parquet as pq

from src.config import DEFAULT_LIGHTGBM_PARAMS, get_research_config
from src.data import (
    V6_IDENTITY_COLUMNS,
    prepare_v6_training_data,
    resolve_v6_numeric_feature_set,
)
from src.models import create_lightgbm_model, train_validated_lightgbm
from src.paths import OUTPUT_ROOT, resolve_data_dir


MAX_PROFILE_ROWS = 100_000
FOLD5_TRAIN_ROWS = 6_284_208


def validate_sample_rows(max_sample_rows: int) -> int:
    if not 1 <= max_sample_rows <= MAX_PROFILE_ROWS:
        raise ValueError(
            f"max_sample_rows must be between 1 and {MAX_PROFILE_ROWS:,}."
        )
    return max_sample_rows


def load_bounded_training_sample(
    data_dir: Path,
    features: list[str],
    max_sample_rows: int,
) -> pd.DataFrame:
    """Read at most one bounded batch from Train; Validation is never opened."""
    max_sample_rows = validate_sample_rows(max_sample_rows)
    config = get_research_config("v6.0")
    columns = [
        *V6_IDENTITY_COLUMNS,
        "data_type",
        config.target_column,
        *features,
    ]
    parquet = pq.ParquetFile(data_dir / "train.parquet")
    try:
        missing = sorted(set(columns) - set(parquet.schema_arrow.names))
        if missing:
            raise ValueError(f"Train is missing profiling columns: {missing}")
        batch = next(parquet.iter_batches(batch_size=max_sample_rows, columns=columns))
    finally:
        parquet.close()
    sample = batch.to_pandas()
    if len(sample) > max_sample_rows:
        raise RuntimeError("Bounded reader exceeded max_sample_rows.")
    if set(sample["data_type"].astype(str).unique()) != {"train"}:
        raise ValueError("Profiling sample contains non-Train rows.")
    if sample[config.target_column].isna().any():
        raise ValueError("Profiling sample contains null training targets.")
    return sample


class _PeakRSS:
    def __init__(self, interval_seconds: float = 0.02) -> None:
        self.interval_seconds = interval_seconds
        self.peak = psutil.Process().memory_info().rss
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        process = psutil.Process()
        while not self._stop.is_set():
            self.peak = max(self.peak, process.memory_info().rss)
            self._stop.wait(self.interval_seconds)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, psutil.Process().memory_info().rss)


def profile_feature_memory(
    data_dir: Path,
    feature_set: str,
    max_sample_rows: int,
    fit_lightgbm: bool = False,
) -> dict:
    config = get_research_config("v6.0")
    max_sample_rows = validate_sample_rows(max_sample_rows)
    features = resolve_v6_numeric_feature_set(feature_set, data_dir)
    process = psutil.Process()
    rss_before_load = process.memory_info().rss
    sample = load_bounded_training_sample(data_dir, features, max_sample_rows)
    rss_after_load = process.memory_info().rss

    memory_by_column = sample.memory_usage(index=False, deep=True)
    sample_bytes = int(sample.memory_usage(index=True, deep=True).sum())
    feature_bytes = int(memory_by_column.loc[features].sum())
    input_dtypes = {name: str(sample[name].dtype) for name in features}
    dtype_counts = dict(sorted(Counter(input_dtypes.values()).items()))

    rss_before_extract = process.memory_info().rss
    model_data = prepare_v6_training_data(sample, features)
    rss_after_extract = process.memory_info().rss
    model_feature_bytes = int(
        model_data.features.memory_usage(index=True, deep=True).sum()
    )
    extracted_dtypes = {
        name: str(model_data.features[name].dtype) for name in features
    }
    conversions = [
        {"feature": name, "before": input_dtypes[name], "after": extracted_dtypes[name]}
        for name in features
        if input_dtypes[name] != extracted_dtypes[name]
    ]
    bytes_per_row = feature_bytes / len(sample)
    legacy_feature_copy_bytes = feature_bytes + 2 * model_feature_bytes
    optimized_feature_copy_bytes = feature_bytes + model_feature_bytes

    fit_report = {
        "executed": False,
        "parameters": DEFAULT_LIGHTGBM_PARAMS,
        "elapsed_seconds": None,
        "rss_before_bytes": None,
        "peak_rss_bytes": None,
        "rss_after_bytes": None,
        "peak_measurement": "psutil RSS sampled every 0.02 seconds",
    }
    if fit_lightgbm:
        model = create_lightgbm_model(DEFAULT_LIGHTGBM_PARAMS)
        fit_report["rss_before_bytes"] = process.memory_info().rss
        started = time.perf_counter()
        with _PeakRSS() as monitor:
            train_validated_lightgbm(
                model,
                model_data.features,
                model_data.target,
                features,
            )
        fit_report.update(
            {
                "executed": True,
                "elapsed_seconds": time.perf_counter() - started,
                "peak_rss_bytes": monitor.peak,
                "rss_after_bytes": process.memory_info().rss,
            }
        )

    return {
        "report_type": "bounded_feature_memory_preflight",
        "dataset_version": config.dataset_version,
        "feature_set": feature_set,
        "target": config.target_column,
        "scope": "Train-only engineering profile; no OOF scoring or holdout access",
        "sample": {
            "max_requested_rows": max_sample_rows,
            "actual_rows": len(sample),
            "feature_count": len(features),
            "feature_dtype_counts": dtype_counts,
            "sample_frame_deep_bytes": sample_bytes,
            "sample_feature_bytes": feature_bytes,
            "model_feature_copy_bytes": model_feature_bytes,
            "feature_bytes_per_row": bytes_per_row,
            "dtype_conversions": conversions,
            "legacy_concurrent_feature_bytes": legacy_feature_copy_bytes,
            "optimized_concurrent_feature_bytes": optimized_feature_copy_bytes,
            "estimated_avoided_feature_copy_bytes": (
                legacy_feature_copy_bytes - optimized_feature_copy_bytes
            ),
        },
        "process_memory": {
            "rss_before_load_bytes": rss_before_load,
            "rss_after_load_bytes": rss_after_load,
            "rss_before_extract_bytes": rss_before_extract,
            "rss_after_extract_bytes": rss_after_extract,
        },
        "fold5_extrapolation": {
            "training_rows": FOLD5_TRAIN_ROWS,
            "single_feature_matrix_bytes": int(bytes_per_row * FOLD5_TRAIN_ROWS),
            "method": "sample feature bytes per row multiplied by observed Fold 5 rows",
            "limitations": (
                "Excludes DataFrame/index overhead changes, research-history and fold copies, "
                "validation matrices, LightGBM bins/datasets, temporary arrays, and OS memory."
            ),
        },
        "lightgbm_fit": fit_report,
        "risk_assessment": {
            "classification": "High risk / not recommended",
            "scope": "Optimized all-feature pipeline on a 32GB Windows machine",
            "evidence": [
                "A single Fold 5 pandas feature matrix extrapolates to roughly 6.94 GB.",
                "The optimized fold path removes the full concatenated history and whole-fold copies.",
                "LightGBM requires additional native dataset, bin, histogram, and temporary memory.",
                "The 100,000-row fit peak is not a reliable linear predictor of multi-million-row peak memory.",
            ],
            "decision": (
                "Do not start the full all-feature walk-forward until a larger staged "
                "profile demonstrates sufficient headroom."
            ),
        },
    }


def write_profile_report(report: dict, output_path: Path) -> Path:
    if output_path.exists():
        raise FileExistsError(f"Profiling report already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=2, allow_nan=False)
    return output_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--feature-set", required=True)
    parser.add_argument("--max-sample-rows", type=int, required=True)
    parser.add_argument("--fit-lightgbm", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--data-dir", type=Path)
    args = parser.parse_args(argv)
    if args.version != "v6.0":
        raise SystemExit("Memory profiling currently supports only v6.0.")
    data_dir = args.data_dir or resolve_data_dir(args.version)
    output = args.output or (
        OUTPUT_ROOT
        / args.version
        / "profiling"
        / f"{args.feature_set}_{args.max_sample_rows}_rows.json"
    )
    report = profile_feature_memory(
        data_dir,
        args.feature_set,
        args.max_sample_rows,
        args.fit_lightgbm,
    )
    write_profile_report(report, output)
    print(output)


if __name__ == "__main__":
    main()
