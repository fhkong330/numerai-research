"""Dependency-only runner for isolated Hosted Model package validation."""

import argparse
import json
import time
import tracemalloc

import cloudpickle
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True)
    parser.add_argument("--live", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--drop-feature")
    args = parser.parse_args()
    started = time.perf_counter()
    with open(args.package, "rb") as package_file:
        predict = cloudpickle.load(package_file)
    load_seconds = time.perf_counter() - started
    live = pd.read_parquet(args.live).set_index("numerai_ticker")
    if args.shuffle:
        live = live.sample(frac=1.0, random_state=42)
    if args.drop_feature:
        live = live.drop(columns=args.drop_feature)
    benchmark = pd.DataFrame(index=live.index)
    tracemalloc.start()
    inference_started = time.perf_counter()
    output = predict(live, benchmark)
    inference_seconds = time.perf_counter() - inference_started
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    if output.columns.tolist() != ["prediction"] or not output.index.equals(live.index):
        raise ValueError("Hosted output contract is invalid.")
    values = output["prediction"].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or np.min(values) < 0 or np.max(values) > 1:
        raise ValueError("Hosted predictions are invalid.")
    saved = output.reset_index()
    saved.to_parquet(args.output, index=False)
    with open(args.report, "w", encoding="utf-8") as report_file:
        json.dump(
            {
                "rows": len(output),
                "package_load_seconds": load_seconds,
                "inference_seconds": inference_seconds,
                "python_tracemalloc_peak_bytes": peak_bytes,
                "prediction_min": float(np.min(values)),
                "prediction_max": float(np.max(values)),
                "output_columns": output.columns.tolist(),
                "input_index_preserved": True,
                "extra_columns_present": len(live.columns) > 350,
            },
            report_file,
            indent=2,
        )


if __name__ == "__main__":
    main()
