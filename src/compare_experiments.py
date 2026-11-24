"""Validate and compare two completed v6 experiment artifacts."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiment_artifacts import load_v6_experiment_artifacts
from src.ensemble import load_ensemble_artifacts
from src.paths import OUTPUT_ROOT


IDENTITY_COLUMNS = ["date", "numerai_ticker", "composite_figi"]
SCORE_COLUMNS = ["ncorr", "numerai_corr_jupiter_60"]


def _validate_compatible(baseline: dict, candidate: dict) -> None:
    baseline_meta = baseline["metadata"]
    candidate_meta = candidate["metadata"]
    for field in ("dataset_version", "target", "metric_identity"):
        if baseline_meta.get(field) != candidate_meta.get(field):
            raise ValueError(f"Experiment {field} values do not match.")
    if baseline_meta.get("ncorr_reference") != candidate_meta.get("ncorr_reference"):
        raise ValueError("Experiment NCORR scoring references do not match.")

    baseline_oof = baseline["oof_predictions"]
    candidate_oof = candidate["oof_predictions"]
    required_oof = set(IDENTITY_COLUMNS)
    for name, frame in (("baseline", baseline_oof), ("candidate", candidate_oof)):
        missing = sorted(required_oof - set(frame.columns))
        if missing:
            raise ValueError(f"{name} OOF is missing columns: {missing}")
    baseline_order = baseline_oof.sort_values(IDENTITY_COLUMNS, na_position="first")
    candidate_order = candidate_oof.sort_values(IDENTITY_COLUMNS, na_position="first")
    baseline_identity = baseline_order[IDENTITY_COLUMNS].reset_index(drop=True).astype("string")
    candidate_identity = candidate_order[IDENTITY_COLUMNS].reset_index(drop=True).astype("string")
    if not baseline_identity.equals(candidate_identity):
        raise ValueError("Experiment OOF row identities or ordering do not match.")
    target = baseline_meta["target"]
    if target in baseline_oof and target in candidate_oof:
        if not np.array_equal(
            baseline_order[target].to_numpy(), candidate_order[target].to_numpy(), equal_nan=True
        ):
            raise ValueError("Experiment OOF target values do not match.")

    for name, experiment in (("baseline", baseline), ("candidate", candidate)):
        per_date = experiment["per_date_metrics"]
        missing = sorted({"date", "fold_id", *SCORE_COLUMNS} - set(per_date.columns))
        if missing:
            raise ValueError(f"{name} per-date metrics are missing columns: {missing}")
        if per_date["date"].duplicated().any():
            raise ValueError(f"{name} per-date metrics contain duplicate dates.")
    if baseline["per_date_metrics"]["date"].astype(str).tolist() != candidate[
        "per_date_metrics"
    ]["date"].astype(str).tolist():
        raise ValueError("Experiment scoring dates or ordering do not match.")


def compare_loaded_v6_experiments(baseline: dict, candidate: dict) -> dict:
    _validate_compatible(baseline, candidate)
    baseline_meta = baseline["metadata"]
    candidate_meta = candidate["metadata"]
    left = baseline["per_date_metrics"].copy()
    right = candidate["per_date_metrics"].copy()
    paired = left[["date", "fold_id", *SCORE_COLUMNS]].merge(
        right[["date", "fold_id", *SCORE_COLUMNS]],
        on=["date", "fold_id"],
        suffixes=("_baseline", "_candidate"),
        validate="one_to_one",
    )
    paired["ncorr_difference"] = paired["ncorr_candidate"] - paired["ncorr_baseline"]
    paired["legacy_corr_difference"] = (
        paired["numerai_corr_jupiter_60_candidate"]
        - paired["numerai_corr_jupiter_60_baseline"]
    )
    paired["year"] = paired["date"].str.slice(0, 4).astype(int)

    def grouped(field: str) -> list[dict]:
        rows = []
        for value, group in paired.groupby(field, sort=True):
            rows.append(
                {
                    field: int(value),
                    "date_count": int(len(group)),
                    "baseline_ncorr": float(group["ncorr_baseline"].mean()),
                    "candidate_ncorr": float(group["ncorr_candidate"].mean()),
                    "ncorr_difference": float(group["ncorr_difference"].mean()),
                    "baseline_legacy_corr": float(
                        group["numerai_corr_jupiter_60_baseline"].mean()
                    ),
                    "candidate_legacy_corr": float(
                        group["numerai_corr_jupiter_60_candidate"].mean()
                    ),
                    "legacy_corr_difference": float(
                        group["legacy_corr_difference"].mean()
                    ),
                }
            )
        return rows

    differences = paired["ncorr_difference"]

    def feature_count(metadata: dict) -> int:
        if metadata.get("artifact_type") == "ensemble":
            return 0
        selection = metadata.get("feature_selection")
        return (
            int(selection["selected_feature_count_per_fold"])
            if selection
            else len(metadata["ordered_feature_names"])
        )
    return {
        "baseline": {
            "experiment_id": baseline_meta["experiment_id"],
            "feature_set": baseline_meta["feature_set"],
            "feature_count": feature_count(baseline_meta),
            "metrics": baseline["metrics"],
        },
        "candidate": {
            "experiment_id": candidate_meta["experiment_id"],
            "feature_set": candidate_meta["feature_set"],
            "feature_count": feature_count(candidate_meta),
            "metrics": candidate["metrics"],
        },
        "paired_ncorr": {
            "date_count": int(len(paired)),
            "mean_difference": float(differences.mean()),
            "std_difference": float(differences.std()),
            "median_difference": float(differences.median()),
            "p10_difference": float(differences.quantile(0.10)),
            "p90_difference": float(differences.quantile(0.90)),
            "candidate_outperformance_percentage": float(
                (differences > 0).mean() * 100
            ),
            "top_10_absolute_difference_share": float(
                differences.abs().nlargest(10).sum() / differences.abs().sum()
            ),
            "largest_improvement": float(differences.max()),
            "largest_decline": float(differences.min()),
        },
        "per_fold": grouped("fold_id"),
        "per_year": grouped("year"),
    }


def compare_v6_experiment_paths(baseline_dir: Path, candidate_dir: Path) -> dict:
    return compare_loaded_v6_experiments(
        load_comparable_v6_artifacts(baseline_dir),
        load_comparable_v6_artifacts(candidate_dir),
    )


def load_comparable_v6_artifacts(path: Path) -> dict:
    """Load either a completed experiment or a scored ensemble."""
    metadata_path = path / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Artifact metadata not found: {metadata_path}")
    with metadata_path.open(encoding="utf-8") as file:
        metadata = json.load(file)
    if metadata.get("artifact_type") != "ensemble":
        return load_v6_experiment_artifacts(path)
    ensemble = load_ensemble_artifacts(path, require_scored=True)
    return {
        "metadata": ensemble["metadata"],
        "metrics": ensemble["metrics"],
        "oof_predictions": ensemble["predictions"],
        "per_date_metrics": ensemble["per_date_metrics"],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    args = parser.parse_args(argv)
    if args.version != "v6.0":
        raise SystemExit("Experiment comparison currently supports only v6.0.")
    root = OUTPUT_ROOT / args.version / "experiments"
    result = compare_v6_experiment_paths(root / args.baseline, root / args.candidate)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
