"""Report saved metrics for the two predeclared Task 21B ensembles."""

import argparse
import json
from pathlib import Path

from src.compare_experiments import compare_loaded_v6_experiments
from src.ensemble import load_ensemble_artifacts
from src.experiment_artifacts import load_v6_experiment_artifacts
from src.paths import OUTPUT_ROOT


EXPERIMENT_IDS = (
    "v6_medium_baseline_001",
    "v6_medium_top250_ic_002",
    "v6_medium_ridge_003",
)
ENSEMBLE_IDS = (
    "v6_medium_top250_rank_equal_001",
    "v6_medium_top250_ridge_rank_442_002",
)


def _comparable_ensemble(loaded: dict) -> dict:
    return {
        "metadata": loaded["metadata"],
        "metrics": loaded["metrics"],
        "oof_predictions": loaded["predictions"],
        "per_date_metrics": loaded["per_date_metrics"],
    }


def summarize_candidate(candidate_id: str, loaded: dict) -> dict:
    per_date = loaded["per_date_metrics"]
    metrics = loaded["metrics"]
    ncorr = metrics["ncorr"]["overall"]
    legacy = metrics["legacy_corr"]["overall"]
    scores = per_date["ncorr"]
    return {
        "candidate_id": candidate_id,
        "mean_ncorr": float(ncorr["mean"]),
        "ncorr_std": float(ncorr["std"]),
        "ncorr_mean_std_ratio": float(ncorr["sharpe_like"]),
        "positive_ncorr_date_percentage": float(ncorr["positive_date_percentage"]),
        "mean_legacy_corr": float(legacy["mean"]),
        "legacy_corr_mean_std_ratio": float(legacy["sharpe_like"]),
        "worst_weekly_ncorr": float(scores.min()),
        "p10_weekly_ncorr": float(scores.quantile(0.10)),
    }


def build_saved_ensemble_report(output_root: Path = OUTPUT_ROOT) -> dict:
    experiment_root = output_root / "v6.0" / "experiments"
    ensemble_root = output_root / "v6.0" / "ensembles"
    experiments = {
        name: load_v6_experiment_artifacts(experiment_root / name)
        for name in EXPERIMENT_IDS
    }
    ensembles = {
        name: _comparable_ensemble(
            load_ensemble_artifacts(ensemble_root / name, require_scored=True)
        )
        for name in ENSEMBLE_IDS
    }
    all_candidates = {**experiments, **ensembles}
    comparisons = {
        "ensemble_a_minus_medium": compare_loaded_v6_experiments(
            experiments[EXPERIMENT_IDS[0]], ensembles[ENSEMBLE_IDS[0]]
        ),
        "ensemble_a_minus_top250": compare_loaded_v6_experiments(
            experiments[EXPERIMENT_IDS[1]], ensembles[ENSEMBLE_IDS[0]]
        ),
        "ensemble_b_minus_ensemble_a": compare_loaded_v6_experiments(
            ensembles[ENSEMBLE_IDS[0]], ensembles[ENSEMBLE_IDS[1]]
        ),
    }
    return {
        "candidate_summaries": [
            summarize_candidate(name, all_candidates[name]) for name in all_candidates
        ],
        "paired_comparisons": comparisons,
        "ensemble_metrics": {
            name: ensembles[name]["metrics"] for name in ENSEMBLE_IDS
        },
        "target_columns_read": [],
        "scores_recalculated": False,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    args = parser.parse_args(argv)
    if args.version != "v6.0":
        raise SystemExit("Ensemble reporting currently supports only v6.0.")
    print(json.dumps(build_saved_ensemble_report(), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
