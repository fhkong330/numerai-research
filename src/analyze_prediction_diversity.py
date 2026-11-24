"""Target-free diversity audit for aligned saved v6 OOF predictions."""

import argparse
import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiment_artifacts import (
    current_git_state,
    load_v6_experiment_artifacts,
)
from src.paths import OUTPUT_ROOT


IDENTITY_COLUMNS = ["date", "numerai_ticker", "composite_figi", "fold_id"]
MODEL_LABELS = ("A", "B", "C")


def _canonical_prediction_frame(artifact: dict, label: str) -> pd.DataFrame:
    oof = artifact["oof_predictions"]
    required = {*IDENTITY_COLUMNS, "prediction"}
    missing = sorted(required - set(oof.columns))
    if missing:
        raise ValueError(f"Experiment {label} OOF is missing columns: {missing}")
    frame = oof.loc[:, [*IDENTITY_COLUMNS, "prediction"]].copy()
    if frame.duplicated(IDENTITY_COLUMNS).any():
        raise ValueError(f"Experiment {label} contains duplicate OOF identities.")
    return frame.sort_values(IDENTITY_COLUMNS, na_position="first").reset_index(
        drop=True
    )


def align_saved_predictions(artifacts: dict[str, dict]) -> tuple[pd.DataFrame, dict]:
    """Align exactly three experiments without reading any target column."""
    if tuple(artifacts) != MODEL_LABELS:
        raise ValueError("Prediction diversity requires ordered labels A, B, and C.")
    metadata = {label: artifacts[label]["metadata"] for label in MODEL_LABELS}
    baseline = metadata["A"]
    for label in MODEL_LABELS[1:]:
        candidate = metadata[label]
        for field in ("dataset_version", "target", "folds", "metric_identity"):
            if candidate.get(field) != baseline.get(field):
                raise ValueError(f"Experiment {label} metadata mismatch: {field}")
        if candidate.get("ncorr_reference") != baseline.get("ncorr_reference"):
            raise ValueError(f"Experiment {label} NCORR reference mismatch.")

    frames = {
        label: _canonical_prediction_frame(artifacts[label], label)
        for label in MODEL_LABELS
    }
    base_identity = frames["A"][IDENTITY_COLUMNS]
    aligned = base_identity.copy()
    diagnostics = {}
    for label in MODEL_LABELS:
        frame = frames[label]
        if not frame[IDENTITY_COLUMNS].equals(base_identity):
            raise ValueError(f"Experiment {label} OOF identities do not match exactly.")
        prediction = pd.to_numeric(frame["prediction"], errors="coerce")
        aligned[f"prediction_{label}"] = prediction.to_numpy()
        diagnostics[label] = {
            "missing_or_nonfinite_predictions": int(
                (~np.isfinite(prediction.to_numpy(dtype=np.float64))).sum()
            )
        }
    return aligned, diagnostics


def _decile_membership(values: np.ndarray) -> tuple[set[int], set[int]]:
    count = max(1, math.ceil(len(values) * 0.10))
    order = np.argsort(values, kind="stable")
    return set(order[:count]), set(order[-count:])


def calculate_per_date_diversity(aligned: pd.DataFrame) -> pd.DataFrame:
    required = {*IDENTITY_COLUMNS, *(f"prediction_{label}" for label in MODEL_LABELS)}
    missing = sorted(required - set(aligned.columns))
    if missing:
        raise ValueError(f"Aligned predictions are missing columns: {missing}")
    rows = []
    for date, date_frame in aligned.groupby("date", sort=True):
        fold_values = date_frame["fold_id"].unique()
        if len(fold_values) != 1:
            raise ValueError(f"Date {date} spans multiple folds.")
        for left, right in combinations(MODEL_LABELS, 2):
            left_values = date_frame[f"prediction_{left}"].to_numpy(dtype=np.float64)
            right_values = date_frame[f"prediction_{right}"].to_numpy(dtype=np.float64)
            valid = np.isfinite(left_values) & np.isfinite(right_values)
            x = left_values[valid]
            y = right_values[valid]
            constant_left = len(x) > 0 and np.ptp(x) == 0
            constant_right = len(y) > 0 and np.ptp(y) == 0
            correlations_valid = len(x) >= 2 and not constant_left and not constant_right
            if correlations_valid:
                pearson = float(np.corrcoef(x, y)[0, 1])
                left_rank = pd.Series(x).rank(method="average", pct=True).to_numpy()
                right_rank = pd.Series(y).rank(method="average", pct=True).to_numpy()
                spearman = float(np.corrcoef(left_rank, right_rank)[0, 1])
                rank_disagreement = float(np.abs(left_rank - right_rank).mean())
                bottom_left, top_left = _decile_membership(x)
                bottom_right, top_right = _decile_membership(y)
                decile_count = len(top_left)
                top_overlap = len(top_left & top_right)
                bottom_overlap = len(bottom_left & bottom_right)
            else:
                pearson = spearman = rank_disagreement = float("nan")
                decile_count = top_overlap = bottom_overlap = 0
            rows.append(
                {
                    "date": str(date),
                    "year": int(str(date)[:4]),
                    "fold_id": int(fold_values[0]),
                    "pair": f"{left}_vs_{right}",
                    "model_left": left,
                    "model_right": right,
                    "aligned_observations": int(valid.sum()),
                    "missing_left": int((~np.isfinite(left_values)).sum()),
                    "missing_right": int((~np.isfinite(right_values)).sum()),
                    "constant_left": bool(constant_left),
                    "constant_right": bool(constant_right),
                    "pearson": pearson,
                    "spearman": spearman,
                    "mean_absolute_rank_disagreement": rank_disagreement,
                    "decile_count": decile_count,
                    "top_decile_overlap": top_overlap,
                    "top_decile_overlap_percentage": (
                        100.0 * top_overlap / decile_count if decile_count else float("nan")
                    ),
                    "bottom_decile_overlap": bottom_overlap,
                    "bottom_decile_overlap_percentage": (
                        100.0 * bottom_overlap / decile_count
                        if decile_count
                        else float("nan")
                    ),
                }
            )
    return pd.DataFrame(rows)


def _summarize_group(group: pd.DataFrame, identity: dict) -> dict:
    row = {**identity, "date_count": int(group["date"].nunique())}
    for metric in ("pearson", "spearman"):
        values = group[metric].dropna()
        row.update(
            {
                f"{metric}_mean": float(values.mean()),
                f"{metric}_median": float(values.median()),
                f"{metric}_std": float(values.std(ddof=1)),
                f"{metric}_min": float(values.min()),
                f"{metric}_max": float(values.max()),
                f"{metric}_above_0_9_percentage": float(values.gt(0.9).mean() * 100),
                f"{metric}_below_0_5_percentage": float(values.lt(0.5).mean() * 100),
            }
        )
    row["rank_disagreement_mean"] = float(
        group["mean_absolute_rank_disagreement"].mean()
    )
    row["rank_disagreement_median"] = float(
        group["mean_absolute_rank_disagreement"].median()
    )
    row["top_decile_overlap_mean_percentage"] = float(
        group["top_decile_overlap_percentage"].mean()
    )
    row["bottom_decile_overlap_mean_percentage"] = float(
        group["bottom_decile_overlap_percentage"].mean()
    )
    return row


def summarize_diversity(per_date: pd.DataFrame, group_column: str | None = None) -> pd.DataFrame:
    rows = []
    if group_column is None:
        for pair, group in per_date.groupby("pair", sort=True):
            rows.append(_summarize_group(group, {"pair": pair}))
    else:
        for (pair, value), group in per_date.groupby(["pair", group_column], sort=True):
            rows.append(
                _summarize_group(group, {"pair": pair, group_column: value})
            )
    return pd.DataFrame(rows)


def build_research_summary(
    per_date: pd.DataFrame,
    overall: pd.DataFrame,
) -> dict:
    pairs = {}
    for row in overall.to_dict(orient="records"):
        pair = row.pop("pair")
        group = per_date[per_date["pair"].eq(pair)]
        lowest = group.nsmallest(5, "spearman")[["date", "fold_id", "spearman"]]
        highest = group.nlargest(5, "spearman")[["date", "fold_id", "spearman"]]
        pairs[pair] = {
            **row,
            "lowest_spearman_dates": lowest.to_dict(orient="records"),
            "highest_spearman_dates": highest.to_dict(orient="records"),
        }
    return {"pairs": pairs}


def save_diversity_artifacts(
    analysis_id: str,
    experiment_ids: dict[str, str],
    artifacts: dict[str, dict],
    aligned: pd.DataFrame,
    diagnostics: dict,
    per_date: pd.DataFrame,
    output_root: Path = OUTPUT_ROOT,
) -> Path:
    analysis_dir = output_root / "v6.0" / "prediction_diversity" / analysis_id
    if analysis_dir.exists():
        raise FileExistsError(f"Prediction-diversity analysis exists: {analysis_dir}")
    overall = summarize_diversity(per_date)
    folds = summarize_diversity(per_date, "fold_id")
    annual = summarize_diversity(per_date, "year")
    research_summary = build_research_summary(per_date, overall)
    metadata = {
        "analysis_id": analysis_id,
        "dataset_version": "v6.0",
        "experiment_ids": experiment_ids,
        "model_labels": list(MODEL_LABELS),
        "prediction_columns": [f"prediction_{label}" for label in MODEL_LABELS],
        "identity_columns": IDENTITY_COLUMNS,
        "identity_rule": "exact canonical sort with null identities first",
        "target_columns_accessed": [],
        "date_start": str(aligned["date"].min()),
        "date_end": str(aligned["date"].max()),
        "date_count": int(aligned["date"].nunique()),
        "row_count": int(len(aligned)),
        "diagnostics": diagnostics,
        "source_metadata": {
            label: {
                "experiment_id": artifacts[label]["metadata"]["experiment_id"],
                "model_family": artifacts[label]["metadata"]["model_family"],
                "feature_set": artifacts[label]["metadata"]["feature_set"],
                "target": artifacts[label]["metadata"]["target"],
                "metric_identity": artifacts[label]["metadata"]["metric_identity"],
                "ncorr_reference": artifacts[label]["metadata"]["ncorr_reference"],
            }
            for label in MODEL_LABELS
        },
        "git": current_git_state(),
    }
    analysis_dir.mkdir(parents=True, exist_ok=False)
    try:
        per_date.loc[
            :,
            [
                "date", "year", "fold_id", "pair", "model_left", "model_right",
                "aligned_observations", "missing_left", "missing_right",
                "constant_left", "constant_right", "pearson", "spearman",
            ],
        ].to_parquet(analysis_dir / "per_date_correlations.parquet", index=False)
        folds.to_parquet(analysis_dir / "fold_summaries.parquet", index=False)
        annual.to_parquet(analysis_dir / "annual_summaries.parquet", index=False)
        per_date.loc[
            :, ["date", "year", "fold_id", "pair", "mean_absolute_rank_disagreement"]
        ].to_parquet(analysis_dir / "rank_disagreement.parquet", index=False)
        per_date.loc[
            :,
            [
                "date", "year", "fold_id", "pair", "decile_count",
                "top_decile_overlap", "top_decile_overlap_percentage",
                "bottom_decile_overlap", "bottom_decile_overlap_percentage",
            ],
        ].to_parquet(analysis_dir / "decile_overlap.parquet", index=False)
        with (analysis_dir / "metadata.json").open("x", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2, allow_nan=False)
        with (analysis_dir / "research_summary.json").open("x", encoding="utf-8") as file:
            json.dump(research_summary, file, indent=2, allow_nan=False)
        with (analysis_dir / "COMPLETED.json").open("x", encoding="utf-8") as file:
            json.dump({"status": "complete"}, file, indent=2)
    except Exception:
        raise
    return analysis_dir


def run_prediction_diversity_analysis(
    analysis_id: str,
    experiment_ids: dict[str, str],
    output_root: Path = OUTPUT_ROOT,
) -> Path:
    experiment_root = output_root / "v6.0" / "experiments"
    artifacts = {
        label: load_v6_experiment_artifacts(experiment_root / experiment_ids[label])
        for label in MODEL_LABELS
    }
    aligned, diagnostics = align_saved_predictions(artifacts)
    if len(aligned) != 2_180_295 or aligned["date"].nunique() != 330:
        raise ValueError("Aligned OOF does not match the approved research identity set.")
    per_date = calculate_per_date_diversity(aligned)
    return save_diversity_artifacts(
        analysis_id,
        experiment_ids,
        artifacts,
        aligned,
        diagnostics,
        per_date,
        output_root,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--analysis-id", required=True)
    parser.add_argument("--experiment-a", required=True)
    parser.add_argument("--experiment-b", required=True)
    parser.add_argument("--experiment-c", required=True)
    args = parser.parse_args(argv)
    if args.version != "v6.0":
        raise SystemExit("Prediction diversity currently supports only v6.0.")
    result = run_prediction_diversity_analysis(
        args.analysis_id,
        {"A": args.experiment_a, "B": args.experiment_b, "C": args.experiment_c},
    )
    print(result)


if __name__ == "__main__":
    main()
