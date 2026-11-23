"""Analyze saved v6 LightGBM importance without fitting or scoring models."""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import Booster

from src.data import resolve_v6_numeric_feature_set
from src.experiment_artifacts import current_git_state, load_v6_experiment_artifacts
from src.feature_quality import load_feature_quality_artifacts
from src.paths import OUTPUT_ROOT, resolve_data_dir


TOP_RANK_THRESHOLD = 35
CONSISTENT_TOP_FOLDS = 4


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def extract_fold_importance(
    booster,
    fold_id: int,
    expected_features: list[str],
) -> pd.DataFrame:
    names = list(booster.feature_name())
    if names != expected_features:
        raise ValueError(f"Fold {fold_id} model feature names or order do not match.")
    if len(names) != 350:
        raise ValueError(f"Fold {fold_id} has {len(names)} features; expected 350.")
    gain = np.asarray(booster.feature_importance(importance_type="gain"), dtype=float)
    split = np.asarray(booster.feature_importance(importance_type="split"), dtype=float)
    if len(gain) != len(names) or len(split) != len(names):
        raise ValueError(f"Fold {fold_id} importance length does not match features.")
    if (gain < 0).any() or (split < 0).any() or not (
        np.isfinite(gain).all() and np.isfinite(split).all()
    ):
        raise ValueError(f"Fold {fold_id} contains invalid importance values.")
    gain_total = float(gain.sum())
    split_total = float(split.sum())
    if gain_total <= 0 or split_total <= 0:
        raise ValueError(f"Fold {fold_id} has zero total model importance.")
    frame = pd.DataFrame(
        {
            "fold_id": fold_id,
            "feature_position": np.arange(len(names), dtype=int),
            "feature": names,
            "gain_importance": gain,
            "split_importance": split,
            "normalized_gain_share": gain / gain_total,
            "normalized_split_share": split / split_total,
        }
    )
    frame["gain_rank"] = frame["gain_importance"].rank(
        ascending=False, method="average"
    )
    frame["split_rank"] = frame["split_importance"].rank(
        ascending=False, method="average"
    )
    frame["feature_usage_rank"] = frame["gain_rank"]
    frame["zero_gain_importance"] = frame["gain_importance"].eq(0)
    frame["zero_split_importance"] = frame["split_importance"].eq(0)
    frame["zero_importance"] = frame["zero_gain_importance"] & frame[
        "zero_split_importance"
    ]
    return frame


def load_saved_fold_importance(
    experiment_dir: Path,
    expected_features: list[str],
) -> pd.DataFrame:
    model_dir = experiment_dir / "models"
    paths = [model_dir / f"fold_{fold_id}.txt" for fold_id in range(1, 6)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Saved fold models are missing: {missing}")
    parts = []
    for fold_id, path in enumerate(paths, start=1):
        parts.append(extract_fold_importance(Booster(model_file=str(path)), fold_id, expected_features))
    return pd.concat(parts, ignore_index=True)


def summarize_cross_fold_importance(per_fold: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if per_fold.duplicated(["fold_id", "feature"]).any():
        raise ValueError("Per-fold importance contains duplicate identities.")
    if set(per_fold["fold_id"]) != {1, 2, 3, 4, 5}:
        raise ValueError("Per-fold importance must contain exactly folds 1 through 5.")
    rows = []
    for feature, group in per_fold.groupby("feature", sort=False):
        ordered = group.sort_values("fold_id")
        top_fold_count = int((ordered["gain_rank"] <= TOP_RANK_THRESHOLD).sum())
        rows.append(
            {
                "feature": feature,
                "feature_position": int(ordered["feature_position"].iloc[0]),
                "mean_normalized_gain_share": float(ordered["normalized_gain_share"].mean()),
                "median_normalized_gain_share": float(ordered["normalized_gain_share"].median()),
                "mean_normalized_split_share": float(ordered["normalized_split_share"].mean()),
                "median_normalized_split_share": float(ordered["normalized_split_share"].median()),
                "mean_gain_rank": float(ordered["gain_rank"].mean()),
                "median_gain_rank": float(ordered["gain_rank"].median()),
                "gain_rank_std": float(ordered["gain_rank"].std(ddof=1)),
                "fold_1_gain_rank": float(ordered.loc[ordered["fold_id"].eq(1), "gain_rank"].iloc[0]),
                "fold_5_gain_rank": float(ordered.loc[ordered["fold_id"].eq(5), "gain_rank"].iloc[0]),
                "fold_1_to_5_rank_change": float(
                    ordered.loc[ordered["fold_id"].eq(5), "gain_rank"].iloc[0]
                    - ordered.loc[ordered["fold_id"].eq(1), "gain_rank"].iloc[0]
                ),
                "used_fold_count": int((~ordered["zero_importance"]).sum()),
                "top_rank_fold_count": top_fold_count,
                "consistently_top": top_fold_count >= CONSISTENT_TOP_FOLDS,
                "fold_specific_top": top_fold_count == 1,
                "never_used": bool(ordered["zero_importance"].all()),
            }
        )
    summary = pd.DataFrame(rows).sort_values("feature_position").reset_index(drop=True)
    rank_pivot = per_fold.pivot(index="feature", columns="fold_id", values="gain_rank")
    correlations = rank_pivot.corr(method="spearman")
    correlation_rows = []
    for left in correlations.index:
        for right in correlations.columns:
            if left < right:
                correlation_rows.append(
                    {
                        "left_fold": int(left),
                        "right_fold": int(right),
                        "gain_rank_spearman": float(correlations.loc[left, right]),
                    }
                )
    return summary, pd.DataFrame(correlation_rows)


def summarize_feature_groups(
    per_fold: pd.DataFrame,
    small_features: list[str],
) -> pd.DataFrame:
    if len(small_features) != 100:
        raise ValueError("Expected exactly 100 Small numeric features.")
    all_features = per_fold.sort_values("feature_position")["feature"].drop_duplicates().tolist()
    if len(all_features) != 350 or not set(small_features) <= set(all_features):
        raise ValueError("Small/Medium feature membership is invalid.")
    frame = per_fold.copy()
    frame["feature_group"] = np.where(
        frame["feature"].isin(small_features), "small", "medium_only"
    )
    rows = []
    for (fold_id, name), group in frame.groupby(["fold_id", "feature_group"], sort=True):
        gain_total = float(group["normalized_gain_share"].sum())
        split_total = float(group["normalized_split_share"].sum())
        top_gain = float(group["normalized_gain_share"].nlargest(10).sum())
        rows.append(
            {
                "fold_id": int(fold_id),
                "feature_group": str(name),
                "feature_count": int(len(group)),
                "total_gain_share": gain_total,
                "total_split_share": split_total,
                "mean_gain_share_per_feature": float(group["normalized_gain_share"].mean()),
                "mean_split_share_per_feature": float(group["normalized_split_share"].mean()),
                "nonzero_gain_features": int((~group["zero_gain_importance"]).sum()),
                "nonzero_split_features": int((~group["zero_split_importance"]).sum()),
                "top_10_within_group_gain_concentration": (
                    top_gain / gain_total if gain_total > 0 else np.nan
                ),
                "gain_hhi_within_group": float(
                    ((group["normalized_gain_share"] / gain_total) ** 2).sum()
                )
                if gain_total > 0
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def join_importance_with_fold_ic(
    per_fold: pd.DataFrame,
    fold_ic: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required_ic = {
        "fold_id",
        "feature",
        "absolute_mean_ic",
        "sign_consistency",
        "train_start",
        "train_end",
    }
    missing = sorted(required_ic - set(fold_ic.columns))
    if missing:
        raise ValueError(f"Fold IC summaries are missing columns: {missing}")
    if fold_ic.duplicated(["fold_id", "feature"]).any():
        raise ValueError("Fold IC summaries contain duplicate identities.")
    joined = per_fold.merge(
        fold_ic,
        on=["fold_id", "feature"],
        how="left",
        validate="one_to_one",
    )
    if joined["absolute_mean_ic"].isna().any():
        raise ValueError("Importance and fold-specific IC identities do not align.")
    joined["gain_share_percentile"] = joined.groupby("fold_id")[
        "normalized_gain_share"
    ].rank(pct=True, method="average")
    joined["absolute_ic_percentile"] = joined.groupby("fold_id")[
        "absolute_mean_ic"
    ].rank(pct=True, method="average")
    joined["high_ic_low_gain"] = (
        joined["absolute_ic_percentile"].ge(0.75)
        & joined["gain_share_percentile"].le(0.25)
    )
    joined["low_ic_high_gain"] = (
        joined["absolute_ic_percentile"].le(0.25)
        & joined["gain_share_percentile"].ge(0.75)
    )
    rows = []
    for fold_id, group in joined.groupby("fold_id", sort=True):
        rows.append(
            {
                "fold_id": int(fold_id),
                "train_start": str(group["train_start"].iloc[0]),
                "train_end": str(group["train_end"].iloc[0]),
                "absolute_ic_gain_spearman": float(
                    group["absolute_mean_ic"].corr(
                        group["normalized_gain_share"], method="spearman"
                    )
                ),
                "sign_consistency_gain_spearman": float(
                    group["sign_consistency"].corr(
                        group["normalized_gain_share"], method="spearman"
                    )
                ),
                "high_ic_low_gain_count": int(group["high_ic_low_gain"].sum()),
                "low_ic_high_gain_count": int(group["low_ic_high_gain"].sum()),
            }
        )
    return joined, pd.DataFrame(rows)


def build_research_summary(
    per_fold: pd.DataFrame,
    cross_fold: pd.DataFrame,
    rank_correlations: pd.DataFrame,
    groups: pd.DataFrame,
    ic_relationships: pd.DataFrame,
) -> dict:
    top = cross_fold.sort_values(
        ["mean_normalized_gain_share", "feature"], ascending=[False, True]
    ).head(15)
    return _json_safe({
        "scope": "Saved-model gain/split diagnostics; no fitting, scoring, or causal interpretation.",
        "feature_count": int(len(cross_fold)),
        "fold_count": int(per_fold["fold_id"].nunique()),
        "top_mean_gain_features": top[
            [
                "feature",
                "mean_normalized_gain_share",
                "median_normalized_gain_share",
                "mean_gain_rank",
                "used_fold_count",
            ]
        ].to_dict("records"),
        "never_used_feature_count": int(cross_fold["never_used"].sum()),
        "consistently_top_feature_count": int(cross_fold["consistently_top"].sum()),
        "fold_specific_top_feature_count": int(cross_fold["fold_specific_top"].sum()),
        "gain_rank_correlation": {
            "mean_pairwise": float(rank_correlations["gain_rank_spearman"].mean()),
            "minimum_pairwise": float(rank_correlations["gain_rank_spearman"].min()),
            "fold_1_to_5": float(
                rank_correlations.loc[
                    rank_correlations["left_fold"].eq(1)
                    & rank_correlations["right_fold"].eq(5),
                    "gain_rank_spearman",
                ].iloc[0]
            ),
            "median_absolute_fold_1_to_5_rank_change": float(
                cross_fold["fold_1_to_5_rank_change"].abs().median()
            ),
        },
        "group_summaries": groups.to_dict("records"),
        "ic_importance_relationships": ic_relationships.to_dict("records"),
        "limitations": (
            "Gain and split importance are model-dependent usage diagnostics. "
            "They do not prove causality, standalone alpha, or incremental value."
        ),
    })


def save_model_importance_artifacts(
    analysis_id: str,
    experiment_metadata: dict,
    feature_quality_metadata: dict,
    per_fold: pd.DataFrame,
    cross_fold: pd.DataFrame,
    groups: pd.DataFrame,
    ic_join: pd.DataFrame,
    rank_correlations: pd.DataFrame,
    ic_relationships: pd.DataFrame,
    research_summary: dict,
    *,
    output_root: Path = OUTPUT_ROOT,
    git_state: dict | None = None,
) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", analysis_id):
        raise ValueError("Analysis ID contains unsupported characters.")
    output = output_root / "v6.0" / "model_importance" / analysis_id
    if output.exists():
        raise FileExistsError(f"Model-importance analysis already exists: {output}")
    metadata = {
        "analysis_id": analysis_id,
        "dataset_version": "v6.0",
        "model_experiment_id": experiment_metadata["experiment_id"],
        "feature_quality_analysis_id": feature_quality_metadata["analysis_id"],
        "target": experiment_metadata["target"],
        "ordered_feature_names": experiment_metadata["ordered_feature_names"],
        "model_parameters": experiment_metadata["model_parameters"],
        "importance_types": ["gain", "split"],
        "rank_tie_policy": "average rank; zero-importance features share tied bottom ranks",
        "top_rank_threshold": TOP_RANK_THRESHOLD,
        "consistent_top_folds": CONSISTENT_TOP_FOLDS,
        "folds": experiment_metadata["folds"],
        "git": git_state if git_state is not None else current_git_state(),
        "artifacts": {
            "per_fold_importance": "per_fold_importance.parquet",
            "cross_fold_summary": "cross_fold_summary.parquet",
            "group_summary": "group_summary.parquet",
            "ic_importance": "ic_importance.parquet",
            "rank_correlations": "rank_correlations.parquet",
            "ic_relationships": "ic_relationships.parquet",
            "research_summary": "research_summary.json",
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    per_fold.to_parquet(output / "per_fold_importance.parquet", index=False)
    cross_fold.to_parquet(output / "cross_fold_summary.parquet", index=False)
    groups.to_parquet(output / "group_summary.parquet", index=False)
    ic_join.to_parquet(output / "ic_importance.parquet", index=False)
    rank_correlations.to_parquet(output / "rank_correlations.parquet", index=False)
    ic_relationships.to_parquet(output / "ic_relationships.parquet", index=False)
    with (output / "metadata.json").open("x", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, allow_nan=False)
    with (output / "research_summary.json").open("x", encoding="utf-8") as file:
        json.dump(research_summary, file, indent=2, allow_nan=False)
    with (output / "COMPLETED.json").open("x", encoding="utf-8") as file:
        json.dump({"status": "complete"}, file, indent=2)
    return output


def run_model_importance_analysis(
    analysis_id: str,
    experiment_id: str,
    feature_quality_id: str,
    *,
    output_root: Path = OUTPUT_ROOT,
    data_dir: Path | None = None,
) -> Path:
    data_dir = data_dir or resolve_data_dir("v6.0")
    experiment_dir = output_root / "v6.0" / "experiments" / experiment_id
    quality_dir = output_root / "v6.0" / "feature_quality" / feature_quality_id
    experiment = load_v6_experiment_artifacts(experiment_dir)
    quality = load_feature_quality_artifacts(quality_dir)
    features = experiment["metadata"]["ordered_feature_names"]
    if features != quality["metadata"]["ordered_feature_names"]:
        raise ValueError("Experiment and feature-quality ordered features do not match.")
    official_medium = resolve_v6_numeric_feature_set("medium", data_dir)
    if features != official_medium:
        raise ValueError("Experiment feature list is not official Medium order.")
    small = resolve_v6_numeric_feature_set("small", data_dir)
    per_fold = load_saved_fold_importance(experiment_dir, features)
    cross_fold, rank_correlations = summarize_cross_fold_importance(per_fold)
    groups = summarize_feature_groups(per_fold, small)
    ic_join, ic_relationships = join_importance_with_fold_ic(
        per_fold, quality["fold_summaries"]
    )
    research_summary = build_research_summary(
        per_fold, cross_fold, rank_correlations, groups, ic_relationships
    )
    return save_model_importance_artifacts(
        analysis_id,
        experiment["metadata"],
        quality["metadata"],
        per_fold,
        cross_fold,
        groups,
        ic_join,
        rank_correlations,
        ic_relationships,
        research_summary,
        output_root=output_root,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--feature-quality-id", required=True)
    parser.add_argument("--analysis-id", required=True)
    args = parser.parse_args(argv)
    if args.version != "v6.0":
        raise SystemExit("Model-importance analysis currently supports only v6.0.")
    print(
        run_model_importance_analysis(
            args.analysis_id, args.experiment_id, args.feature_quality_id
        )
    )


if __name__ == "__main__":
    main()
