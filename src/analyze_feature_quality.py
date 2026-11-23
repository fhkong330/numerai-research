"""Analyze completed v6 feature-quality artifacts without recomputing IC."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import resolve_v6_numeric_feature_set
from src.feature_quality import load_feature_quality_artifacts
from src.paths import OUTPUT_ROOT, resolve_data_dir


def _distribution(series: pd.Series) -> dict:
    clean = series.dropna().astype(float)
    return {
        "count": int(len(clean)),
        "mean": float(clean.mean()),
        "std": float(clean.std(ddof=1)),
        "min": float(clean.min()),
        "p10": float(clean.quantile(0.10)),
        "p25": float(clean.quantile(0.25)),
        "median": float(clean.median()),
        "p75": float(clean.quantile(0.75)),
        "p90": float(clean.quantile(0.90)),
        "max": float(clean.max()),
    }


def _records(frame: pd.DataFrame, columns: list[str], limit: int = 15) -> list[dict]:
    result = []
    for row in frame.loc[:, columns].head(limit).to_dict("records"):
        result.append(
            {
                key: (int(value) if isinstance(value, np.integer) else float(value))
                if isinstance(value, (np.integer, np.floating)) and pd.notna(value)
                else value
                for key, value in row.items()
            }
        )
    return result


def analyze_feature_quality(
    metadata: dict,
    per_date_ic: pd.DataFrame,
    fold_summaries: pd.DataFrame,
    rolling_ic: pd.DataFrame,
    small_features: list[str],
) -> dict:
    medium = metadata["ordered_feature_names"]
    if len(medium) != 350 or len(small_features) != 100:
        raise ValueError("Expected 350 Medium and 100 Small numeric features.")
    if not set(small_features) <= set(medium):
        raise ValueError("Small features must be a subset of Medium.")
    if per_date_ic.duplicated(["date", "feature"]).any():
        raise ValueError("Per-date IC identities must be unique.")
    latest_fold = int(fold_summaries["fold_id"].max())
    latest = fold_summaries[fold_summaries["fold_id"].eq(latest_fold)].copy()
    if latest["feature"].tolist() != medium:
        raise ValueError("Latest fold summary feature order does not match metadata.")

    rank_frames = []
    for fold_id, group in fold_summaries.groupby("fold_id", sort=True):
        ranked = group.loc[:, ["feature", "absolute_mean_ic"]].copy()
        ranked[f"fold_{int(fold_id)}_rank"] = ranked["absolute_mean_ic"].rank(
            ascending=False, method="average"
        )
        rank_frames.append(ranked.drop(columns="absolute_mean_ic").set_index("feature"))
    ranks = pd.concat(rank_frames, axis=1)
    ranks["fold_1_to_5_change"] = ranks["fold_5_rank"] - ranks["fold_1_rank"]
    ranks["absolute_fold_1_to_5_change"] = ranks["fold_1_to_5_change"].abs()

    latest_train_end = str(latest["train_end"].iloc[0])
    latest_rolling = rolling_ic[
        rolling_ic["fold_id"].eq(latest_fold)
        & rolling_ic["date"].astype(str).eq(latest_train_end)
    ][["feature", "rolling_mean_ic"]].rename(
        columns={"rolling_mean_ic": "recent_52_mean_ic"}
    )
    eligible_dates = sorted(
        per_date_ic.loc[
            per_date_ic["date"].astype(str).le(latest_train_end), "date"
        ].astype(str).unique()
    )
    recent_dates = set(eligible_dates[-52:])
    earlier = per_date_ic[
        per_date_ic["date"].astype(str).isin(set(eligible_dates) - recent_dates)
        & per_date_ic["status"].eq("valid")
    ]
    earlier_means = (
        earlier.groupby("feature", sort=False)["ic"]
        .mean()
        .rename("earlier_mean_ic")
        .reset_index()
    )
    feature_table = (
        latest.merge(latest_rolling, on="feature", validate="one_to_one")
        .merge(earlier_means, on="feature", how="left", validate="one_to_one")
        .merge(ranks.reset_index(), on="feature", validate="one_to_one")
    )
    feature_table["recent_absolute_ic"] = feature_table["recent_52_mean_ic"].abs()
    feature_table["earlier_absolute_ic"] = feature_table["earlier_mean_ic"].abs()
    feature_table["recent_weakening"] = (
        feature_table["recent_absolute_ic"] < feature_table["earlier_absolute_ic"]
    )
    feature_table["recent_sign_reversal"] = (
        np.sign(feature_table["recent_52_mean_ic"])
        != np.sign(feature_table["earlier_mean_ic"])
    )
    feature_table["group"] = np.where(
        feature_table["feature"].isin(small_features), "small", "medium_only"
    )

    def group_summary(group: pd.DataFrame) -> dict:
        return {
            "feature_count": int(len(group)),
            "mean_absolute_mean_ic": float(group["absolute_mean_ic"].mean()),
            "median_absolute_mean_ic": float(group["absolute_mean_ic"].median()),
            "median_sign_consistency": float(group["sign_consistency"].median()),
            "mean_valid_date_coverage": float(group["valid_date_coverage"].mean()),
            "mean_recent_52_absolute_ic": float(group["recent_absolute_ic"].mean()),
            "mean_earlier_absolute_ic": float(group["earlier_absolute_ic"].mean()),
            "recent_weakening_percentage": float(group["recent_weakening"].mean() * 100),
            "recent_sign_reversal_percentage": float(
                group["recent_sign_reversal"].mean() * 100
            ),
            "median_absolute_fold_1_to_5_rank_change": float(
                group["absolute_fold_1_to_5_change"].median()
            ),
        }

    top = feature_table.sort_values(
        ["absolute_mean_ic", "feature"], ascending=[False, True]
    )
    weak = feature_table.sort_values(
        ["absolute_mean_ic", "feature"], ascending=[True, True]
    )
    unstable = feature_table.sort_values(
        ["sign_consistency", "absolute_mean_ic"], ascending=[True, False]
    )
    rank_changes = feature_table.sort_values(
        ["absolute_fold_1_to_5_change", "feature"], ascending=[False, True]
    )
    status_counts = {
        str(key): int(value)
        for key, value in per_date_ic["status"].value_counts(dropna=False).items()
    }
    return {
        "analysis_id": metadata["analysis_id"],
        "scope": (
            "Latest approved Fold 5 training history through 2024-08-30; "
            "descriptive standalone IC, not feature selection or model performance."
        ),
        "feature_count": len(medium),
        "date_count": int(per_date_ic["date"].nunique()),
        "feature_date_rows": int(len(per_date_ic)),
        "date_start": str(per_date_ic["date"].min()),
        "date_end": str(per_date_ic["date"].max()),
        "status_counts": status_counts,
        "valid_ic_percentage": float(
            per_date_ic["status"].eq("valid").mean() * 100
        ),
        "latest_fold": latest_fold,
        "latest_train_end": latest_train_end,
        "mean_signed_ic_distribution": _distribution(feature_table["mean_ic"]),
        "absolute_mean_ic_distribution": _distribution(
            feature_table["absolute_mean_ic"]
        ),
        "sign_consistency_distribution": _distribution(
            feature_table["sign_consistency"]
        ),
        "valid_date_coverage_distribution": _distribution(
            feature_table["valid_date_coverage"]
        ),
        "positive_mean_features": int((feature_table["mean_ic"] > 0).sum()),
        "negative_mean_features": int((feature_table["mean_ic"] < 0).sum()),
        "zero_mean_features": int((feature_table["mean_ic"] == 0).sum()),
        "top_absolute_mean_ic": _records(
            top,
            [
                "feature",
                "group",
                "mean_ic",
                "absolute_mean_ic",
                "sign_consistency",
                "valid_date_coverage",
                "recent_52_mean_ic",
            ],
        ),
        "weakest_absolute_mean_ic": _records(
            weak,
            ["feature", "group", "mean_ic", "absolute_mean_ic", "sign_consistency"],
        ),
        "lowest_sign_consistency": _records(
            unstable,
            ["feature", "group", "mean_ic", "absolute_mean_ic", "sign_consistency"],
        ),
        "fold_rank_stability": {
            "fold_1_to_5_spearman": float(
                ranks["fold_1_rank"].corr(ranks["fold_5_rank"], method="spearman")
            ),
            "median_absolute_rank_change": float(
                ranks["absolute_fold_1_to_5_change"].median()
            ),
            "largest_changes": _records(
                rank_changes,
                [
                    "feature",
                    "group",
                    "fold_1_rank",
                    "fold_5_rank",
                    "fold_1_to_5_change",
                ],
                limit=10,
            ),
        },
        "recent_vs_earlier": {
            "recent_window_dates": 52,
            "recent_start": eligible_dates[-52],
            "recent_end": eligible_dates[-1],
            "mean_absolute_recent_ic": float(feature_table["recent_absolute_ic"].mean()),
            "mean_absolute_earlier_ic": float(feature_table["earlier_absolute_ic"].mean()),
            "weakening_feature_count": int(feature_table["recent_weakening"].sum()),
            "sign_reversal_feature_count": int(
                feature_table["recent_sign_reversal"].sum()
            ),
        },
        "groups": {
            name: group_summary(group)
            for name, group in feature_table.groupby("group", sort=True)
        },
        "interpretation_limit": (
            "Group differences are standalone descriptive associations and do not "
            "establish the cause of Small-versus-Medium model performance."
        ),
    }


def write_analysis(result: dict, output_path: Path) -> Path:
    if output_path.exists():
        raise FileExistsError(f"Feature-quality analysis already exists: {output_path}")
    with output_path.open("x", encoding="utf-8") as file:
        json.dump(result, file, indent=2, allow_nan=False)
    return output_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--analysis-id", required=True)
    args = parser.parse_args(argv)
    if args.version != "v6.0":
        raise SystemExit("Feature-quality analysis currently supports only v6.0.")
    analysis_dir = OUTPUT_ROOT / args.version / "feature_quality" / args.analysis_id
    artifacts = load_feature_quality_artifacts(analysis_dir)
    small = resolve_v6_numeric_feature_set("small", resolve_data_dir("v6.0"))
    result = analyze_feature_quality(
        artifacts["metadata"],
        artifacts["per_date_ic"],
        artifacts["fold_summaries"],
        artifacts["rolling_ic"],
        small,
    )
    path = write_analysis(result, analysis_dir / "analysis_summary.json")
    print(path)


if __name__ == "__main__":
    main()
