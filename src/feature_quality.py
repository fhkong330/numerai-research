"""Leakage-safe, model-independent feature quality diagnostics for Numerai v6."""

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import get_research_config
from src.data import resolve_v6_numeric_feature_set
from src.experiment_artifacts import current_git_state
from src.paths import OUTPUT_ROOT, resolve_data_dir
from src.validation import V6WalkForwardFold, generate_v6_walk_forward_folds


IC_COLUMNS = (
    "date",
    "feature",
    "ic",
    "valid_pairs",
    "total_rows",
    "coverage",
    "status",
)
VALID_IC_STATUS = "valid"


@dataclass(frozen=True)
class FeatureQualitySpec:
    analysis_id: str
    feature_set: str
    ordered_feature_names: list[str]
    min_valid_pairs: int = 3
    rolling_window_dates: int = 52


def approved_v6_feature_quality_folds() -> list[V6WalkForwardFold]:
    """Construct the immutable approved folds without reading target data."""
    config = get_research_config("v6.0")
    historical = pd.date_range(
        config.historical_start, config.historical_end, freq="7D"
    ).strftime("%Y-%m-%d")
    research = pd.date_range(
        config.research_start, config.research_end, freq="7D"
    ).strftime("%Y-%m-%d")
    return generate_v6_walk_forward_folds(historical, research, config)


def _validate_approved_fold(fold: V6WalkForwardFold) -> None:
    approved = approved_v6_feature_quality_folds()
    if not 1 <= fold.fold <= len(approved) or fold != approved[fold.fold - 1]:
        raise ValueError("Feature-quality summaries require an approved v6 fold.")


def calculate_cross_sectional_spearman_ic(
    frame: pd.DataFrame,
    features: list[str],
    *,
    date_column: str = "date",
    target_column: str = "target_jupiter_60",
    min_valid_pairs: int = 3,
) -> pd.DataFrame:
    """Calculate pairwise-complete Spearman IC for exactly one observed date."""
    if min_valid_pairs < 2:
        raise ValueError("min_valid_pairs must be at least 2.")
    if not features or len(features) != len(set(features)):
        raise ValueError("Feature names must be nonempty and unique.")
    required = {date_column, target_column, *features}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Feature IC input is missing columns: {missing}")
    dates = frame[date_column].dropna().astype(str).unique()
    if len(dates) != 1 or frame[date_column].isna().any():
        raise ValueError("Cross-sectional IC input must contain exactly one date.")
    date = str(dates[0])
    total_rows = int(len(frame))
    rows = []
    for feature in features:
        pairs = frame.loc[:, [feature, target_column]].dropna()
        valid_pairs = int(len(pairs))
        coverage = valid_pairs / total_rows if total_rows else 0.0
        status = VALID_IC_STATUS
        ic = np.nan
        if valid_pairs < min_valid_pairs:
            status = "insufficient_pairs"
        elif pairs[feature].nunique(dropna=True) < 2:
            status = "constant_feature"
        elif pairs[target_column].nunique(dropna=True) < 2:
            status = "constant_target"
        else:
            feature_ranks = pairs[feature].rank(method="average")
            target_ranks = pairs[target_column].rank(method="average")
            ic = float(feature_ranks.corr(target_ranks, method="pearson"))
            if not np.isfinite(ic):
                status = "undefined"
                ic = np.nan
        rows.append(
            {
                "date": date,
                "feature": feature,
                "ic": ic,
                "valid_pairs": valid_pairs,
                "total_rows": total_rows,
                "coverage": float(coverage),
                "status": status,
            }
        )
    return pd.DataFrame(rows, columns=IC_COLUMNS)


def _validate_ic_frame(per_date_ic: pd.DataFrame, features: list[str]) -> None:
    missing = sorted(set(IC_COLUMNS) - set(per_date_ic.columns))
    if missing:
        raise ValueError(f"Per-date IC input is missing columns: {missing}")
    if not features or len(features) != len(set(features)):
        raise ValueError("Feature names must be nonempty and unique.")
    if per_date_ic.duplicated(["date", "feature"]).any():
        raise ValueError("Per-date IC contains duplicate date/feature rows.")
    unexpected = sorted(set(per_date_ic["feature"]) - set(features))
    if unexpected:
        raise ValueError(f"Per-date IC contains unexpected features: {unexpected}")
    valid = per_date_ic["status"].eq(VALID_IC_STATUS)
    if per_date_ic.loc[valid, "ic"].isna().any():
        raise ValueError("Valid IC rows must contain a score.")
    if per_date_ic.loc[~valid, "ic"].notna().any():
        raise ValueError("Invalid IC rows must preserve missing IC, not zero.")


def summarize_feature_ic_as_of(
    per_date_ic: pd.DataFrame,
    features: list[str],
    fold: V6WalkForwardFold,
    *,
    rolling_window_dates: int = 52,
) -> dict[str, pd.DataFrame]:
    """Summarize IC using only one approved fold's exact training dates."""
    _validate_approved_fold(fold)
    _validate_ic_frame(per_date_ic, features)
    if rolling_window_dates < 2:
        raise ValueError("rolling_window_dates must be at least 2.")
    observed_dates = set(per_date_ic["date"].astype(str))
    train_dates = set(fold.train_dates)
    forbidden = observed_dates - train_dates
    if forbidden:
        raise ValueError(
            "As-of IC input contains dates outside this fold's training history: "
            f"{sorted(forbidden)[:3]}"
        )
    expected_pairs = {(date, feature) for date in fold.train_dates for feature in features}
    actual_pairs = set(
        zip(per_date_ic["date"].astype(str), per_date_ic["feature"].astype(str))
    )
    missing_pairs = expected_pairs - actual_pairs
    if missing_pairs:
        raise ValueError(
            "As-of IC input is incomplete for the fold training history; "
            f"missing {len(missing_pairs)} date/feature rows."
        )

    ordered = per_date_ic.copy()
    ordered["date"] = ordered["date"].astype(str)
    ordered = ordered.sort_values(["feature", "date"], kind="stable")
    valid = ordered[ordered["status"].eq(VALID_IC_STATUS)].copy()
    summaries = []
    for feature in features:
        values = valid.loc[valid["feature"].eq(feature), "ic"].astype(float)
        mean = float(values.mean()) if len(values) else np.nan
        median = float(values.median()) if len(values) else np.nan
        std = float(values.std(ddof=1)) if len(values) >= 2 else np.nan
        ratio = mean / std if np.isfinite(std) and std > 0 else np.nan
        sign_consistency = np.nan
        if len(values) and np.isfinite(mean) and mean != 0:
            sign_consistency = float((np.sign(values) == np.sign(mean)).mean())
        summaries.append(
            {
                "fold_id": fold.fold,
                "feature": feature,
                "train_start": fold.train_dates[0],
                "train_end": fold.train_dates[-1],
                "training_dates": len(fold.train_dates),
                "valid_dates": int(len(values)),
                "valid_date_coverage": float(len(values) / len(fold.train_dates)),
                "mean_ic": mean,
                "median_ic": median,
                "ic_std": std,
                "absolute_mean_ic": abs(mean) if np.isfinite(mean) else np.nan,
                "mean_to_std_ratio": ratio,
                "sign_consistency": sign_consistency,
            }
        )

    rolling_parts = []
    for feature in features:
        feature_rows = ordered[ordered["feature"].eq(feature)].copy()
        scores = feature_rows["ic"].where(
            feature_rows["status"].eq(VALID_IC_STATUS)
        )
        feature_rows["rolling_mean_ic"] = scores.rolling(
            rolling_window_dates, min_periods=rolling_window_dates
        ).mean()
        feature_rows["rolling_window_dates"] = rolling_window_dates
        feature_rows["fold_id"] = fold.fold
        rolling_parts.append(
            feature_rows[
                [
                    "fold_id",
                    "date",
                    "feature",
                    "rolling_window_dates",
                    "rolling_mean_ic",
                ]
            ]
        )

    annual_rows = []
    training_dates_by_year = pd.Series(fold.train_dates).str.slice(0, 4).value_counts()
    valid["year"] = valid["date"].str.slice(0, 4).astype(int)
    for (feature, year), group in valid.groupby(["feature", "year"], sort=True):
        annual_rows.append(
            {
                "fold_id": fold.fold,
                "feature": feature,
                "year": int(year),
                "training_dates": int(training_dates_by_year[str(year)]),
                "valid_dates": int(len(group)),
                "valid_date_coverage": float(
                    len(group) / training_dates_by_year[str(year)]
                ),
                "mean_ic": float(group["ic"].mean()),
                "median_ic": float(group["ic"].median()),
            }
        )
    return {
        "summary": pd.DataFrame(summaries),
        "rolling": pd.concat(rolling_parts, ignore_index=True),
        "annual": pd.DataFrame(annual_rows),
    }


def build_fold_feature_quality_summaries(
    per_date_ic: pd.DataFrame,
    features: list[str],
    folds: list[V6WalkForwardFold],
    *,
    rolling_window_dates: int = 52,
) -> dict[str, pd.DataFrame]:
    if folds != approved_v6_feature_quality_folds():
        raise ValueError("Feature-quality aggregation requires all approved v6 folds.")
    _validate_ic_frame(per_date_ic, features)
    allowed_dates = set().union(*(set(fold.train_dates) for fold in folds))
    unexpected = set(per_date_ic["date"].astype(str)) - allowed_dates
    if unexpected:
        raise ValueError(
            "Feature-quality observations include non-training dates: "
            f"{sorted(unexpected)[:3]}"
        )
    results = {"summary": [], "rolling": [], "annual": []}
    for fold in folds:
        fold_rows = per_date_ic[
            per_date_ic["date"].astype(str).isin(fold.train_dates)
        ]
        fold_result = summarize_feature_ic_as_of(
            fold_rows,
            features,
            fold,
            rolling_window_dates=rolling_window_dates,
        )
        for name in results:
            results[name].append(fold_result[name])
    return {
        name: pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        for name, parts in results.items()
    }


def _validate_analysis_id(analysis_id: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", analysis_id):
        raise ValueError("Analysis ID contains unsupported characters.")


def save_feature_quality_artifacts(
    spec: FeatureQualitySpec,
    per_date_ic: pd.DataFrame,
    fold_outputs: dict[str, pd.DataFrame],
    folds: list[V6WalkForwardFold],
    *,
    output_root: Path = OUTPUT_ROOT,
    git_state: dict | None = None,
) -> Path:
    _validate_analysis_id(spec.analysis_id)
    if folds != approved_v6_feature_quality_folds():
        raise ValueError("Artifacts require the approved v6 folds.")
    _validate_ic_frame(per_date_ic, spec.ordered_feature_names)
    required_outputs = {"summary", "rolling", "annual"}
    if set(fold_outputs) != required_outputs:
        raise ValueError("Fold outputs must contain summary, rolling, and annual tables.")
    analysis_dir = (
        output_root / "v6.0" / "feature_quality" / spec.analysis_id
    )
    if analysis_dir.exists():
        raise FileExistsError(f"Feature-quality analysis already exists: {analysis_dir}")
    metadata = {
        "analysis_id": spec.analysis_id,
        "dataset_version": "v6.0",
        "target": "target_jupiter_60",
        "feature_set": spec.feature_set,
        "ordered_feature_names": spec.ordered_feature_names,
        "ic_definition": (
            "Per-date cross-sectional Spearman correlation: Pearson correlation "
            "of pairwise-complete average ranks."
        ),
        "missingness_policy": (
            "Drop missing feature/target pairs; preserve undefined IC as missing "
            "with an explicit status, never as zero."
        ),
        "min_valid_pairs": spec.min_valid_pairs,
        "rolling_window_dates": spec.rolling_window_dates,
        "folds": [fold.summary() for fold in folds],
        "git": git_state if git_state is not None else current_git_state(),
        "artifacts": {
            "per_date_ic": "per_date_ic.parquet",
            "fold_summaries": "fold_summaries.parquet",
            "rolling_ic": "rolling_ic.parquet",
            "annual_ic": "annual_ic.parquet",
        },
    }
    analysis_dir.mkdir(parents=True, exist_ok=False)
    per_date_ic.to_parquet(analysis_dir / "per_date_ic.parquet", index=False)
    fold_outputs["summary"].to_parquet(
        analysis_dir / "fold_summaries.parquet", index=False
    )
    fold_outputs["rolling"].to_parquet(
        analysis_dir / "rolling_ic.parquet", index=False
    )
    fold_outputs["annual"].to_parquet(
        analysis_dir / "annual_ic.parquet", index=False
    )
    with (analysis_dir / "metadata.json").open("x", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, allow_nan=False)
    with (analysis_dir / "COMPLETED.json").open("x", encoding="utf-8") as file:
        json.dump({"status": "complete"}, file, indent=2)
    return analysis_dir


def load_feature_quality_artifacts(analysis_dir: Path) -> dict:
    required = {
        "COMPLETED.json",
        "metadata.json",
        "per_date_ic.parquet",
        "fold_summaries.parquet",
        "rolling_ic.parquet",
        "annual_ic.parquet",
    }
    missing = sorted(name for name in required if not (analysis_dir / name).is_file())
    if missing:
        raise FileNotFoundError(f"Feature-quality artifacts are missing: {missing}")
    with (analysis_dir / "metadata.json").open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    return {
        "metadata": metadata,
        "per_date_ic": pd.read_parquet(analysis_dir / "per_date_ic.parquet"),
        "fold_summaries": pd.read_parquet(
            analysis_dir / "fold_summaries.parquet"
        ),
        "rolling_ic": pd.read_parquet(analysis_dir / "rolling_ic.parquet"),
        "annual_ic": pd.read_parquet(analysis_dir / "annual_ic.parquet"),
    }


def run_v6_feature_quality_audit(
    spec: FeatureQualitySpec,
    *,
    data_dir: Path | None = None,
    output_root: Path = OUTPUT_ROOT,
) -> Path:
    """Stream approved training dates only; never read gap or holdout targets."""
    config = get_research_config("v6.0")
    if spec.feature_set != "medium":
        raise ValueError("The approved first feature-quality audit is Medium only.")
    data_dir = data_dir or resolve_data_dir("v6.0")
    resolved = resolve_v6_numeric_feature_set(spec.feature_set, data_dir)
    if resolved != spec.ordered_feature_names:
        raise ValueError("Feature-quality spec does not match official feature order.")
    folds = approved_v6_feature_quality_folds()
    eligible_dates = sorted(set().union(*(set(fold.train_dates) for fold in folds)))
    pieces = []
    columns = [config.time_column, config.target_column, *resolved]
    for date in eligible_dates:
        source = "train.parquet" if date <= config.historical_end else "validation.parquet"
        frame = pd.read_parquet(
            data_dir / source,
            columns=columns,
            filters=[(config.time_column, "=", date)],
        )
        if frame.empty or set(frame[config.time_column].astype(str)) != {date}:
            raise ValueError(f"Missing or unexpected rows for approved date {date}.")
        if frame[config.target_column].isna().any():
            raise ValueError(f"Approved training date {date} has immature targets.")
        pieces.append(
            calculate_cross_sectional_spearman_ic(
                frame,
                resolved,
                min_valid_pairs=spec.min_valid_pairs,
            )
        )
    per_date_ic = pd.concat(pieces, ignore_index=True)
    fold_outputs = build_fold_feature_quality_summaries(
        per_date_ic,
        resolved,
        folds,
        rolling_window_dates=spec.rolling_window_dates,
    )
    return save_feature_quality_artifacts(
        spec, per_date_ic, fold_outputs, folds, output_root=output_root
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--feature-set", choices=["medium"], required=True)
    parser.add_argument("--analysis-id", required=True)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--confirm-feature-audit", action="store_true")
    args = parser.parse_args(argv)
    if args.version != "v6.0":
        raise SystemExit("Feature-quality audits currently support only v6.0.")
    if not args.run or not args.confirm_feature_audit:
        raise SystemExit("A real feature audit requires --run and --confirm-feature-audit.")
    data_dir = resolve_data_dir("v6.0")
    features = resolve_v6_numeric_feature_set(args.feature_set, data_dir)
    path = run_v6_feature_quality_audit(
        FeatureQualitySpec(args.analysis_id, args.feature_set, features),
        data_dir=data_dir,
    )
    print(path)


if __name__ == "__main__":
    main()
