import numpy as np
import pandas as pd
from importlib.metadata import version
from numerai_tools.scoring import neutral_correlation
from scipy import stats

from src.config import get_research_config
from src.paths import resolve_data_dir
from src.scoring_references import (
    V6_VALIDATION_NEUTRALIZER_SHA256,
    load_validation_neutralizers_for_date,
    validation_neutralizer_path,
)
from src.validation import V6WalkForwardFold


V6_OOF_COLUMNS = (
    "date",
    "numerai_ticker",
    "composite_figi",
    "fold_id",
    "prediction",
    "target_jupiter_60",
)

V6_SCORING_DEFINITIONS = {
    "NCORR": {
        "api_display_name": "neutral_corr",
        "official_function": "numerai_tools.scoring.neutral_correlation",
        "target": "target_jupiter_60",
        "required_inputs": ["predictions", "v6 neutralizers", "target_jupiter_60"],
        "current_round_role": "payout",
        "historical_validation": True,
        "version": 1,
    },
    "NMMC": {
        "api_display_name": "neutral_mmc",
        "official_function": "numerai_tools.scoring.neutral_contribution",
        "target": "target_jupiter_60",
        "required_inputs": [
            "predictions",
            "v6 neutralizers",
            "classic_v6_nuswmm_pow1 meta model",
            "target_jupiter_60",
        ],
        "current_round_role": "payout",
        "historical_validation": False,
    },
    "NBMC": {
        "api_display_name": "neutral_bmc",
        "official_function": "numerai_tools.scoring.neutral_contribution",
        "target": "target_jupiter_60",
        "required_inputs": [
            "predictions",
            "v6 neutralizers",
            "official shared v6 benchmark reference",
            "target_jupiter_60",
        ],
        "current_round_role": "informational",
        "historical_validation": True,
    },
    "NUMERAI_CORR_JUPITER_60": {
        "api_display_name": None,
        "official_function": "numerai_tools.scoring.numerai_corr",
        "target": "target_jupiter_60",
        "required_inputs": ["predictions", "target_jupiter_60"],
        "current_round_role": "research diagnostic; not a current v6 payout score",
        "historical_validation": True,
    },
}


def v6_metric_availability(data_dir=None) -> dict:
    data_dir = data_dir or resolve_data_dir("v6.0")
    neutralizer = data_dir / "validation_neutralizer.parquet"
    benchmark = data_dir / "validation_benchmark_models.parquet"
    return {
        "NCORR": {
            "implemented": True,
            "inputs_available": neutralizer.exists(),
            "available": neutralizer.exists(),
            "missing_inputs": [] if neutralizer.exists() else [neutralizer.name],
        },
        "NMMC": {
            "implemented": False,
            "inputs_available": False,
            "available": False,
            "missing_inputs": [
                *([] if neutralizer.exists() else [neutralizer.name]),
                "historical classic_v6_nuswmm_pow1 meta-model predictions",
            ],
        },
        "NBMC": {
            "implemented": False,
            "inputs_available": neutralizer.exists() and benchmark.exists(),
            "available": False,
            "missing_inputs": [
                *([] if neutralizer.exists() else [neutralizer.name]),
                *([] if benchmark.exists() else [benchmark.name]),
            ],
        },
        "NUMERAI_CORR_JUPITER_60": {
            "implemented": True,
            "inputs_available": True,
            "available": True,
            "missing_inputs": [],
        },
    }

def numerai_corr(
    predictions: pd.Series,
    target: pd.Series,
) -> float:
    ranked_predictions = (
        predictions.rank(method="average") - 0.5
    ) / predictions.count()

    gaussian_predictions = stats.norm.ppf(ranked_predictions)

    centered_target = target - target.mean()

    predictions_p15 = (
        np.sign(gaussian_predictions)
        * np.abs(gaussian_predictions) ** 1.5
    )

    target_p15 = (
        np.sign(centered_target)
        * np.abs(centered_target) ** 1.5
    )

    return np.corrcoef(
        predictions_p15,
        target_p15,
    )[0, 1]

def calculate_era_correlations(
    predictions,
    target: pd.Series,
    eras: pd.Series,
) -> pd.Series:
    results = pd.DataFrame({
        "prediction": predictions,
        "target": target.to_numpy(),
        "era": eras.to_numpy(),
    })

    era_correlations = results.groupby("era").apply(
        lambda group: numerai_corr(
            group["prediction"],
            group["target"],
        ),
        include_groups=False,
    )

    return era_correlations

def summarize_correlations(
    era_correlations: pd.Series,
) -> dict:
    mean_corr = era_correlations.mean()
    std_corr = era_correlations.std()

    summary = {
        "mean_corr": mean_corr,
        "std_corr": std_corr,
        "median_corr": era_correlations.median(),
        "min_corr": era_correlations.min(),
        "max_corr": era_correlations.max(),
        "sharpe_like": mean_corr / std_corr,
        "positive_era_rate": (era_correlations > 0).mean(),
    }

    return summary


def create_v6_oof_frame(
    identity: pd.DataFrame,
    fold_id: pd.Series,
    predictions: pd.Series,
    target: pd.Series,
) -> pd.DataFrame:
    config = get_research_config("v6.0")
    required_identity = ["date", "numerai_ticker", "composite_figi"]
    missing = [name for name in required_identity if name not in identity.columns]
    if missing:
        raise ValueError(f"OOF identity is missing columns: {missing}")
    for name, series in (
        ("fold_id", fold_id),
        ("prediction", predictions),
        (config.target_column, target),
    ):
        if not isinstance(series, pd.Series):
            raise TypeError(f"{name} must be a pandas Series.")
        if not series.index.equals(identity.index):
            raise ValueError(f"{name} index does not align with OOF identity.")
    return pd.DataFrame(
        {
            "date": identity["date"],
            "numerai_ticker": identity["numerai_ticker"],
            "composite_figi": identity["composite_figi"],
            "fold_id": fold_id,
            "prediction": predictions,
            config.target_column: target,
        },
        index=identity.index,
    )


def validate_v6_oof_predictions(
    oof: pd.DataFrame,
    folds: list[V6WalkForwardFold],
) -> pd.DataFrame:
    config = get_research_config("v6.0")
    missing = [name for name in V6_OOF_COLUMNS if name not in oof.columns]
    if missing:
        raise ValueError(f"OOF predictions are missing columns: {missing}")
    if oof.empty:
        raise ValueError("OOF predictions are empty.")
    if oof[["date", "numerai_ticker"]].isna().any().any():
        raise ValueError("OOF date and numerai_ticker must not contain null values.")
    identity_columns = ["date", "numerai_ticker", "composite_figi"]
    if oof.duplicated(identity_columns).any():
        raise ValueError("Duplicate OOF row identities detected across folds.")
    if not pd.api.types.is_numeric_dtype(oof["prediction"]):
        raise ValueError("OOF predictions must be numeric.")
    if oof["prediction"].isna().any() or not np.isfinite(oof["prediction"]).all():
        raise ValueError("OOF predictions must be present and finite.")
    if not pd.api.types.is_numeric_dtype(oof[config.target_column]):
        raise ValueError("OOF targets must be numeric.")
    if oof[config.target_column].isna().any():
        raise ValueError("Research OOF targets must not be null.")
    if not np.isfinite(oof[config.target_column]).all():
        raise ValueError("Research OOF targets must be finite.")
    if not pd.api.types.is_numeric_dtype(oof["fold_id"]):
        raise ValueError("OOF fold IDs must be integers.")
    fold_values = oof["fold_id"].to_numpy()
    if not np.isfinite(fold_values).all() or not (
        fold_values == fold_values.astype(int)
    ).all():
        raise ValueError("OOF fold IDs must be finite integers.")

    approved_fold_by_date = {}
    for fold in folds:
        for date in fold.validation_dates:
            if date in approved_fold_by_date:
                raise ValueError(f"Research date assigned to multiple folds: {date}")
            approved_fold_by_date[date] = fold.fold
    expected_dates = set(approved_fold_by_date)
    actual_dates = set(oof["date"].astype(str))
    if actual_dates != expected_dates:
        missing_dates = sorted(expected_dates - actual_dates)
        unexpected_dates = sorted(actual_dates - expected_dates)
        raise ValueError(
            "OOF dates do not match approved research folds; "
            f"missing={missing_dates}, unexpected={unexpected_dates}."
        )
    expected_folds = oof["date"].astype(str).map(approved_fold_by_date)
    if not (fold_values.astype(int) == expected_folds.to_numpy()).all():
        raise ValueError("OOF fold assignments do not match approved date folds.")
    return oof.copy()


def calculate_v6_numerai_corr_by_date(oof: pd.DataFrame) -> pd.DataFrame:
    config = get_research_config("v6.0")
    rows = []
    for date, group in oof.groupby("date", sort=True):
        rows.append(
            {
                "date": str(date),
                "fold_id": int(group["fold_id"].iloc[0]),
                "numerai_corr_jupiter_60": numerai_corr(
                    group["prediction"],
                    group[config.target_column],
                ),
                "observation_count": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def calculate_v6_ncorr_by_date(
    oof: pd.DataFrame,
    neutralizer_path=None,
    expected_neutralizer_count: int = 300,
) -> pd.DataFrame:
    """Calculate official NCORR independently for each approved research date."""
    config = get_research_config("v6.0")
    required = {
        "date",
        "numerai_ticker",
        "composite_figi",
        "fold_id",
        "prediction",
        config.target_column,
    }
    missing = sorted(required - set(oof.columns))
    if missing:
        raise ValueError(f"NCORR input is missing columns: {missing}")
    if oof.empty:
        raise ValueError("NCORR input is empty.")
    for name in ("prediction", config.target_column):
        if not pd.api.types.is_numeric_dtype(oof[name]):
            raise ValueError(f"NCORR {name} must be numeric.")
        if oof[name].isna().any() or not np.isfinite(oof[name]).all():
            raise ValueError(f"NCORR {name} must be present and finite.")
    identity_columns = ["date", "numerai_ticker", "composite_figi"]
    if oof.duplicated(identity_columns).any():
        raise ValueError("NCORR input contains duplicate row identities.")

    reference_path = neutralizer_path or validation_neutralizer_path()
    rows = []
    for date, group in oof.groupby("date", sort=True):
        identity = group.loc[:, identity_columns]
        neutralizers = load_validation_neutralizers_for_date(
            str(date),
            identity,
            path=reference_path,
            expected_count=expected_neutralizer_count,
        )
        scoring_index = pd.RangeIndex(len(group))
        predictions = pd.DataFrame(
            {"prediction": group["prediction"].to_numpy()}, index=scoring_index
        )
        targets = pd.Series(
            group[config.target_column].to_numpy(),
            index=scoring_index,
            name=config.target_column,
        )
        neutralizers = neutralizers.set_axis(scoring_index)
        try:
            score = float(
                neutral_correlation(predictions, neutralizers, targets).iloc[0]
            )
        except (AssertionError, np.linalg.LinAlgError, ValueError) as error:
            raise ValueError(
                f"Official NCORR rejected invalid or degenerate inputs for {date}."
            ) from error
        if not np.isfinite(score):
            raise ValueError(f"Official NCORR is nonfinite for date {date}.")
        rows.append(
            {
                "date": str(date),
                "fold_id": int(group["fold_id"].iloc[0]),
                "ncorr": score,
                "observation_count": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def summarize_v6_ncorr(per_date: pd.DataFrame) -> dict:
    required = {"date", "fold_id", "ncorr", "observation_count"}
    missing = sorted(required - set(per_date.columns))
    if missing:
        raise ValueError(f"Per-date NCORR is missing columns: {missing}")
    if per_date.empty or per_date["date"].duplicated().any():
        raise ValueError("Per-date NCORR must contain unique dates.")
    scores = per_date["ncorr"]
    if not pd.api.types.is_numeric_dtype(scores) or not np.isfinite(scores).all():
        raise ValueError("Per-date NCORR contains invalid scores.")
    std = float(scores.std())
    overall = {
            "metric": "NCORR",
            "api_display_name": "neutral_corr",
            "metric_version": 1,
            "numerai_tools_version": version("numerai-tools"),
            "neutralizer_dataset_version": "v6.0",
            "neutralizer_sha256": V6_VALIDATION_NEUTRALIZER_SHA256,
            "date_start": str(per_date["date"].min()),
            "date_end": str(per_date["date"].max()),
            "date_count": int(len(per_date)),
            "observation_count": int(per_date["observation_count"].sum()),
            "mean": float(scores.mean()),
            "std": std,
            "sharpe_like": float(scores.mean() / std) if std else None,
            "positive_date_percentage": float((scores > 0).mean() * 100),
        }
    per_fold = []
    for fold_id, group in per_date.groupby("fold_id", sort=True):
        per_fold.append(
            {
                "fold_id": int(fold_id),
                "date_count": int(len(group)),
                "observation_count": int(group["observation_count"].sum()),
                "mean": float(group["ncorr"].mean()),
            }
        )
    return {"overall": overall, "per_fold": per_fold}


def combine_v6_metric_summaries(
    legacy_corr: dict,
    ncorr: dict,
) -> dict:
    if legacy_corr.get("overall", {}).get("metric") != "NUMERAI_CORR_JUPITER_60":
        raise ValueError("Legacy metric identity must be NUMERAI_CORR_JUPITER_60.")
    if ncorr.get("overall", {}).get("metric") != "NCORR":
        raise ValueError("Neutral metric identity must be NCORR.")
    return {"legacy_corr": legacy_corr, "ncorr": ncorr}


def summarize_v6_oof_corr(per_date: pd.DataFrame) -> dict:
    score_column = "numerai_corr_jupiter_60"
    if per_date[score_column].isna().any():
        raise ValueError("Per-date Numerai Corr contains null values.")
    scores = per_date[score_column]
    std = float(scores.std())
    overall = {
        "metric": "NUMERAI_CORR_JUPITER_60",
        "date_count": int(len(per_date)),
        "observation_count": int(per_date["observation_count"].sum()),
        "mean": float(scores.mean()),
        "std": std,
        "sharpe_like": float(scores.mean() / std) if std else None,
        "positive_date_percentage": float((scores > 0).mean() * 100),
    }
    per_fold = []
    for fold_id, group in per_date.groupby("fold_id", sort=True):
        per_fold.append(
            {
                "fold_id": int(fold_id),
                "date_count": int(len(group)),
                "observation_count": int(group["observation_count"].sum()),
                "mean": float(group[score_column].mean()),
            }
        )
    dated = per_date.assign(year=per_date["date"].str.slice(0, 4).astype(int))
    per_year = []
    for year, group in dated.groupby("year", sort=True):
        per_year.append(
            {
                "year": int(year),
                "date_count": int(len(group)),
                "observation_count": int(group["observation_count"].sum()),
                "mean": float(group[score_column].mean()),
                "std": float(group[score_column].std()),
                "positive_date_percentage": float(
                    (group[score_column] > 0).mean() * 100
                ),
            }
        )
    return {"overall": overall, "per_fold": per_fold, "per_year": per_year}
