import numpy as np
import pandas as pd
from scipy import stats

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