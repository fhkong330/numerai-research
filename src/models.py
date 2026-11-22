from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from src.config import DEFAULT_LIGHTGBM_PARAMS


def create_lightgbm_model(
    params: dict | None = None,
) -> LGBMRegressor:
    model_params = DEFAULT_LIGHTGBM_PARAMS.copy()

    if params is not None:
        model_params.update(params)

    return LGBMRegressor(**model_params)


def train_model(
    model: LGBMRegressor,
    X_train,
    y_train,
) -> LGBMRegressor:
    model.fit(X_train, y_train)

    return model


def predict_model(
    model: LGBMRegressor,
    X,
):
    return model.predict(X)


def validate_feature_matrix(
    features: pd.DataFrame,
    expected_features: list[str] | tuple[str, ...],
) -> None:
    if not isinstance(features, pd.DataFrame):
        raise TypeError("Features must be a pandas DataFrame.")
    expected = list(expected_features)
    if len(expected) != len(set(expected)):
        raise ValueError("Expected feature names must be unique.")
    missing = [name for name in expected if name not in features.columns]
    if missing:
        raise ValueError(f"Missing selected feature columns: {missing}")
    actual = features.columns.tolist()
    if actual != expected:
        raise ValueError(
            "Feature columns or ordering do not match the selected feature set."
        )
    nonnumeric = [
        name for name in expected if not is_numeric_dtype(features[name].dtype)
    ]
    if nonnumeric:
        raise ValueError(f"Selected feature columns must be numeric: {nonnumeric}")


def validate_training_target(
    target: pd.Series,
    expected_length: int,
    expected_index: pd.Index | None = None,
) -> None:
    if not isinstance(target, pd.Series):
        raise TypeError("Training target must be a pandas Series.")
    if len(target) != expected_length:
        raise ValueError("Training features and target have different lengths.")
    if expected_index is not None and not target.index.equals(expected_index):
        raise ValueError("Training features and target indexes do not align.")
    if not is_numeric_dtype(target.dtype):
        raise ValueError("Training target must be numeric.")
    if target.isna().any():
        raise ValueError("Training target must not contain null values.")
    if not np.isfinite(target.to_numpy()).all():
        raise ValueError("Training target must contain only finite values.")


def train_validated_lightgbm(
    model: LGBMRegressor,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_names: list[str],
) -> LGBMRegressor:
    validate_feature_matrix(X_train, feature_names)
    validate_training_target(y_train, len(X_train), X_train.index)
    model.fit(X_train, y_train)
    model.numerai_feature_names_ = tuple(feature_names)
    return model


def predict_validated_lightgbm(
    model: LGBMRegressor,
    X: pd.DataFrame,
) -> pd.Series:
    if not hasattr(model, "numerai_feature_names_"):
        raise ValueError("Model was not fitted with train_validated_lightgbm.")
    validate_feature_matrix(X, model.numerai_feature_names_)
    predictions = model.predict(X)
    return pd.Series(predictions, index=X.index, name="prediction")
