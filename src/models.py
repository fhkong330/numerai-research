import json
import time
from dataclasses import dataclass
from pathlib import Path

from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from src.config import DEFAULT_LIGHTGBM_PARAMS, DEFAULT_RIDGE_PARAMS


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


@dataclass
class StreamingRidgeModel:
    feature_names: tuple[str, ...]
    means: np.ndarray
    scales: np.ndarray
    coefficients: np.ndarray
    intercept: float
    alpha: float
    batch_size: int
    near_constant_threshold: float
    sample_count: int
    max_batch_rows_seen: int
    training_runtime_seconds: float
    constant_or_near_constant_count: int
    entirely_missing_count: int


def _ridge_numeric_batch(frame: pd.DataFrame, start: int, stop: int) -> np.ndarray:
    values = frame.iloc[start:stop].to_numpy(dtype=np.float32, copy=True)
    if np.isinf(values).any():
        raise ValueError("Ridge features must not contain infinite values.")
    return values


def _ridge_batches(frame: pd.DataFrame, batch_size: int):
    for start in range(0, len(frame), batch_size):
        stop = min(start + batch_size, len(frame))
        yield start, stop, _ridge_numeric_batch(frame, start, stop)


def _standardize_ridge_batch(
    values: np.ndarray,
    means: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    missing = np.isnan(values)
    if missing.any():
        rows, columns = np.nonzero(missing)
        values[rows, columns] = means[columns].astype(np.float32, copy=False)
    values -= means.astype(np.float32, copy=False)
    values /= scales.astype(np.float32, copy=False)
    return values


def train_streaming_ridge(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_names: list[str],
    params: dict | None = None,
) -> StreamingRidgeModel:
    """Fit exact normalized Ridge from bounded sufficient-statistics batches."""
    validate_feature_matrix(X_train, feature_names)
    validate_training_target(y_train, len(X_train), X_train.index)
    settings = DEFAULT_RIDGE_PARAMS.copy()
    if params:
        settings.update(params)
    alpha = float(settings["alpha"])
    batch_size = int(settings["batch_size"])
    threshold = float(settings["near_constant_threshold"])
    if alpha <= 0 or batch_size <= 0 or threshold < 0:
        raise ValueError("Ridge alpha and batch size must be positive.")
    if settings.get("solver") != "batched_sufficient_statistics":
        raise ValueError("Unsupported Ridge solver convention.")

    started = time.perf_counter()
    feature_count = len(feature_names)
    sums = np.zeros(feature_count, dtype=np.float64)
    squared_sums = np.zeros(feature_count, dtype=np.float64)
    counts = np.zeros(feature_count, dtype=np.int64)
    max_batch = 0
    for start, stop, values in _ridge_batches(X_train, batch_size):
        max_batch = max(max_batch, stop - start)
        valid = ~np.isnan(values)
        counts += valid.sum(axis=0)
        sums += np.nansum(values, axis=0, dtype=np.float64)
        squared_sums += np.nansum(values * values, axis=0, dtype=np.float64)
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    variances = np.divide(
        squared_sums,
        counts,
        out=np.zeros_like(squared_sums),
        where=counts > 0,
    ) - means * means
    scales = np.sqrt(np.maximum(variances, 0.0))
    entirely_missing = counts == 0
    constant_or_near_constant = entirely_missing | (scales <= threshold)
    scales[constant_or_near_constant] = 1.0

    target_values = y_train.to_numpy(dtype=np.float64, copy=False)
    intercept = float(target_values.mean())
    gram = np.zeros((feature_count, feature_count), dtype=np.float64)
    cross = np.zeros(feature_count, dtype=np.float64)
    for start, stop, values in _ridge_batches(X_train, batch_size):
        standardized = _standardize_ridge_batch(values, means, scales)
        standardized64 = standardized.astype(np.float64)
        centered_target = target_values[start:stop] - intercept
        gram += standardized64.T @ standardized64
        cross += standardized64.T @ centered_target
    gram.flat[:: feature_count + 1] += len(X_train) * alpha
    coefficients = np.linalg.solve(gram, cross)
    return StreamingRidgeModel(
        feature_names=tuple(feature_names),
        means=means,
        scales=scales,
        coefficients=coefficients,
        intercept=intercept,
        alpha=alpha,
        batch_size=batch_size,
        near_constant_threshold=threshold,
        sample_count=len(X_train),
        max_batch_rows_seen=max_batch,
        training_runtime_seconds=time.perf_counter() - started,
        constant_or_near_constant_count=int(constant_or_near_constant.sum()),
        entirely_missing_count=int(entirely_missing.sum()),
    )


def predict_streaming_ridge(
    model: StreamingRidgeModel,
    X: pd.DataFrame,
) -> pd.Series:
    validate_feature_matrix(X, model.feature_names)
    result = np.empty(len(X), dtype=np.float64)
    for start, stop, values in _ridge_batches(X, model.batch_size):
        standardized = _standardize_ridge_batch(values, model.means, model.scales)
        result[start:stop] = model.intercept + standardized @ model.coefficients
    return pd.Series(result, index=X.index, name="prediction")


def save_streaming_ridge(model: StreamingRidgeModel, path: Path) -> None:
    payload = {
        "format": "numerai_streaming_ridge_v1",
        "feature_names": list(model.feature_names),
        "means": model.means.tolist(),
        "scales": model.scales.tolist(),
        "coefficients": model.coefficients.tolist(),
        "intercept": model.intercept,
        "alpha": model.alpha,
        "batch_size": model.batch_size,
        "near_constant_threshold": model.near_constant_threshold,
        "sample_count": model.sample_count,
        "max_batch_rows_seen": model.max_batch_rows_seen,
        "training_runtime_seconds": model.training_runtime_seconds,
        "constant_or_near_constant_count": model.constant_or_near_constant_count,
        "entirely_missing_count": model.entirely_missing_count,
    }
    with path.open("x", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, allow_nan=False)


def load_streaming_ridge(path: Path) -> StreamingRidgeModel:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if payload.get("format") != "numerai_streaming_ridge_v1":
        raise ValueError("Unsupported Ridge artifact format.")
    return StreamingRidgeModel(
        feature_names=tuple(payload["feature_names"]),
        means=np.asarray(payload["means"], dtype=np.float64),
        scales=np.asarray(payload["scales"], dtype=np.float64),
        coefficients=np.asarray(payload["coefficients"], dtype=np.float64),
        intercept=float(payload["intercept"]),
        alpha=float(payload["alpha"]),
        batch_size=int(payload["batch_size"]),
        near_constant_threshold=float(payload["near_constant_threshold"]),
        sample_count=int(payload["sample_count"]),
        max_batch_rows_seen=int(payload["max_batch_rows_seen"]),
        training_runtime_seconds=float(payload["training_runtime_seconds"]),
        constant_or_near_constant_count=int(
            payload["constant_or_near_constant_count"]
        ),
        entirely_missing_count=int(payload["entirely_missing_count"]),
    )
