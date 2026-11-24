"""Memory-conscious access to official Numerai v6 scoring references."""

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.config import get_research_config
from src.paths import resolve_data_dir


REFERENCE_IDENTITY_COLUMNS = ("date", "numerai_ticker", "composite_figi")
V6_NEUTRALIZER_COLUMN_COUNT = 300
V6_VALIDATION_NEUTRALIZER_SHA256 = (
    "1ae870b17d501b8d9ec114ab2096eece342812c424a826f44a772087a82c6c6f"
)


def validation_neutralizer_path(data_dir: Path | None = None) -> Path:
    return (data_dir or resolve_data_dir("v6.0")) / "validation_neutralizer.parquet"


def ordered_neutralizer_columns(
    path: Path,
    expected_count: int = V6_NEUTRALIZER_COLUMN_COUNT,
) -> list[str]:
    names = pq.ParquetFile(path).schema_arrow.names
    columns = [name for name in names if name.startswith("neutralizer_column_")]
    expected = [f"neutralizer_column_{index}" for index in range(expected_count)]
    if columns != expected:
        raise ValueError(
            "Unexpected neutralizer schema or column order: "
            f"found {len(columns)}, expected ordered columns 0–{expected_count - 1}."
        )
    return columns


def approved_research_dates() -> set[str]:
    config = get_research_config("v6.0")
    return set(
        pd.date_range(
            config.research_start,
            config.research_end,
            freq="7D",
        ).strftime("%Y-%m-%d")
    )


def approved_final_holdout_dates() -> set[str]:
    config = get_research_config("v6.0")
    return set(
        pd.date_range(config.holdout_start, config.holdout_end, freq="7D").strftime("%Y-%m-%d")
    )


def load_validation_neutralizers_for_date(
    date: str,
    expected_identity: pd.DataFrame,
    path: Path | None = None,
    expected_count: int = V6_NEUTRALIZER_COLUMN_COUNT,
) -> pd.DataFrame:
    """Load one approved date and require exact identity/order alignment."""
    date = str(date)
    if date not in approved_research_dates():
        raise ValueError(f"Unsupported NCORR research date: {date}")
    missing_identity = sorted(
        set(REFERENCE_IDENTITY_COLUMNS) - set(expected_identity.columns)
    )
    if missing_identity:
        raise ValueError(f"Expected identity is missing columns: {missing_identity}")
    if expected_identity.empty:
        raise ValueError("Expected identity is empty.")
    if set(expected_identity["date"].astype(str)) != {date}:
        raise ValueError("Expected identity contains rows from another date.")
    if expected_identity[["date", "numerai_ticker"]].isna().any().any():
        raise ValueError("Expected date or numerai_ticker contains null values.")
    if expected_identity.duplicated(list(REFERENCE_IDENTITY_COLUMNS)).any():
        raise ValueError("Expected identity contains duplicate rows.")

    path = path or validation_neutralizer_path()
    columns = ordered_neutralizer_columns(path, expected_count)
    reference = pd.read_parquet(
        path,
        columns=[*REFERENCE_IDENTITY_COLUMNS, *columns],
        filters=[("date", "=", date)],
    )
    if reference.empty:
        raise ValueError(f"Neutralizer reference has no rows for {date}.")
    if reference[["date", "numerai_ticker"]].isna().any().any():
        raise ValueError(
            f"Neutralizer date or numerai_ticker contains null values for {date}."
        )
    if reference.duplicated(list(REFERENCE_IDENTITY_COLUMNS)).any():
        raise ValueError(f"Neutralizer reference contains duplicate rows for {date}.")

    expected_keys = expected_identity.loc[:, REFERENCE_IDENTITY_COLUMNS].reset_index(
        drop=True
    ).astype("string")
    reference_keys = reference.loc[:, REFERENCE_IDENTITY_COLUMNS].reset_index(
        drop=True
    ).astype("string")
    if not reference_keys.equals(expected_keys):
        raise ValueError(
            f"Neutralizer rows do not exactly align with OOF identity/order for {date}."
        )
    neutralizers = reference.loc[:, columns]
    if neutralizers.isna().any().any() or not np.isfinite(neutralizers).all().all():
        raise ValueError(f"Neutralizers contain missing or nonfinite values for {date}.")
    neutralizers.index = expected_identity.index
    return neutralizers


def load_final_holdout_neutralizers_for_date(
    date: str,
    expected_identity: pd.DataFrame,
    path: Path | None = None,
    expected_count: int = V6_NEUTRALIZER_COLUMN_COUNT,
) -> pd.DataFrame:
    """Holdout-only wrapper that preserves the Research OOF date restriction."""
    date = str(date)
    if date not in approved_final_holdout_dates():
        raise ValueError(f"Unsupported NCORR final-holdout date: {date}")
    # Perform the same exact schema, identity, and finiteness checks as research access.
    missing_identity = sorted(set(REFERENCE_IDENTITY_COLUMNS) - set(expected_identity.columns))
    if missing_identity or expected_identity.empty:
        raise ValueError("Final-holdout neutralizer identity is incomplete.")
    path = path or validation_neutralizer_path()
    columns = ordered_neutralizer_columns(path, expected_count)
    reference = pd.read_parquet(
        path, columns=[*REFERENCE_IDENTITY_COLUMNS, *columns], filters=[("date", "=", date)]
    )
    expected_keys = expected_identity.loc[:, REFERENCE_IDENTITY_COLUMNS].reset_index(drop=True).astype("string")
    reference_keys = reference.loc[:, REFERENCE_IDENTITY_COLUMNS].reset_index(drop=True).astype("string")
    if reference.empty or reference.duplicated(list(REFERENCE_IDENTITY_COLUMNS)).any() or not reference_keys.equals(expected_keys):
        raise ValueError(f"Final-holdout neutralizers do not align for {date}.")
    neutralizers = reference.loc[:, columns]
    if neutralizers.isna().any().any() or not np.isfinite(neutralizers).all().all():
        raise ValueError(f"Final-holdout neutralizers are invalid for {date}.")
    neutralizers.index = expected_identity.index
    return neutralizers
