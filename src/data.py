import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.config import (
    DATASET_VERSION,
    PURGE_ERAS,
    TARGET_COLUMN,
    ResearchConfig,
    get_research_config,
)
from src.paths import (
    FEATURES_PATH,
    TRAIN_PATH,
    VALIDATION_PATH,
    resolve_data_dir,
)


V6_IDENTITY_COLUMNS = ("numerai_ticker", "composite_figi", "date")


@dataclass(frozen=True)
class V6ValidationPartitions:
    research_validation: pd.DataFrame
    holdout_gap: pd.DataFrame
    final_holdout: pd.DataFrame
    immature: pd.DataFrame


@dataclass(frozen=True)
class V6PartitionIndices(V6ValidationPartitions):
    historical_training: pd.DataFrame


@dataclass(frozen=True)
class V6ModelData:
    features: pd.DataFrame
    target: pd.Series
    identity: pd.DataFrame


def load_feature_set(
    feature_set_name: str,
    dataset_version: str = DATASET_VERSION,
) -> list[str]:
    features_path = (
        FEATURES_PATH
        if dataset_version == DATASET_VERSION
        else resolve_data_dir(dataset_version) / "features.json"
    )
    with open(features_path, "r", encoding="utf-8") as f:
        feature_metadata = json.load(f)

    feature_sets = feature_metadata["feature_sets"]

    if feature_set_name not in feature_sets:
        raise ValueError(f"Unknown feature set: {feature_set_name}")

    return feature_sets[feature_set_name]


def resolve_v6_numeric_feature_set(
    feature_set_name: str,
    data_dir: Path | None = None,
) -> list[str]:
    config = get_research_config("v6.0")
    data_dir = data_dir or resolve_data_dir(config.dataset_version)
    official_features = load_feature_set(
        feature_set_name,
        dataset_version=config.dataset_version,
    )
    if feature_set_name == "medium" and len(official_features) != 351:
        raise ValueError(
            "Unexpected v6 medium feature count: "
            f"{len(official_features)}; expected 351."
        )

    parquet = pq.ParquetFile(data_dir / "train.parquet")
    schema = parquet.schema_arrow
    missing = [name for name in official_features if name not in schema.names]
    if missing:
        raise ValueError(
            f"v6 {feature_set_name} features missing from Train schema: {missing}"
        )

    numeric_features = []
    for name in official_features:
        if name == "feature_country":
            continue
        column_type = schema.field(name).type
        if pa.types.is_integer(column_type) or pa.types.is_floating(column_type):
            numeric_features.append(name)

    if feature_set_name == "medium" and len(numeric_features) != 350:
        raise ValueError(
            "Unexpected numeric v6 medium feature count: "
            f"{len(numeric_features)}; expected 350."
        )
    return numeric_features


def prepare_v6_model_data(
    data: pd.DataFrame,
    features: list[str],
) -> V6ModelData:
    config = get_research_config("v6.0")
    required = {*V6_IDENTITY_COLUMNS, config.target_column, *features}
    _require_columns(data, required, "v6 model data")
    if len(features) != len(set(features)):
        raise ValueError("v6 model feature names must be unique.")
    _require_mature_target(data, config, "v6 model data")
    return V6ModelData(
        features=data.loc[:, features].copy(),
        target=data.loc[:, config.target_column].copy(),
        identity=data.loc[:, V6_IDENTITY_COLUMNS].copy(),
    )


def _expected_weekly_dates(start: str, end: str) -> list[str]:
    return pd.date_range(start, end, freq="7D").strftime("%Y-%m-%d").tolist()


def validate_observed_dates(
    name: str,
    observed_dates,
    start: str,
    end: str,
    expected_count: int,
) -> list[str]:
    observed = sorted(
        pd.Series(list(observed_dates), dtype="string").dropna().unique()
    )
    expected = _expected_weekly_dates(start, end)
    if len(expected) != expected_count:
        raise ValueError(f"Invalid configured date count for {name}: {len(expected)}")
    if observed != expected:
        actual_range = (
            f"{observed[0]} through {observed[-1]} ({len(observed)} dates)"
            if observed
            else "no dates"
        )
        raise ValueError(
            f"Unexpected {name} dates: {actual_range}; expected "
            f"{start} through {end} ({expected_count} weekly dates)."
        )
    return observed


def _require_columns(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _require_mature_target(
    frame: pd.DataFrame,
    config: ResearchConfig,
    name: str,
) -> None:
    _require_columns(frame, {config.target_column}, name)
    null_rows = int(frame[config.target_column].isna().sum())
    if null_rows:
        raise ValueError(
            f"{name} contains {null_rows} null {config.target_column} rows; "
            "approved research dates must have mature labels."
        )


def partition_v6_validation_index(
    validation_index: pd.DataFrame,
    config: ResearchConfig | None = None,
) -> V6ValidationPartitions:
    config = config or get_research_config("v6.0")
    required = {*V6_IDENTITY_COLUMNS, "data_type"}
    _require_columns(validation_index, required, "v6 validation index")

    date = validation_index[config.time_column].astype("string")
    research = validation_index[
        date.between(config.research_start, config.research_end)
    ].copy()
    gap = validation_index[
        date.between(config.holdout_gap_start, config.holdout_gap_end)
    ].copy()
    holdout = validation_index[
        date.between(config.holdout_start, config.holdout_end)
    ].copy()
    immature = validation_index[
        date.between(config.immature_start, config.immature_end)
    ].copy()

    partitions = (research, gap, holdout, immature)
    if sum(len(partition) for partition in partitions) != len(validation_index):
        raise ValueError("v6 validation contains rows outside approved partitions.")
    for name, frame, start, end, count, expected_type in (
        (
            "research validation",
            research,
            config.research_start,
            config.research_end,
            config.expected_research_dates,
            "validation",
        ),
        (
            "holdout gap",
            gap,
            config.holdout_gap_start,
            config.holdout_gap_end,
            config.expected_holdout_gap_dates,
            "validation",
        ),
        (
            "final holdout",
            holdout,
            config.holdout_start,
            config.holdout_end,
            config.expected_holdout_dates,
            "validation",
        ),
        (
            "immature",
            immature,
            config.immature_start,
            config.immature_end,
            config.expected_immature_dates,
            "test",
        ),
    ):
        validate_observed_dates(name, frame[config.time_column], start, end, count)
        values = set(frame["data_type"].dropna().astype(str).unique())
        if values != {expected_type}:
            raise ValueError(
                f"Unexpected data_type values for {name}: {sorted(values)}"
            )

    return V6ValidationPartitions(
        research_validation=research,
        holdout_gap=gap,
        final_holdout=holdout,
        immature=immature,
    )


def load_v6_partition_indices(
    data_dir: Path | None = None,
) -> V6PartitionIndices:
    config = get_research_config("v6.0")
    data_dir = data_dir or resolve_data_dir(config.dataset_version)
    columns = [*V6_IDENTITY_COLUMNS, "data_type"]
    train = pd.read_parquet(data_dir / "train.parquet", columns=columns)
    validation = pd.read_parquet(
        data_dir / "validation.parquet",
        columns=columns,
    )
    validate_observed_dates(
        "historical training",
        train[config.time_column],
        config.historical_start,
        config.historical_end,
        config.expected_historical_dates,
    )
    if set(train["data_type"].dropna().astype(str).unique()) != {"train"}:
        raise ValueError("Unexpected data_type values in v6 historical training.")

    split = partition_v6_validation_index(validation, config)
    return V6PartitionIndices(
        historical_training=train,
        research_validation=split.research_validation,
        holdout_gap=split.holdout_gap,
        final_holdout=split.final_holdout,
        immature=split.immature,
    )


def _v6_columns(features: list[str], include_target: bool = True) -> list[str]:
    config = get_research_config("v6.0")
    columns = [*V6_IDENTITY_COLUMNS, "data_type"]
    if include_target:
        columns.append(config.target_column)
    columns.extend(features)
    return list(dict.fromkeys(columns))


def load_v6_historical_training_data(
    features: list[str],
    data_dir: Path | None = None,
) -> pd.DataFrame:
    config = get_research_config("v6.0")
    data_dir = data_dir or resolve_data_dir(config.dataset_version)
    data = pd.read_parquet(
        data_dir / "train.parquet",
        columns=_v6_columns(features),
    )
    validate_observed_dates(
        "historical training",
        data[config.time_column],
        config.historical_start,
        config.historical_end,
        config.expected_historical_dates,
    )
    _require_mature_target(data, config, "v6 historical training")
    return data


def load_v6_research_validation_data(
    features: list[str],
    data_dir: Path | None = None,
) -> pd.DataFrame:
    config = get_research_config("v6.0")
    data_dir = data_dir or resolve_data_dir(config.dataset_version)
    data = pd.read_parquet(
        data_dir / "validation.parquet",
        columns=_v6_columns(features),
        filters=[
            ("data_type", "=", "validation"),
            (config.time_column, ">=", config.research_start),
            (config.time_column, "<=", config.research_end),
        ],
    )
    validate_observed_dates(
        "research validation",
        data[config.time_column],
        config.research_start,
        config.research_end,
        config.expected_research_dates,
    )
    _require_mature_target(data, config, "v6 research validation")
    return data


def build_v6_research_history(
    historical_training: pd.DataFrame,
    research_validation: pd.DataFrame,
) -> pd.DataFrame:
    config = get_research_config("v6.0")
    required = {*V6_IDENTITY_COLUMNS, config.time_column, config.target_column}
    _require_columns(historical_training, required, "v6 historical training")
    _require_columns(research_validation, required, "v6 research validation")
    _require_mature_target(historical_training, config, "v6 historical training")
    _require_mature_target(research_validation, config, "v6 research validation")
    validate_observed_dates(
        "historical training",
        historical_training[config.time_column],
        config.historical_start,
        config.historical_end,
        config.expected_historical_dates,
    )
    validate_observed_dates(
        "research validation",
        research_validation[config.time_column],
        config.research_start,
        config.research_end,
        config.expected_research_dates,
    )

    common_columns = historical_training.columns.intersection(
        research_validation.columns
    )
    history = pd.concat(
        [
            historical_training[common_columns],
            research_validation[common_columns],
        ],
        ignore_index=True,
    )
    return history.sort_values(
        [config.time_column, "numerai_ticker"]
    ).reset_index(drop=True)

def load_training_data(features: list[str]) -> pd.DataFrame:
    columns = ["era", TARGET_COLUMN] + features

    train_data = pd.read_parquet(
        TRAIN_PATH,
        columns=columns,
    )

    return train_data

def load_validation_data(features: list[str]) -> pd.DataFrame:
    columns = ["era", "data_type", TARGET_COLUMN] + features

    validation_data = pd.read_parquet(
        VALIDATION_PATH,
        columns=columns,
    )

    validation_data = validation_data[
        validation_data["data_type"] == "validation"
    ].copy()

    return validation_data

def apply_purge(
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    purge_eras: int = PURGE_ERAS,
) -> pd.DataFrame:
    validation_start_era = int(validation_data["era"].min())
    train_end_era = validation_start_era - purge_eras - 1

    train_purged = train_data[
        train_data["era"].astype(int) <= train_end_era
    ].copy()

    return train_purged

#basic, without walk-forward, no longer use
def prepare_datasets(
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    features: list[str],
):
    train_purged = apply_purge(
        train_data,
        validation_data,
    )

    X_train = train_purged[features]
    y_train = train_purged[TARGET_COLUMN]

    X_valid = validation_data[features]
    y_valid = validation_data[TARGET_COLUMN]
    validation_eras = validation_data["era"].copy()

    return X_train, y_train, X_valid, y_valid, validation_eras

#train+valid=total histroy, use to walk forward split
def build_research_history(
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
) -> pd.DataFrame:
    common_columns = train_data.columns.intersection(
        validation_data.columns
    )

    research_history = pd.concat(
        [
            train_data[common_columns],
            validation_data[common_columns],
        ],
        ignore_index=True,
    )

    research_history = research_history.sort_values(
        "era"
    ).reset_index(drop=True)

    return research_history
