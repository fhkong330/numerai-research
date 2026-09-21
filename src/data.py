import json

import pandas as pd

from src.config import PURGE_ERAS, TARGET_COLUMN
from src.paths import FEATURES_PATH, TRAIN_PATH, VALIDATION_PATH

def load_feature_set(feature_set_name: str) -> list[str]:
    with open(FEATURES_PATH, "r", encoding="utf-8") as f:
        feature_metadata = json.load(f)

    feature_sets = feature_metadata["feature_sets"]

    if feature_set_name not in feature_sets:
        raise ValueError(f"Unknown feature set: {feature_set_name}")

    return feature_sets[feature_set_name]

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