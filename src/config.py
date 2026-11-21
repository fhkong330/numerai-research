from dataclasses import dataclass


@dataclass(frozen=True)
class ResearchConfig:
    dataset_version: str
    target_column: str
    time_column: str
    purge_dates: int
    validation_window_dates: int
    historical_start: str
    historical_end: str
    research_start: str
    research_end: str
    holdout_gap_start: str | None
    holdout_gap_end: str | None
    holdout_start: str
    holdout_end: str
    immature_start: str | None
    immature_end: str | None
    expected_historical_dates: int
    expected_research_dates: int
    expected_holdout_gap_dates: int
    expected_holdout_dates: int
    expected_immature_dates: int


V5_3_RESEARCH_CONFIG = ResearchConfig(
    dataset_version="v5.3",
    target_column="target",
    time_column="era",
    purge_dates=16,
    validation_window_dates=156,
    historical_start="0001",
    historical_end="0558",
    research_start="0575",
    research_end="1221",
    holdout_gap_start=None,
    holdout_gap_end=None,
    holdout_start="1222",
    holdout_end="1234",
    immature_start=None,
    immature_end=None,
    expected_historical_dates=558,
    expected_research_dates=647,
    expected_holdout_gap_dates=0,
    expected_holdout_dates=13,
    expected_immature_dates=0,
)

V6_0_RESEARCH_CONFIG = ResearchConfig(
    dataset_version="v6.0",
    target_column="target_jupiter_60",
    time_column="date",
    purge_dates=16,
    validation_window_dates=78,
    historical_start="2003-01-03",
    historical_end="2018-12-28",
    research_start="2019-01-04",
    research_end="2025-04-25",
    holdout_gap_start="2025-05-02",
    holdout_gap_end="2025-08-15",
    holdout_start="2025-08-22",
    holdout_end="2026-08-14",
    immature_start="2026-08-21",
    immature_end="2026-11-13",
    expected_historical_dates=835,
    expected_research_dates=330,
    expected_holdout_gap_dates=16,
    expected_holdout_dates=52,
    expected_immature_dates=13,
)

RESEARCH_CONFIGS = {
    config.dataset_version: config
    for config in (V5_3_RESEARCH_CONFIG, V6_0_RESEARCH_CONFIG)
}


def get_research_config(dataset_version: str) -> ResearchConfig:
    try:
        return RESEARCH_CONFIGS[dataset_version]
    except KeyError as error:
        supported = ", ".join(RESEARCH_CONFIGS)
        raise ValueError(
            f"Unsupported research dataset version: {dataset_version}. "
            f"Supported versions: {supported}"
        ) from error


# Historical v5.3 defaults remain unchanged for the existing pipeline.
DATASET_VERSION = "v5.3"
SUPPORTED_DATASET_VERSIONS = tuple(RESEARCH_CONFIGS)

PURGE_ERAS = V5_3_RESEARCH_CONFIG.purge_dates
TARGET_COLUMN = V5_3_RESEARCH_CONFIG.target_column
DEFAULT_FEATURE_SET = "medium"


# Reproducibility
RANDOM_STATE = 42

# Historical v5.3 validation configuration
WALK_FORWARD_VALIDATION_ERAS = V5_3_RESEARCH_CONFIG.validation_window_dates


# Model configuration
MODEL_NAME = "LightGBM"

DEFAULT_LIGHTGBM_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}
