# Data configuration
PURGE_ERAS = 16
TARGET_COLUMN = "target"
DEFAULT_FEATURE_SET = "medium"


# Reproducibility
RANDOM_STATE = 42

# Validation configuration
WALK_FORWARD_VALIDATION_ERAS = 156


# Model configuration
MODEL_NAME = "LightGBM"

DEFAULT_LIGHTGBM_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}