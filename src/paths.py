from pathlib import Path

from src.config import DATASET_VERSION, SUPPORTED_DATASET_VERSIONS


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = PROJECT_ROOT / "data"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"


def validate_dataset_version(dataset_version: str) -> None:
    if dataset_version not in SUPPORTED_DATASET_VERSIONS:
        supported = ", ".join(SUPPORTED_DATASET_VERSIONS)
        raise ValueError(
            f"Unsupported dataset version: {dataset_version}. "
            f"Supported versions: {supported}"
        )


def resolve_data_dir(dataset_version: str = DATASET_VERSION) -> Path:
    validate_dataset_version(dataset_version)

    versioned_dir = DATA_ROOT / dataset_version
    if dataset_version == "v5.3" and not versioned_dir.exists():
        return DATA_ROOT

    return versioned_dir


def resolve_output_dir(dataset_version: str = DATASET_VERSION) -> Path:
    validate_dataset_version(dataset_version)

    versioned_dir = OUTPUT_ROOT / dataset_version
    if dataset_version == "v5.3" and not versioned_dir.exists():
        return OUTPUT_ROOT

    return versioned_dir


DATA_DIR = resolve_data_dir()
OUTPUT_DIR = resolve_output_dir()

TRAIN_PATH = DATA_DIR / "train.parquet"
VALIDATION_PATH = DATA_DIR / "validation.parquet"
FEATURES_PATH = DATA_DIR / "features.json"
