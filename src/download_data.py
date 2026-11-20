from numerapi import NumerAPI

from src.config import DATASET_VERSION
from src.paths import DATA_DIR


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    api = NumerAPI()

    datasets = [
        f"{DATASET_VERSION}/train.parquet",
        f"{DATASET_VERSION}/validation.parquet",
        f"{DATASET_VERSION}/features.json",
    ]

    for dataset in datasets:
        filename = dataset.split("/")[-1]
        file_path = DATA_DIR / filename

        if file_path.exists():
            print(f"Skipping {filename}: already exists.")
            continue

        print(f"Downloading {filename}...")
        api.download_dataset(dataset, str(file_path))


if __name__ == "__main__":
    main()
