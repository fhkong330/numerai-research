from pathlib import Path

from numerapi import NumerAPI


def main() -> None:
    data_dir = Path(__file__).resolve().parent.parent / "data"
    data_dir.mkdir(exist_ok=True)

    api = NumerAPI()

    datasets = [
        "v5.3/train.parquet",
        "v5.3/validation.parquet",
        "v5.3/features.json",
    ]

    for dataset in datasets:
        filename = dataset.split("/")[-1]
        file_path = data_dir / filename

        if file_path.exists():
            print(f"Skipping {filename}: already exists.")
            continue

        print(f"Downloading {filename}...")
        api.download_dataset(dataset, str(file_path))


if __name__ == "__main__":
    main()