from pathlib import Path

from numerapi import NumerAPI


def main() -> None:
    data_dir = Path(__file__).resolve().parent.parent / "data"
    data_dir.mkdir(exist_ok=True)

    api = NumerAPI()
    api.download_dataset("v5.3/train.parquet", str(data_dir / "train.parquet"))
    api.download_dataset("v5.3/features.json", str(data_dir / "features.json"))


if __name__ == "__main__":
    main()
