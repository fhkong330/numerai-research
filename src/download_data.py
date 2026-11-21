import argparse
import shutil
from pathlib import Path
from uuid import uuid4

from numerapi import NumerAPI

from src.paths import DATA_ROOT, validate_dataset_version


AVAILABLE_DATASETS = {
    "v5.3": {
        "features.json",
        "train.parquet",
        "validation.parquet",
    },
    "v6.0": {
        "features.json",
        "live.parquet",
        "live_benchmark_models.parquet",
        "live_example_preds.csv",
        "live_example_preds.parquet",
        "live_neutralizer.parquet",
        "train.parquet",
        "train_benchmark_models.parquet",
        "train_neutralizer.parquet",
        "validation.parquet",
        "validation_benchmark_models.parquet",
        "validation_example_preds.csv",
        "validation_example_preds.parquet",
        "validation_neutralizer.parquet",
    },
}


def resolve_dataset_file(
    dataset_version: str,
    requested_file: str,
) -> tuple[str, str]:
    validate_dataset_version(dataset_version)

    normalized_file = requested_file.replace("\\", "/")
    version_prefix = f"{dataset_version}/"

    if normalized_file.startswith(version_prefix):
        normalized_file = normalized_file[len(version_prefix):]
    elif "/" in normalized_file:
        raise ValueError(
            f"Dataset file does not belong to {dataset_version}: "
            f"{requested_file}"
        )

    if normalized_file not in AVAILABLE_DATASETS[dataset_version]:
        available = ", ".join(sorted(AVAILABLE_DATASETS[dataset_version]))
        raise ValueError(
            f"Unavailable dataset file for {dataset_version}: "
            f"{requested_file}. Available files: {available}"
        )

    return f"{dataset_version}/{normalized_file}", normalized_file


def download_files(
    dataset_version: str,
    requested_files: list[str],
    *,
    data_root: Path = DATA_ROOT,
    api: NumerAPI | None = None,
) -> dict[str, list[Path]]:
    validate_dataset_version(dataset_version)

    if not requested_files:
        raise ValueError("At least one dataset file must be selected.")

    resolved_files = [
        resolve_dataset_file(dataset_version, requested_file)
        for requested_file in requested_files
    ]
    destination_dir = data_root / dataset_version
    downloaded = []
    skipped = []

    for dataset_path, relative_file in resolved_files:
        destination = destination_dir / relative_file

        if destination.exists():
            print(f"Skipping {dataset_path}: {destination} already exists.")
            skipped.append(destination)
            continue

        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(
            f".{destination.name}.{uuid4().hex}.download"
        )

        if api is None:
            api = NumerAPI()

        print(f"Downloading {dataset_path} to {destination}...")
        try:
            api.download_dataset(dataset_path, str(temporary))

            try:
                with temporary.open("rb") as source:
                    with destination.open("xb") as target:
                        shutil.copyfileobj(source, target)
            except FileExistsError:
                print(
                    f"Skipping {dataset_path}: {destination} appeared "
                    "during download."
                )
                skipped.append(destination)
                continue
        finally:
            temporary.unlink(missing_ok=True)

        downloaded.append(destination)

    return {"downloaded": downloaded, "skipped": skipped}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download selected Numerai dataset files safely."
    )
    parser.add_argument(
        "--version",
        required=True,
        help="Dataset version, for example v5.3 or v6.0.",
    )
    parser.add_argument(
        "--file",
        action="append",
        dest="files",
        required=True,
        help=(
            "Dataset filename or full catalog path. Repeat --file to select "
            "multiple files."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        download_files(args.version, args.files)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
