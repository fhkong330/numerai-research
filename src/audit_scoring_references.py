"""Verify local Numerai scoring-reference files without reading target columns."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from src.config import get_research_config
from src.paths import DATA_ROOT, validate_dataset_version


ALIGNMENT_KEYS = ("numerai_ticker", "date", "composite_figi")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def parquet_summary(path: Path) -> dict:
    parquet = pq.ParquetFile(path)
    type_counts = Counter(str(field.type) for field in parquet.schema_arrow)
    display_path = path
    try:
        display_path = path.relative_to(Path.cwd())
    except ValueError:
        pass
    return {
        "path": display_path.as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "row_count": parquet.metadata.num_rows,
        "column_count": parquet.metadata.num_columns,
        "row_group_count": parquet.metadata.num_row_groups,
        "column_types": dict(sorted(type_counts.items())),
        "identity_columns": [
            name for name in ALIGNMENT_KEYS if name in parquet.schema_arrow.names
        ],
        "neutralizer_column_count": sum(
            name.startswith("neutralizer_column_")
            for name in parquet.schema_arrow.names
        ),
        "reference_columns": [
            name
            for name in parquet.schema_arrow.names
            if not name.startswith("neutralizer_column_")
        ],
    }


def verify_row_alignment(
    validation_path: Path,
    reference_path: Path,
    keys: tuple[str, ...] = ALIGNMENT_KEYS,
    batch_size: int = 65_536,
) -> dict:
    validation = pq.ParquetFile(validation_path)
    reference = pq.ParquetFile(reference_path)
    for path, parquet in ((validation_path, validation), (reference_path, reference)):
        missing = sorted(set(keys) - set(parquet.schema_arrow.names))
        if missing:
            raise ValueError(f"Missing alignment keys in {path}: {missing}")

    validation_batches = validation.iter_batches(columns=list(keys), batch_size=batch_size)
    reference_batches = reference.iter_batches(columns=list(keys), batch_size=batch_size)
    rows_compared = 0
    dates = Counter()
    research_dates = set()
    config = get_research_config("v6.0")

    while True:
        validation_batch = next(validation_batches, None)
        reference_batch = next(reference_batches, None)
        if validation_batch is None or reference_batch is None:
            if validation_batch is not None or reference_batch is not None:
                raise ValueError("Validation and reference row counts differ")
            break
        if not validation_batch.equals(reference_batch):
            raise ValueError(
                f"Validation/reference identity mismatch after row {rows_compared}"
            )
        date_index = validation_batch.schema.get_field_index("date")
        for date, count in Counter(validation_batch.column(date_index).to_pylist()).items():
            dates[date] += count
            if config.research_start <= date <= config.research_end:
                research_dates.add(date)
        rows_compared += validation_batch.num_rows

    return {
        "alignment_keys": list(keys),
        "row_order_and_identity_match": True,
        "rows_compared": rows_compared,
        "date_count": len(dates),
        "minimum_date": min(dates) if dates else None,
        "maximum_date": max(dates) if dates else None,
        "research_validation_date_count": len(research_dates),
        "expected_research_validation_date_count": config.expected_research_dates,
        "covers_all_research_validation_dates": (
            len(research_dates) == config.expected_research_dates
        ),
    }


def audit_validation_neutralizer(data_dir: Path) -> dict:
    validation_path = data_dir / "validation.parquet"
    neutralizer_path = data_dir / "validation_neutralizer.parquet"
    for path in (validation_path, neutralizer_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required local file not found: {path}")

    return {
        "dataset_version": "v6.0",
        "reference": parquet_summary(neutralizer_path),
        "alignment": verify_row_alignment(validation_path, neutralizer_path),
        "targets_read": False,
        "statistics": "Exact Parquet metadata, SHA-256, dates, and row-key alignment",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    validate_dataset_version(args.version)
    if args.version != "v6.0":
        raise ValueError("Scoring-reference audit currently supports only v6.0")
    report = audit_validation_neutralizer(DATA_ROOT / args.version)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
