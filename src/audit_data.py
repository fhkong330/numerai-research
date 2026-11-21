import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.paths import DATA_ROOT, OUTPUT_ROOT, validate_dataset_version


FOCUS_TARGETS = (
    "target",
    "target_jupiter_60",
    "target_jupiter_20",
    "target_eleven_4",
)


def evenly_spaced_indices(total: int, count: int) -> list[int]:
    if total <= 0 or count <= 0:
        return []
    if count >= total:
        return list(range(total))
    return sorted(set(np.linspace(0, total - 1, count, dtype=int).tolist()))


def summarize_date_gaps(dates: list[str]) -> dict:
    if len(dates) < 2:
        return {
            "gap_count": 0,
            "gap_days_counts": {},
            "min_gap_days": None,
            "median_gap_days": None,
            "max_gap_days": None,
        }

    parsed = pd.to_datetime(pd.Series(dates), errors="raise")
    gaps = parsed.diff().dropna().dt.days.astype(int)
    counts = gaps.value_counts().sort_index()
    return {
        "gap_count": int(len(gaps)),
        "gap_days_counts": {
            str(int(gap)): int(count) for gap, count in counts.items()
        },
        "min_gap_days": int(gaps.min()),
        "median_gap_days": float(gaps.median()),
        "max_gap_days": int(gaps.max()),
    }


def parquet_schema_summary(parquet: pq.ParquetFile) -> dict:
    columns = []
    for index, name in enumerate(parquet.schema.names):
        column = parquet.schema.column(index)
        columns.append(
            {
                "name": name,
                "physical_type": column.physical_type,
                "logical_type": str(column.logical_type),
            }
        )

    features = [
        column for column in columns if column["name"].startswith("feature_")
    ]
    targets = [
        column for column in columns if column["name"].startswith("target")
    ]
    others = [
        column
        for column in columns
        if not column["name"].startswith(("feature_", "target"))
    ]

    return {
        "row_count": parquet.metadata.num_rows,
        "column_count": parquet.metadata.num_columns,
        "row_group_count": parquet.metadata.num_row_groups,
        "created_by": parquet.metadata.created_by,
        "column_names": [column["name"] for column in columns],
        "feature_columns": [column["name"] for column in features],
        "target_columns": [column["name"] for column in targets],
        "other_columns": others,
        "feature_physical_types": dict(
            sorted(Counter(column["physical_type"] for column in features).items())
        ),
        "target_physical_types": dict(
            sorted(Counter(column["physical_type"] for column in targets).items())
        ),
    }


def _target_result(
    target: str,
    arrow_type: str,
    total_rows: int,
    null_count: int,
    nonnull_dates: set[str],
    value_counts: Counter,
    dates: list[str],
    rows_by_date: Counter,
    nonnull_by_date: Counter,
) -> dict:
    recent_dates = dates[-10:]
    trailing_any_null = 0
    trailing_all_null = 0
    for date in reversed(dates):
        total = rows_by_date[date]
        nonnull = nonnull_by_date[date]
        if nonnull < total:
            trailing_any_null += 1
        else:
            break
    for date in reversed(dates):
        if nonnull_by_date[date] == 0:
            trailing_all_null += 1
        else:
            break

    return {
        "arrow_type": arrow_type,
        "null_count": int(null_count),
        "null_percentage": 100.0 * null_count / total_rows,
        "dates_with_available_target": len(nonnull_dates),
        "first_date_with_available_target": min(nonnull_dates)
        if nonnull_dates
        else None,
        "last_date_with_available_target": max(nonnull_dates)
        if nonnull_dates
        else None,
        "observed_value_counts": {
            str(value): int(count)
            for value, count in sorted(value_counts.items(), key=lambda item: item[0])
        },
        "trailing_dates_with_any_null": trailing_any_null,
        "trailing_dates_all_null": trailing_all_null,
        "recent_date_availability": [
            {
                "date": date,
                "total_rows": int(rows_by_date[date]),
                "nonnull_rows": int(nonnull_by_date[date]),
                "null_rows": int(rows_by_date[date] - nonnull_by_date[date]),
            }
            for date in recent_dates
        ],
    }


def audit_file(
    path: Path,
    expected_features: set[str],
    expected_targets: set[str],
) -> dict:
    parquet = pq.ParquetFile(path)
    schema = parquet_schema_summary(parquet)
    schema_names = set(schema["column_names"])

    missing_columns = {
        "features": sorted(expected_features - schema_names),
        "targets": sorted(expected_targets - schema_names),
        "audit": sorted(
            {
                "date",
                "data_type",
                "numerai_ticker",
                "feature_country",
                *FOCUS_TARGETS,
            }
            - schema_names
        ),
    }
    if any(missing_columns.values()):
        raise ValueError(f"Missing required columns in {path}: {missing_columns}")

    float_features = [
        name
        for index, name in enumerate(parquet.schema.names)
        if name.startswith("feature_")
        and parquet.schema.column(index).physical_type == "FLOAT"
    ]
    integer_features = [
        name
        for index, name in enumerate(parquet.schema.names)
        if name.startswith("feature_")
        and parquet.schema.column(index).physical_type == "INT32"
    ]
    selected_columns = list(
        dict.fromkeys(
            [
                "date",
                "data_type",
                "numerai_ticker",
                "feature_country",
                *FOCUS_TARGETS,
                *float_features,
            ]
        )
    )

    rows_by_date = Counter()
    rows_by_data_type = Counter()
    dates_by_data_type = defaultdict(set)
    date_values = set()
    chronological = True
    previous_date = None
    current_duplicate_date = None
    current_tickers = set()
    duplicate_ticker_rows = 0
    float_null_counts = Counter()
    country_null_count = 0
    country_categories = set()
    target_null_counts = Counter()
    target_nonnull_dates = defaultdict(set)
    target_value_counts = defaultdict(Counter)
    target_nonnull_by_date = defaultdict(Counter)
    alias_compared_rows = 0
    alias_mismatch_rows = 0

    for row_group in range(parquet.metadata.num_row_groups):
        frame = parquet.read_row_group(
            row_group,
            columns=selected_columns,
        ).to_pandas()
        dates = frame["date"].astype("string")

        if not dates.is_monotonic_increasing:
            chronological = False
        if previous_date is not None and len(dates) and previous_date > dates.iloc[0]:
            chronological = False
        if len(dates):
            previous_date = dates.iloc[-1]

        unique_dates = dates.dropna().unique().tolist()
        date_values.update(unique_dates)

        rows_by_date.update(dates.value_counts().to_dict())
        rows_by_data_type.update(frame["data_type"].value_counts().to_dict())
        for data_type, group in frame.groupby("data_type", dropna=False):
            dates_by_data_type[str(data_type)].update(
                group["date"].dropna().astype(str).unique().tolist()
            )

        for date, group in frame.groupby("date", sort=False, dropna=False):
            date = str(date)
            tickers = group["numerai_ticker"].astype(str)
            if date != current_duplicate_date:
                current_duplicate_date = date
                current_tickers = set()
            duplicate_ticker_rows += int(tickers.duplicated().sum())
            duplicate_ticker_rows += sum(
                ticker in current_tickers for ticker in tickers.drop_duplicates()
            )
            current_tickers.update(tickers.tolist())

        float_null_counts.update(
            frame[float_features].isna().sum().astype(int).to_dict()
        )
        country_null_count += int(frame["feature_country"].isna().sum())
        country_categories.update(
            frame["feature_country"].dropna().astype(str).unique().tolist()
        )

        for target in FOCUS_TARGETS:
            series = frame[target]
            target_null_counts[target] += int(series.isna().sum())
            target_value_counts[target].update(series.dropna().value_counts().to_dict())
            available = frame.loc[series.notna(), "date"].astype(str)
            target_nonnull_dates[target].update(available.unique().tolist())
            counts = frame.groupby("date")[target].count().to_dict()
            target_nonnull_by_date[target].update(counts)

        comparable = frame["target"].notna() & frame["target_jupiter_60"].notna()
        alias_compared_rows += int(comparable.sum())
        alias_mismatch_rows += int(
            (
                frame.loc[comparable, "target"]
                != frame.loc[comparable, "target_jupiter_60"]
            ).sum()
        )

    sorted_dates = sorted(date_values)
    date_formats = Counter(
        "YYYY-MM-DD"
        if pd.Series([date]).str.fullmatch(r"\d{4}-\d{2}-\d{2}").iloc[0]
        else "other"
        for date in sorted_dates
    )
    parsed_dates = pd.to_datetime(pd.Series(sorted_dates), errors="raise")
    date_row_counts = pd.Series(
        [rows_by_date[date] for date in sorted_dates],
        dtype="int64",
    )
    target_types = {
        target: str(parquet.schema_arrow.field(target).type)
        for target in FOCUS_TARGETS
    }

    targets = {
        target: _target_result(
            target,
            target_types[target],
            parquet.metadata.num_rows,
            target_null_counts[target],
            target_nonnull_dates[target],
            target_value_counts[target],
            sorted_dates,
            rows_by_date,
            target_nonnull_by_date[target],
        )
        for target in FOCUS_TARGETS
    }

    return {
        "path": f"data/{path.parent.name}/{path.name}",
        "schema": schema,
        "schema_matches_features_json": {
            "features": set(schema["feature_columns"]) == expected_features,
            "targets": set(schema["target_columns"]) == expected_targets,
        },
        "temporal": {
            "date_count": len(sorted_dates),
            "min_date": sorted_dates[0] if sorted_dates else None,
            "max_date": sorted_dates[-1] if sorted_dates else None,
            "date_storage_type": str(parquet.schema_arrow.field("date").type),
            "observed_date_formats": dict(sorted(date_formats.items())),
            "chronologically_ordered": chronological,
            "date_gaps": summarize_date_gaps(sorted_dates),
            "rows_per_date": {
                "min": int(date_row_counts.min()),
                "median": float(date_row_counts.median()),
                "max": int(date_row_counts.max()),
            },
            "duplicate_numerai_ticker_rows_within_date": duplicate_ticker_rows
            if chronological
            else None,
        },
        "splits": {
            "rows_by_data_type": {
                str(key): int(value)
                for key, value in sorted(rows_by_data_type.items())
            },
            "dates_by_data_type": {
                data_type: {
                    "date_count": len(dates),
                    "min_date": min(dates) if dates else None,
                    "max_date": max(dates) if dates else None,
                }
                for data_type, dates in sorted(dates_by_data_type.items())
            },
        },
        "targets": targets,
        "target_alias_check": {
            "columns": ["target", "target_jupiter_60"],
            "compared_nonnull_rows": alias_compared_rows,
            "mismatch_rows": alias_mismatch_rows,
            "identical_where_both_nonnull": alias_mismatch_rows == 0,
        },
        "feature_missingness_exact": {
            "float_features": {
                feature: {
                    "null_count": int(float_null_counts[feature]),
                    "null_percentage": 100.0
                    * float_null_counts[feature]
                    / parquet.metadata.num_rows,
                }
                for feature in float_features
            },
            "feature_country": {
                "null_count": country_null_count,
                "null_percentage": 100.0
                * country_null_count
                / parquet.metadata.num_rows,
                "observed_category_count": len(country_categories),
            },
        },
        "feature_type_columns": {
            "float": float_features,
            "integer": integer_features,
            "country": ["feature_country"],
        },
        "_dates": sorted_dates,
    }


def sample_integer_features(
    files: list[tuple[Path, pq.ParquetFile]],
    integer_features: list[str],
    row_group_count: int,
    rows_per_group: int,
) -> dict:
    feature_min = {feature: None for feature in integer_features}
    feature_max = {feature: None for feature in integer_features}
    feature_values = {feature: set() for feature in integer_features}
    selected_groups = {}
    sampled_rows = 0

    for path, parquet in files:
        groups = evenly_spaced_indices(
            parquet.metadata.num_row_groups,
            row_group_count,
        )
        selected_groups[f"data/{path.parent.name}/{path.name}"] = groups
        for row_group in groups:
            batches = parquet.iter_batches(
                batch_size=rows_per_group,
                row_groups=[row_group],
                columns=integer_features,
            )
            try:
                batch = next(batches)
            except StopIteration:
                continue
            frame = batch.to_pandas()
            sampled_rows += len(frame)
            minima = frame.min(skipna=True)
            maxima = frame.max(skipna=True)
            for feature in integer_features:
                minimum = minima[feature]
                maximum = maxima[feature]
                if pd.notna(minimum):
                    value = int(minimum)
                    current = feature_min[feature]
                    feature_min[feature] = value if current is None else min(current, value)
                if pd.notna(maximum):
                    value = int(maximum)
                    current = feature_max[feature]
                    feature_max[feature] = value if current is None else max(current, value)
                feature_values[feature].update(
                    int(value) for value in frame[feature].dropna().unique().tolist()
                )

    per_feature = {
        feature: {
            "min": feature_min[feature],
            "max": feature_max[feature],
            "distinct_value_count": len(feature_values[feature]),
            "observed_values": sorted(feature_values[feature]),
        }
        for feature in integer_features
    }
    return {
        "scope": "sampled",
        "method": (
            "First bounded batch from evenly spaced row groups in both train "
            "and validation files."
        ),
        "requested_row_groups_per_file": row_group_count,
        "requested_rows_per_group": rows_per_group,
        "selected_row_groups": selected_groups,
        "sampled_rows": sampled_rows,
        "feature_count": len(integer_features),
        "per_feature": per_feature,
        "distinct_value_count_distribution": {
            str(key): int(value)
            for key, value in sorted(
                Counter(
                    result["distinct_value_count"] for result in per_feature.values()
                ).items()
            )
        },
    }


def write_report(report: dict, report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("x", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2, sort_keys=True)


def run_audit(
    dataset_version: str,
    *,
    data_root: Path = DATA_ROOT,
    output_root: Path = OUTPUT_ROOT,
    sample_row_groups: int = 5,
    sample_rows_per_group: int = 2000,
) -> Path:
    validate_dataset_version(dataset_version)
    if dataset_version != "v6.0":
        raise ValueError("The schema audit currently supports only v6.0.")

    data_dir = data_root / dataset_version
    features_path = data_dir / "features.json"
    train_path = data_dir / "train.parquet"
    validation_path = data_dir / "validation.parquet"
    report_path = output_root / dataset_version / "data_audit" / "audit.json"

    if report_path.exists():
        raise FileExistsError(f"Audit report already exists: {report_path}")
    for path in (features_path, train_path, validation_path):
        if not path.exists():
            raise FileNotFoundError(f"Required audit input does not exist: {path}")

    with features_path.open("r", encoding="utf-8") as feature_file:
        metadata = json.load(feature_file)
    expected_features = set(metadata["feature_sets"]["all"])
    expected_targets = set(metadata["targets"])

    train = audit_file(train_path, expected_features, expected_targets)
    validation = audit_file(validation_path, expected_features, expected_targets)

    train_dates = set(train.pop("_dates"))
    validation_dates = set(validation.pop("_dates"))
    train_parquet = pq.ParquetFile(train_path)
    validation_parquet = pq.ParquetFile(validation_path)
    integer_features = train["feature_type_columns"]["integer"]
    integer_sample = sample_integer_features(
        [(train_path, train_parquet), (validation_path, validation_parquet)],
        integer_features,
        sample_row_groups,
        sample_rows_per_group,
    )

    report = {
        "audit_version": 1,
        "dataset_version": dataset_version,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "statistics_scope": {
            "exact": [
                "Parquet schema and footer metadata",
                "Temporal and data_type counts",
                "Focused target availability and distributions",
                "Float-feature and feature_country missingness",
                "feature_country observed category count",
                "Duplicate numerai_ticker checks within date when chronologically ordered",
            ],
            "sampled": [
                "Integer-feature ranges and distinct values",
            ],
        },
        "train": train,
        "validation": validation,
        "cross_file": {
            "shared_date_count": len(train_dates & validation_dates),
            "shared_dates": sorted(train_dates & validation_dates),
            "schemas_identical": train["schema"]["column_names"]
            == validation["schema"]["column_names"],
        },
        "integer_feature_sample": integer_sample,
    }
    write_report(report, report_path)
    return report_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit local Numerai dataset structure without modeling."
    )
    parser.add_argument("--version", required=True, help="Dataset version to audit.")
    parser.add_argument("--sample-row-groups", type=int, default=5)
    parser.add_argument("--sample-rows-per-group", type=int, default=2000)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        report_path = run_audit(
            args.version,
            sample_row_groups=args.sample_row_groups,
            sample_rows_per_group=args.sample_rows_per_group,
        )
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        raise SystemExit(str(error)) from error
    print(f"Saved audit report: {report_path}")


if __name__ == "__main__":
    main()
