import argparse
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

from src.config import (
    PURGE_ERAS,
    TARGET_COLUMN,
    WALK_FORWARD_VALIDATION_ERAS,
    ResearchConfig,
    get_research_config,
)
from src.data import validate_observed_dates
from src.paths import resolve_data_dir


@dataclass(frozen=True)
class V6WalkForwardFold:
    fold: int
    train_dates: tuple[str, ...]
    purge_dates: tuple[str, ...]
    validation_dates: tuple[str, ...]

    def summary(self) -> dict:
        return {
            "fold": self.fold,
            "train_start": self.train_dates[0],
            "train_end": self.train_dates[-1],
            "train_dates": len(self.train_dates),
            "purge_start": self.purge_dates[0],
            "purge_end": self.purge_dates[-1],
            "purge_dates": len(self.purge_dates),
            "validation_start": self.validation_dates[0],
            "validation_end": self.validation_dates[-1],
            "validation_dates": len(self.validation_dates),
        }


def generate_walk_forward_folds(
    validation_start_era: int,
    validation_end_era: int,
) -> list[dict]:
    folds = []

    fold_number = 1
    valid_start = validation_start_era

    while valid_start <= validation_end_era:
        train_end = valid_start - PURGE_ERAS - 1
        purge_start = train_end + 1
        purge_end = valid_start - 1

        valid_end = min(
            valid_start + WALK_FORWARD_VALIDATION_ERAS - 1,
            validation_end_era,
        )

        fold = {
            "fold": fold_number,
            "train_end": train_end,
            "purge_start": purge_start,
            "purge_end": purge_end,
            "valid_start": valid_start,
            "valid_end": valid_end,
        }

        folds.append(fold)

        fold_number += 1
        valid_start += WALK_FORWARD_VALIDATION_ERAS

    return folds

def generate_folds_from_data(
    validation_data,
) -> list[dict]:
    validation_start_era = int(
        validation_data["era"].min()
    )

    validation_end_era = int(
        validation_data["era"].max()
    )

    return generate_walk_forward_folds(
        validation_start_era,
        validation_end_era,
    )

def prepare_walk_forward_fold(
    research_history,
    features: list[str],
    fold: dict,
):
    fold_train = research_history[
        research_history["era"].astype(int)
        <= fold["train_end"]
    ]

    fold_valid = research_history[
        (
            research_history["era"].astype(int)
            >= fold["valid_start"]
        )
        & (
            research_history["era"].astype(int)
            <= fold["valid_end"]
        )
    ]

    X_train = fold_train[features]
    y_train = fold_train[TARGET_COLUMN]

    X_valid = fold_valid[features]
    y_valid = fold_valid[TARGET_COLUMN]
    validation_eras = fold_valid["era"].copy()

    return (
        X_train,
        y_train,
        X_valid,
        y_valid,
        validation_eras,
    )


def generate_v6_walk_forward_folds(
    historical_dates,
    research_validation_dates,
    config: ResearchConfig | None = None,
) -> list[V6WalkForwardFold]:
    config = config or get_research_config("v6.0")
    if config.dataset_version != "v6.0":
        raise ValueError("v6 fold generation requires the v6.0 research config.")

    historical = validate_observed_dates(
        "historical training",
        historical_dates,
        config.historical_start,
        config.historical_end,
        config.expected_historical_dates,
    )
    research = validate_observed_dates(
        "research validation",
        research_validation_dates,
        config.research_start,
        config.research_end,
        config.expected_research_dates,
    )
    if set(historical) & set(research):
        raise ValueError("Historical and research-validation dates overlap.")

    observed = historical + research
    folds = []
    for start in range(0, len(research), config.validation_window_dates):
        validation = tuple(
            research[start : start + config.validation_window_dates]
        )
        validation_position = len(historical) + start
        purge_start = validation_position - config.purge_dates
        if purge_start <= 0:
            raise ValueError("Insufficient history for the configured v6 purge.")
        train = tuple(observed[:purge_start])
        purge = tuple(observed[purge_start:validation_position])

        if set(train) & set(validation):
            raise ValueError("Training and validation dates overlap.")
        if set(purge) & (set(train) | set(validation)):
            raise ValueError("Purge dates overlap training or validation.")
        folds.append(
            V6WalkForwardFold(
                fold=len(folds) + 1,
                train_dates=train,
                purge_dates=purge,
                validation_dates=validation,
            )
        )

    if len(folds) != 5 or len(folds[-1].validation_dates) != 18:
        raise ValueError(
            "Unexpected v6 fold structure; expected five folds with an "
            "18-date final fold."
        )
    return folds


def prepare_v6_walk_forward_fold(research_history, fold: V6WalkForwardFold):
    config = get_research_config("v6.0")
    train = research_history[
        research_history[config.time_column].isin(fold.train_dates)
    ].copy()
    validation = research_history[
        research_history[config.time_column].isin(fold.validation_dates)
    ].copy()
    if train.empty or validation.empty:
        raise ValueError("Fold preparation produced an empty partition.")
    if set(train[config.time_column]) & set(validation[config.time_column]):
        raise ValueError("Prepared v6 training and validation dates overlap.")
    if train[config.target_column].isna().any():
        raise ValueError("Prepared v6 training data contains null targets.")
    if validation[config.target_column].isna().any():
        raise ValueError("Prepared v6 validation data contains null targets.")
    return train, validation


def _parquet_date_types(path: Path) -> tuple[set[str], dict[str, set[str]]]:
    parquet = pq.ParquetFile(path)
    required = {"date", "data_type"}
    missing = sorted(required - set(parquet.schema_arrow.names))
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")

    dates = set()
    types_by_date: dict[str, set[str]] = {}
    for row_group in range(parquet.metadata.num_row_groups):
        values = parquet.read_row_group(
            row_group,
            columns=["date", "data_type"],
        ).to_pydict()
        for date, data_type in zip(values["date"], values["data_type"]):
            dates.add(date)
            types_by_date.setdefault(date, set()).add(data_type)
    return dates, types_by_date


def load_v6_fold_dates(
    data_dir: Path | None = None,
) -> tuple[list[str], list[str]]:
    config = get_research_config("v6.0")
    data_dir = data_dir or resolve_data_dir(config.dataset_version)
    train_dates, train_types = _parquet_date_types(data_dir / "train.parquet")
    validation_dates, validation_types = _parquet_date_types(
        data_dir / "validation.parquet"
    )
    historical = validate_observed_dates(
        "historical training",
        train_dates,
        config.historical_start,
        config.historical_end,
        config.expected_historical_dates,
    )
    if any(values != {"train"} for values in train_types.values()):
        raise ValueError("Unexpected data_type in v6 historical training.")

    boundaries = (
        (
            "research validation",
            config.research_start,
            config.research_end,
            config.expected_research_dates,
            "validation",
        ),
        (
            "holdout gap",
            config.holdout_gap_start,
            config.holdout_gap_end,
            config.expected_holdout_gap_dates,
            "validation",
        ),
        (
            "final holdout",
            config.holdout_start,
            config.holdout_end,
            config.expected_holdout_dates,
            "validation",
        ),
        (
            "immature",
            config.immature_start,
            config.immature_end,
            config.expected_immature_dates,
            "test",
        ),
    )
    approved_dates = set()
    research = []
    for name, start, end, count, expected_type in boundaries:
        selected = sorted(date for date in validation_dates if start <= date <= end)
        selected = validate_observed_dates(name, selected, start, end, count)
        approved_dates.update(selected)
        if any(validation_types[date] != {expected_type} for date in selected):
            raise ValueError(f"Unexpected data_type in {name}.")
        if name == "research validation":
            research = selected
    if approved_dates != validation_dates:
        raise ValueError("v6 validation contains dates outside approved boundaries.")
    return historical, research


def _print_v6_folds(folds: list[V6WalkForwardFold]) -> None:
    print(
        "fold | train start | train end | purge start | purge end | "
        "validation start | validation end | validation dates"
    )
    for fold in folds:
        summary = fold.summary()
        print(
            f"{summary['fold']} | {summary['train_start']} | "
            f"{summary['train_end']} | {summary['purge_start']} | "
            f"{summary['purge_end']} | {summary['validation_start']} | "
            f"{summary['validation_end']} | {summary['validation_dates']}"
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Verify and print configured research folds."
    )
    parser.add_argument("--version", required=True)
    args = parser.parse_args(argv)
    if args.version != "v6.0":
        raise SystemExit("Fold verification currently supports --version v6.0.")
    historical, research = load_v6_fold_dates()
    _print_v6_folds(generate_v6_walk_forward_folds(historical, research))


if __name__ == "__main__":
    main()
