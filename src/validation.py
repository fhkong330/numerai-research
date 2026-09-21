from src.config import (
    PURGE_ERAS,
    TARGET_COLUMN,
    WALK_FORWARD_VALIDATION_ERAS,
)


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