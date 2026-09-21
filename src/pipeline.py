from src.data import (
    load_feature_set,
    load_training_data,
    load_validation_data,
    prepare_datasets,
    build_research_history,
)
from src.evaluation import (
    calculate_era_correlations,
    summarize_correlations,
)
from src.models import (
    create_lightgbm_model,
    train_model,
    predict_model,
)
from src.validation import (
    generate_folds_from_data,
    prepare_walk_forward_fold,
)
import json
import pandas as pd

from src.config import MODEL_NAME, PURGE_ERAS, TARGET_COLUMN, WALK_FORWARD_VALIDATION_ERAS
from src.paths import OUTPUT_DIR


"""
save output:
    model.txt
    era_correlations.csv
    summary.json
"""


def save_experiment(
    experiment_name: str,
    feature_set_name: str,
    model,
    era_correlations,
    summary: dict,
) -> None:
    experiment_dir = OUTPUT_DIR / "experiments" / experiment_name

    #if file exist, stop, avoid covering old version
    if experiment_dir.exists():
        raise FileExistsError(
            f"Experiment already exists: {experiment_name}"
        )

    experiment_dir.mkdir(parents=True, exist_ok=False)

    model.booster_.save_model(
        str(experiment_dir / "model.txt")
    )

    era_correlations.to_csv(
        experiment_dir / "era_correlations.csv",
        header=["corr"],
    )

    experiment_metadata = {
        "experiment_name": experiment_name,
        "feature_set": feature_set_name,
        "target": TARGET_COLUMN,
        "purge_eras": PURGE_ERAS,
        "model": MODEL_NAME,
        "model_params": model.get_params(),
        "metrics": {
            key: float(value)
            for key, value in summary.items()
        },
    }

    with open(
        experiment_dir / "summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            experiment_metadata,
            f,
            indent=4,
        )

#no walk forward, train once, use to select feature and model 
def run_experiment(
    feature_set_name: str,
    model_params: dict | None = None,
    experiment_name: str | None = None,
):
    features = load_feature_set(feature_set_name)

    train_data = load_training_data(features)
    validation_data = load_validation_data(features)

    X_train, y_train, X_valid, y_valid, validation_eras = prepare_datasets(
        train_data,
        validation_data,
        features,
    )

    model = create_lightgbm_model(model_params)

    model = train_model(
        model,
        X_train,
        y_train,
    )

    predictions = predict_model(
        model,
        X_valid,
    )

    era_correlations = calculate_era_correlations(
        predictions,
        y_valid,
        validation_eras,
    )

    summary = summarize_correlations(
        era_correlations,
    )

    if experiment_name is not None:
        save_experiment(
            experiment_name,
            feature_set_name,
            model,
            era_correlations,
            summary,
        )

    return model, era_correlations, summary

def save_walk_forward_experiment(
    experiment_name: str,
    feature_set_name: str,
    model_params: dict,
    era_correlations,
    fold_summaries: list[dict],
    overall_summary: dict,
) -> None:
    experiment_dir = (
        OUTPUT_DIR
        / "experiments"
        / experiment_name
    )

    if experiment_dir.exists():
        raise FileExistsError(
            f"Experiment already exists: {experiment_name}"
        )

    experiment_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    # Save all out-of-sample era correlations
    era_correlations.to_csv(
        experiment_dir / "era_correlations.csv"
    )

    # Save per-fold summaries
    fold_summary_df = pd.DataFrame(
        fold_summaries
    )

    fold_summary_df.to_csv(
        experiment_dir / "fold_summaries.csv",
        index=False,
    )

    # Save experiment metadata and overall metrics
    experiment_metadata = {
        "experiment_name": experiment_name,
        "validation_method": "expanding_walk_forward",
        "feature_set": feature_set_name,
        "target": TARGET_COLUMN,
        "purge_eras": PURGE_ERAS,
        "validation_eras_per_fold": WALK_FORWARD_VALIDATION_ERAS,
        "model": MODEL_NAME,
        "model_params": model_params,
        "metrics": {
            key: float(value)
            for key, value in overall_summary.items()
        },
    }

    with open(
        experiment_dir / "summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            experiment_metadata,
            f,
            indent=4,
        )

def run_walk_forward_experiment(
    feature_set_name: str,
    model_params: dict | None = None,
    experiment_name: str | None = None,
):
    features = load_feature_set(feature_set_name)

    train_data = load_training_data(features)
    validation_data = load_validation_data(features)

    research_history = build_research_history(
        train_data,
        validation_data,
    )

    folds = generate_folds_from_data(
        validation_data,
    )

    all_era_correlations = []
    fold_summaries = []

    for fold in folds:
        print(f"Running Fold {fold['fold']}...")

        (
            X_train,
            y_train,
            X_valid,
            y_valid,
            validation_eras,
        ) = prepare_walk_forward_fold(
            research_history,
            features,
            fold,
        )

        model = create_lightgbm_model(
            model_params
        )

        model = train_model(
            model,
            X_train,
            y_train,
        )

        predictions = predict_model(
            model,
            X_valid,
        )

        era_correlations = calculate_era_correlations(
            predictions,
            y_valid,
            validation_eras,
        )

        fold_summary = summarize_correlations(
            era_correlations
        )

        fold_summary["fold"] = fold["fold"]
        fold_summary["train_end"] = fold["train_end"]
        fold_summary["valid_start"] = fold["valid_start"]
        fold_summary["valid_end"] = fold["valid_end"]

        fold_summaries.append(
            fold_summary
        )

        all_era_correlations.append(
            era_correlations
        )

    combined_era_correlations = pd.concat(
        all_era_correlations
    ).sort_index()

    overall_summary = summarize_correlations(
        combined_era_correlations
    )

    if experiment_name is not None:
        save_walk_forward_experiment(
            experiment_name,
            feature_set_name,
            model.get_params(),
            combined_era_correlations,
            fold_summaries,
            overall_summary,
        )

    return (
        combined_era_correlations,
        fold_summaries,
        overall_summary,
    )