import argparse
import gc
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import (
    V6_IDENTITY_COLUMNS,
    load_feature_set,
    load_training_data,
    load_validation_data,
    load_v6_historical_training_data,
    load_v6_research_validation_data,
    prepare_datasets,
    prepare_v6_fold_model_data,
    resolve_v6_numeric_feature_set,
    build_research_history,
)
from src.evaluation import (
    calculate_v6_ncorr_by_date,
    calculate_v6_numerai_corr_by_date,
    calculate_era_correlations,
    combine_v6_metric_summaries,
    create_v6_oof_frame,
    summarize_v6_ncorr,
    summarize_v6_oof_corr,
    summarize_correlations,
    validate_v6_oof_predictions,
)
from src.experiment_artifacts import V6ExperimentSpec, save_v6_experiment_artifacts
from src.feature_quality import load_feature_quality_artifacts
from src.feature_selection import (
    POLICY_ID,
    FoldFeatureManifest,
    build_top250_fold_manifests,
)
from src.models import (
    create_lightgbm_model,
    predict_streaming_ridge,
    predict_validated_lightgbm,
    save_streaming_ridge,
    train_model,
    train_streaming_ridge,
    train_validated_lightgbm,
    predict_model,
)
from src.validation import (
    V6WalkForwardFold,
    generate_v6_walk_forward_folds,
    generate_folds_from_data,
    load_v6_fold_dates,
    prepare_walk_forward_fold,
)

from src.config import (
    DEFAULT_LIGHTGBM_PARAMS,
    DEFAULT_RIDGE_PARAMS,
    MODEL_NAME,
    PURGE_ERAS,
    RANDOM_STATE,
    TARGET_COLUMN,
    WALK_FORWARD_VALIDATION_ERAS,
    get_research_config,
)
from src.paths import OUTPUT_DIR, OUTPUT_ROOT, resolve_data_dir


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


def execute_v6_walk_forward(
    historical_training: pd.DataFrame,
    research_validation: pd.DataFrame,
    features: list[str],
    folds: list[V6WalkForwardFold],
    experiment_id: str,
    feature_set_name: str = "medium",
    model_params: dict | None = None,
    neutralizer_path: Path | None = None,
    expected_neutralizer_count: int = 300,
    output_root: Path = OUTPUT_ROOT,
    run_kind: str = "research",
    fold_feature_manifests: dict[int, FoldFeatureManifest] | None = None,
    model_config_name: str = "baseline",
) -> Path:
    """Run the v6 fold loop from protected frames and save completed artifacts."""
    config = get_research_config("v6.0")
    if model_config_name not in {"baseline", "ridge_baseline"}:
        raise ValueError(f"Unsupported model configuration: {model_config_name}")
    if model_config_name == "ridge_baseline" and fold_feature_manifests is not None:
        raise ValueError("The fixed Ridge baseline does not use feature selection.")
    params = (
        DEFAULT_RIDGE_PARAMS.copy()
        if model_config_name == "ridge_baseline"
        else DEFAULT_LIGHTGBM_PARAMS.copy()
    )
    if model_params:
        params.update(model_params)
    if fold_feature_manifests is not None:
        expected_fold_ids = {fold.fold for fold in folds}
        if set(fold_feature_manifests) != expected_fold_ids:
            raise ValueError("Expected one feature-selection manifest per fold.")
        for fold in folds:
            manifest = fold_feature_manifests[fold.fold]
            if (
                manifest.policy_id != POLICY_ID
                or manifest.fold_id != fold.fold
                or manifest.train_end != fold.train_dates[-1]
            ):
                raise ValueError("Fold feature-selection manifest is inconsistent.")
    oof_parts = []

    with tempfile.TemporaryDirectory(prefix="numerai-v6-models-") as temp_dir:
        model_files = {}
        for fold in folds:
            fold_features = (
                list(fold_feature_manifests[fold.fold].model_feature_names)
                if fold_feature_manifests is not None
                else features
            )
            train_data, valid_data = prepare_v6_fold_model_data(
                historical_training,
                research_validation,
                fold_features,
                fold,
            )
            if model_config_name == "ridge_baseline":
                model = train_streaming_ridge(
                    train_data.features,
                    train_data.target,
                    fold_features,
                    params,
                )
                predictions = predict_streaming_ridge(model, valid_data.features)
            else:
                model = create_lightgbm_model(params)
                model = train_validated_lightgbm(
                    model, train_data.features, train_data.target, fold_features
                )
                predictions = predict_validated_lightgbm(model, valid_data.features)
            fold_ids = pd.Series(
                fold.fold,
                index=valid_data.identity.index,
                name="fold_id",
                dtype="int64",
            )
            oof_parts.append(
                create_v6_oof_frame(
                    valid_data.identity,
                    fold_ids,
                    predictions,
                    valid_data.target,
                )
            )
            extension = "json" if model_config_name == "ridge_baseline" else "txt"
            model_path = Path(temp_dir) / f"fold_{fold.fold}.{extension}"
            if model_config_name == "ridge_baseline":
                save_streaming_ridge(model, model_path)
            else:
                model.booster_.save_model(str(model_path))
            model_files[fold.fold] = model_path
            del train_data, valid_data, model, predictions, fold_ids
            gc.collect()

        oof = pd.concat(oof_parts).sort_values(
            [config.time_column, "numerai_ticker"], kind="stable"
        )
        oof = validate_v6_oof_predictions(oof, folds)
        legacy_per_date = calculate_v6_numerai_corr_by_date(oof)
        ncorr_per_date = calculate_v6_ncorr_by_date(
            oof,
            neutralizer_path=neutralizer_path,
            expected_neutralizer_count=expected_neutralizer_count,
        )
        per_date = legacy_per_date.merge(
            ncorr_per_date[["date", "ncorr"]],
            on="date",
            how="left",
            validate="one_to_one",
        )
        metrics = combine_v6_metric_summaries(
            summarize_v6_oof_corr(per_date),
            summarize_v6_ncorr(
                per_date[["date", "fold_id", "ncorr", "observation_count"]]
            ),
        )
        spec = V6ExperimentSpec(
            experiment_id=experiment_id,
            feature_set_name=feature_set_name,
            feature_names=features,
            model_family=("Ridge" if model_config_name == "ridge_baseline" else MODEL_NAME),
            model_parameters=params,
            random_seed=RANDOM_STATE,
            run_kind=run_kind,
            feature_selection_policy_id=(
                POLICY_ID if fold_feature_manifests is not None else None
            ),
            source_feature_quality_analysis_id=(
                next(iter(fold_feature_manifests.values())).source_ic_analysis_id
                if fold_feature_manifests is not None
                else None
            ),
            fold_feature_manifests=(
                tuple(
                    fold_feature_manifests[fold.fold].to_dict() for fold in folds
                )
                if fold_feature_manifests is not None
                else None
            ),
            model_artifact_extension=(
                "json" if model_config_name == "ridge_baseline" else "txt"
            ),
        )
        return save_v6_experiment_artifacts(
            spec,
            folds,
            oof,
            per_date,
            metrics,
            output_root=output_root,
            model_files=model_files,
        )


def run_v6_walk_forward_pipeline(
    experiment_id: str,
    feature_set_name: str = "medium",
    model_params: dict | None = None,
    data_dir: Path | None = None,
    output_root: Path = OUTPUT_ROOT,
    selection_policy_id: str | None = None,
    feature_quality_analysis_id: str | None = None,
    model_config_name: str = "baseline",
) -> Path:
    get_research_config("v6.0")
    data_dir = data_dir or resolve_data_dir("v6.0")
    features = resolve_v6_numeric_feature_set(feature_set_name, data_dir)
    if model_config_name == "ridge_baseline":
        if feature_set_name != "medium":
            raise ValueError("The fixed Ridge baseline requires Medium features.")
        if selection_policy_id or feature_quality_analysis_id:
            raise ValueError("The fixed Ridge baseline does not use feature selection.")
    historical = load_v6_historical_training_data(features, data_dir)
    research = load_v6_research_validation_data(features, data_dir)
    historical_dates, research_dates = load_v6_fold_dates(data_dir)
    folds = generate_v6_walk_forward_folds(historical_dates, research_dates)
    fold_feature_manifests = None
    if selection_policy_id is not None:
        if selection_policy_id != POLICY_ID:
            raise ValueError(f"Unsupported feature-selection policy: {selection_policy_id}")
        if feature_set_name != "medium" or not feature_quality_analysis_id:
            raise ValueError(
                "Top-250 selection requires Medium and a feature-quality analysis ID."
            )
        quality = load_feature_quality_artifacts(
            output_root
            / "v6.0"
            / "feature_quality"
            / feature_quality_analysis_id
        )
        fold_feature_manifests = build_top250_fold_manifests(
            quality["fold_summaries"],
            quality["metadata"],
            features,
            folds,
        )
    return execute_v6_walk_forward(
        historical,
        research,
        features,
        folds,
        experiment_id,
        feature_set_name,
        model_params,
        data_dir / "validation_neutralizer.parquet",
        output_root=output_root,
        fold_feature_manifests=fold_feature_manifests,
        model_config_name=model_config_name,
    )


def _synthetic_v6_smoke_inputs(root: Path):
    config = get_research_config("v6.0")
    historical_dates = pd.date_range(
        config.historical_start,
        periods=config.expected_historical_dates,
        freq="7D",
    ).strftime("%Y-%m-%d").tolist()
    research_dates = pd.date_range(
        config.research_start,
        periods=config.expected_research_dates,
        freq="7D",
    ).strftime("%Y-%m-%d").tolist()
    folds = generate_v6_walk_forward_folds(historical_dates, research_dates)
    features = ["feature_smoke_a", "feature_smoke_b"]

    def rows(dates, rows_per_date):
        records = []
        for date_number, date in enumerate(dates):
            for position in range(rows_per_date):
                x = position / (rows_per_date - 1)
                records.append(
                    {
                        "date": date,
                        "numerai_ticker": f"ticker_{position:03d}",
                        "composite_figi": f"figi_{position:03d}",
                        "data_type": "train" if date < config.research_start else "validation",
                        "target_jupiter_60": (x + 0.07 * date_number) % 1,
                        "feature_smoke_a": x,
                        "feature_smoke_b": np.sin(position + date_number),
                    }
                )
        return pd.DataFrame(records)

    historical = rows(historical_dates, 8)
    research = rows(research_dates, 8)
    reference = research.loc[:, V6_IDENTITY_COLUMNS].copy()
    position = reference["numerai_ticker"].str.slice(-3).astype(int)
    reference["neutralizer_column_0"] = np.sin(position).astype("float32")
    reference["neutralizer_column_1"] = np.cos(position).astype("float32")
    reference_path = root / "synthetic_neutralizer.parquet"
    reference.to_parquet(reference_path, index=False)
    return historical, research, features, folds, reference_path


def run_v6_smoke_pipeline(experiment_id: str, output_root: Path = OUTPUT_ROOT) -> Path:
    with tempfile.TemporaryDirectory(prefix="numerai-v6-smoke-") as temp_dir:
        historical, research, features, folds, reference = _synthetic_v6_smoke_inputs(
            Path(temp_dir)
        )
        return execute_v6_walk_forward(
            historical,
            research,
            features,
            folds,
            experiment_id,
            model_params={"n_estimators": 3, "n_jobs": 1, "verbose": -1},
            neutralizer_path=reference,
            expected_neutralizer_count=2,
            output_root=output_root,
            run_kind="synthetic_smoke",
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the Numerai research pipeline.")
    parser.add_argument("--version", required=True)
    parser.add_argument("--feature-set", default="medium")
    parser.add_argument(
        "--model-config", choices=["baseline", "ridge_baseline"], default="baseline"
    )
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--selection-policy", choices=[POLICY_ID])
    parser.add_argument("--feature-quality-id")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--confirm-full-run", action="store_true")
    args = parser.parse_args(argv)
    if args.version != "v6.0":
        raise SystemExit("The new pipeline CLI currently supports only --version v6.0.")
    if bool(args.selection_policy) != bool(args.feature_quality_id):
        raise SystemExit(
            "Feature selection requires both --selection-policy and --feature-quality-id."
        )
    if args.smoke:
        if args.confirm_full_run:
            raise SystemExit("--confirm-full-run is invalid with --smoke.")
        if args.feature_set != "medium":
            raise SystemExit("Synthetic smoke mode supports --feature-set medium only.")
        if args.selection_policy or args.feature_quality_id:
            raise SystemExit("Synthetic smoke mode does not use real feature selection.")
        if args.model_config != "baseline":
            raise SystemExit("Synthetic CLI smoke mode supports the baseline model only.")
        path = run_v6_smoke_pipeline(args.experiment_id)
    else:
        if not args.confirm_full_run:
            raise SystemExit("Real execution requires both --run and --confirm-full-run.")
        path = run_v6_walk_forward_pipeline(
            args.experiment_id,
            args.feature_set,
            (
                DEFAULT_RIDGE_PARAMS
                if args.model_config == "ridge_baseline"
                else DEFAULT_LIGHTGBM_PARAMS
            ),
            selection_policy_id=args.selection_policy,
            feature_quality_analysis_id=args.feature_quality_id,
            model_config_name=args.model_config,
        )
    print(path)


if __name__ == "__main__":
    main()
