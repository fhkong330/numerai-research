"""Load and strictly validate the frozen Numerai Classic v6 V1 policy."""

import json
from pathlib import Path

from src.config import DEFAULT_LIGHTGBM_PARAMS, RANDOM_STATE, get_research_config
from src.ensemble import APPROVED_ENSEMBLE_POLICIES
from src.feature_selection import POLICY_ID, SELECTED_FEATURE_COUNT, SOURCE_FEATURE_COUNT
from src.paths import PROJECT_ROOT
from src.scoring_references import V6_VALIDATION_NEUTRALIZER_SHA256


V1_POLICY_PATH = PROJECT_ROOT / "docs" / "manifests" / "v6.0_v1_policy.json"
V1_POLICY_ID = "numerai_classic_v6_v1"
MEDIUM_FEATURE_LIST_SHA256 = "1232c4f87c05a4643e546a39f8a48c54f2a1d71ed99852f6679f490106a69af5"


def load_v1_policy(path: Path = V1_POLICY_PATH) -> dict:
    with path.open(encoding="utf-8") as file:
        policy = json.load(file)
    validate_v1_policy(policy)
    return policy


def validate_v1_policy(policy: dict) -> None:
    config = get_research_config("v6.0")
    if policy.get("policy_id") != V1_POLICY_ID:
        raise ValueError("Unexpected frozen V1 policy identity.")
    if policy.get("dataset_version") != config.dataset_version:
        raise ValueError("Frozen V1 dataset version is incompatible.")
    if policy.get("target") != config.target_column or policy.get("time_column") != config.time_column:
        raise ValueError("Frozen V1 target or time column is incompatible.")

    components = policy.get("components", [])
    if [item.get("component_id") for item in components] != [
        "medium_lightgbm", "medium_top250_ic_lightgbm"
    ]:
        raise ValueError("Frozen V1 components are incompatible.")
    if components[0].get("feature_count") != SOURCE_FEATURE_COUNT:
        raise ValueError("V1 Medium component must contain 350 features.")
    if [item.get("source_experiment_id") for item in components] != [
        "v6_medium_baseline_001", "v6_medium_top250_ic_002"
    ]:
        raise ValueError("V1 source experiment identities are incompatible.")
    if any(item.get("feature_list_sha256") != MEDIUM_FEATURE_LIST_SHA256 for item in components):
        raise ValueError("V1 Medium feature identity is incompatible.")
    if components[0].get("feature_order") != "official Medium order" or components[1].get("feature_order") != "selected membership restored to official Medium order":
        raise ValueError("V1 feature ordering is incompatible.")
    selection = components[1].get("feature_selection", {})
    if components[1].get("selected_feature_count") != SELECTED_FEATURE_COUNT:
        raise ValueError("V1 selected component must contain 250 features.")
    if selection.get("policy_id") != POLICY_ID:
        raise ValueError("V1 feature-selection policy is incompatible.")
    if selection.get("as_of_cutoff") != config.research_end:
        raise ValueError("V1 feature-selection cutoff is incompatible.")

    model = policy.get("model", {})
    if model.get("parameters") != DEFAULT_LIGHTGBM_PARAMS:
        raise ValueError("V1 LightGBM parameters are incompatible.")
    if model.get("random_seed") != RANDOM_STATE:
        raise ValueError("V1 random seed is incompatible.")
    if model.get("missing_value_handling") != "LightGBM native NaN handling; no learned imputation":
        raise ValueError("V1 missing-value policy is incompatible.")

    ensemble = policy.get("ensemble", {})
    approved = APPROVED_ENSEMBLE_POLICIES["medium_top250_rank_equal_v1"]
    if ensemble.get("policy_id") != approved.policy_id:
        raise ValueError("V1 ensemble policy is incompatible.")
    if ensemble.get("component_weights") != {
        "medium_lightgbm": 0.5, "medium_top250_ic_lightgbm": 0.5
    }:
        raise ValueError("V1 ensemble weights are incompatible.")
    if ensemble.get("normalization") != "per-date average rank normalized as (rank - 0.5) / n":
        raise ValueError("V1 normalization is incompatible.")
    if not ensemble.get("ridge_excluded"):
        raise ValueError("Ridge must remain excluded from frozen V1.")

    scoring = policy.get("scoring", {})
    primary = scoring.get("primary", {})
    diagnostic = scoring.get("diagnostic", {})
    if primary.get("metric") != "NCORR" or primary.get("api_display_name") != "neutral_corr" or primary.get("metric_version") != 1:
        raise ValueError("V1 primary scoring identity is incompatible.")
    if primary.get("neutralizer_sha256") != V6_VALIDATION_NEUTRALIZER_SHA256:
        raise ValueError("V1 neutralizer identity is incompatible.")
    if diagnostic.get("metric") != "NUMERAI_CORR_JUPITER_60" or diagnostic.get("label") != "Legacy CORR":
        raise ValueError("V1 diagnostic scoring identity is incompatible.")

    purge = policy.get("purge", {})
    if purge.get("dates") != config.purge_dates or purge.get("unit") != "sorted observed weekly dates":
        raise ValueError("V1 purge convention is incompatible.")
    holdout = policy.get("final_holdout", {})
    expected = {
        "gap_start": config.holdout_gap_start,
        "gap_end": config.holdout_gap_end,
        "gap_dates": config.expected_holdout_gap_dates,
        "holdout_start": config.holdout_start,
        "holdout_end": config.holdout_end,
        "holdout_dates": config.expected_holdout_dates,
        "immature_start": config.immature_start,
        "immature_end": config.immature_end,
        "immature_dates": config.expected_immature_dates,
    }
    if any(holdout.get(name) != value for name, value in expected.items()):
        raise ValueError("V1 holdout boundaries are incompatible.")
    if holdout.get("expected_rows") != 365_363 or not holdout.get("targets_sealed"):
        raise ValueError("V1 holdout row count or sealed state is incompatible.")
    training = policy.get("final_training", {})
    if training.get("eligible_data_end") != config.research_end or training.get("feature_ic_recompute_end") != config.research_end:
        raise ValueError("V1 final-training cutoff is incompatible.")

