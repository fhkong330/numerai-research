"""One fixed, leakage-safe v6 Medium feature-selection policy."""

import hashlib
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

from src.feature_quality import approved_v6_feature_quality_folds
from src.validation import V6WalkForwardFold


POLICY_ID = "medium_top250_abs_mean_ic_v1"
SELECTED_FEATURE_COUNT = 250
SOURCE_FEATURE_COUNT = 350


def feature_list_sha256(features: list[str] | tuple[str, ...]) -> str:
    return hashlib.sha256("\n".join(features).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FoldFeatureManifest:
    policy_id: str
    fold_id: int
    selected_feature_count: int
    source_feature_count: int
    train_start: str
    train_end: str
    training_dates: int
    source_ic_analysis_id: str
    source_feature_list_sha256: str
    selected_ranked_features_sha256: str
    selected_model_features_sha256: str
    selected_ranked: tuple[dict, ...]
    model_feature_names: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "policy_id": self.policy_id,
            "fold_id": self.fold_id,
            "selected_feature_count": self.selected_feature_count,
            "source_feature_count": self.source_feature_count,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "training_dates": self.training_dates,
            "source_ic_analysis_id": self.source_ic_analysis_id,
            "source_feature_list_sha256": self.source_feature_list_sha256,
            "selected_ranked_features_sha256": self.selected_ranked_features_sha256,
            "selected_model_features_sha256": self.selected_model_features_sha256,
            "selected_ranked": list(self.selected_ranked),
            "model_feature_names": list(self.model_feature_names),
        }


def _validate_approved_fold(fold: V6WalkForwardFold) -> None:
    approved = approved_v6_feature_quality_folds()
    if not 1 <= fold.fold <= len(approved) or fold != approved[fold.fold - 1]:
        raise ValueError("Feature selection requires an exact approved v6 fold.")


def _validate_source_metadata(
    metadata: dict,
    official_medium_features: list[str],
) -> None:
    if metadata.get("dataset_version") != "v6.0":
        raise ValueError("Feature selection requires dataset version v6.0.")
    if metadata.get("target") != "target_jupiter_60":
        raise ValueError("Feature selection requires target_jupiter_60.")
    if metadata.get("feature_set") != "medium":
        raise ValueError("Feature selection requires the official Medium feature set.")
    if metadata.get("ordered_feature_names") != official_medium_features:
        raise ValueError("Feature-quality feature identity or order does not match Medium.")
    if len(official_medium_features) != SOURCE_FEATURE_COUNT:
        raise ValueError("Official Medium must contain exactly 350 numeric features.")
    expected_folds = [fold.summary() for fold in approved_v6_feature_quality_folds()]
    if metadata.get("folds") != expected_folds:
        raise ValueError("Feature-quality metadata does not contain approved folds.")
    if not metadata.get("analysis_id"):
        raise ValueError("Feature-quality metadata is missing its analysis identity.")


def select_fold_top250(
    fold_summary: pd.DataFrame,
    fold: V6WalkForwardFold,
    source_metadata: dict,
    official_medium_features: list[str],
) -> FoldFeatureManifest:
    """Select Top-250 by absolute training mean IC for exactly one fold."""
    _validate_approved_fold(fold)
    _validate_source_metadata(source_metadata, official_medium_features)
    required = {
        "fold_id",
        "feature",
        "train_start",
        "train_end",
        "training_dates",
        "valid_dates",
        "mean_ic",
        "absolute_mean_ic",
    }
    missing = sorted(required - set(fold_summary.columns))
    if missing:
        raise ValueError(f"Fold selection summary is missing columns: {missing}")
    if fold_summary.duplicated(["fold_id", "feature"]).any():
        raise ValueError("Fold selection summary contains duplicate identities.")
    if set(fold_summary["fold_id"].dropna().astype(int)) != {fold.fold}:
        raise ValueError("Fold selection input contains later or mismatched fold summaries.")
    if len(fold_summary) != SOURCE_FEATURE_COUNT:
        raise ValueError("Fold selection input must contain exactly 350 feature rows.")
    if set(fold_summary["feature"]) != set(official_medium_features):
        raise ValueError("Fold selection input has an incomplete or mismatched feature list.")
    expected_fields = {
        "train_start": fold.train_dates[0],
        "train_end": fold.train_dates[-1],
        "training_dates": len(fold.train_dates),
    }
    for column, expected in expected_fields.items():
        values = set(fold_summary[column].dropna())
        if values != {expected}:
            raise ValueError(
                f"Fold selection {column} does not match the approved training cutoff."
            )
    frame = fold_summary.copy()
    finite_mean = np.isfinite(pd.to_numeric(frame["mean_ic"], errors="coerce"))
    finite_absolute = np.isfinite(
        pd.to_numeric(frame["absolute_mean_ic"], errors="coerce")
    )
    mean_values = pd.to_numeric(frame["mean_ic"], errors="coerce")
    absolute_values = pd.to_numeric(frame["absolute_mean_ic"], errors="coerce")
    finite_rows = finite_mean & finite_absolute
    if not np.allclose(
        absolute_values[finite_rows],
        mean_values[finite_rows].abs(),
        rtol=0.0,
        atol=1e-15,
    ):
        raise ValueError("Fold selection IC statistics are internally inconsistent.")
    valid_dates = pd.to_numeric(frame["valid_dates"], errors="coerce")
    if valid_dates.isna().any() or valid_dates.lt(0).any() or valid_dates.gt(
        len(fold.train_dates)
    ).any():
        raise ValueError("Fold selection valid-date counts are inconsistent.")
    frame["selection_eligible"] = (
        valid_dates.astype(int).gt(0)
        & finite_mean
        & finite_absolute
    )
    frame["ranking_absolute_mean_ic"] = absolute_values
    frame = frame.sort_values(
        ["selection_eligible", "ranking_absolute_mean_ic", "feature"],
        ascending=[False, False, True],
        na_position="last",
        kind="stable",
    ).reset_index(drop=True)
    frame["selection_rank"] = np.arange(1, len(frame) + 1)
    selected = frame.head(SELECTED_FEATURE_COUNT)
    if not selected["selection_eligible"].all():
        raise ValueError("Fewer than 250 features have valid fold-training IC.")
    ranked_names = selected["feature"].tolist()
    selected_set = set(ranked_names)
    model_names = [
        feature for feature in official_medium_features if feature in selected_set
    ]
    if len(model_names) != SELECTED_FEATURE_COUNT:
        raise ValueError("Selected model feature order is incomplete or duplicated.")
    ranked_rows = tuple(
        {
            "selection_rank": int(row.selection_rank),
            "feature": str(row.feature),
            "mean_ic": float(row.mean_ic),
            "absolute_mean_ic": float(row.ranking_absolute_mean_ic),
            "valid_dates": int(row.valid_dates),
        }
        for row in selected.itertuples(index=False)
    )
    return FoldFeatureManifest(
        policy_id=POLICY_ID,
        fold_id=fold.fold,
        selected_feature_count=SELECTED_FEATURE_COUNT,
        source_feature_count=SOURCE_FEATURE_COUNT,
        train_start=fold.train_dates[0],
        train_end=fold.train_dates[-1],
        training_dates=len(fold.train_dates),
        source_ic_analysis_id=str(source_metadata["analysis_id"]),
        source_feature_list_sha256=feature_list_sha256(official_medium_features),
        selected_ranked_features_sha256=feature_list_sha256(ranked_names),
        selected_model_features_sha256=feature_list_sha256(model_names),
        selected_ranked=ranked_rows,
        model_feature_names=tuple(model_names),
    )


def build_top250_fold_manifests(
    fold_summaries: pd.DataFrame,
    source_metadata: dict,
    official_medium_features: list[str],
    folds: list[V6WalkForwardFold],
) -> dict[int, FoldFeatureManifest]:
    if folds != approved_v6_feature_quality_folds():
        raise ValueError("Feature selection requires all approved v6 folds.")
    _validate_source_metadata(source_metadata, official_medium_features)
    if fold_summaries.duplicated(["fold_id", "feature"]).any():
        raise ValueError("Feature-quality summaries contain duplicate identities.")
    expected_pairs = {
        (fold.fold, feature)
        for fold in folds
        for feature in official_medium_features
    }
    actual_pairs = set(
        zip(
            fold_summaries["fold_id"].astype(int),
            fold_summaries["feature"].astype(str),
        )
    )
    if actual_pairs != expected_pairs:
        raise ValueError("Feature-quality summaries are incomplete or inconsistent.")
    return {
        fold.fold: select_fold_top250(
            fold_summaries[fold_summaries["fold_id"].eq(fold.fold)],
            fold,
            source_metadata,
            official_medium_features,
        )
        for fold in folds
    }


def summarize_fold_manifest_stability(
    manifests: list[dict],
    small_features: list[str],
) -> dict:
    """Summarize saved manifests without reading targets, predictions, or models."""
    if len(manifests) != 5:
        raise ValueError("Expected exactly five fold feature manifests.")
    manifests = sorted(manifests, key=lambda item: int(item["fold_id"]))
    if [int(item["fold_id"]) for item in manifests] != [1, 2, 3, 4, 5]:
        raise ValueError("Feature manifests do not contain approved fold identities.")
    selected_sets = []
    for item in manifests:
        if item.get("policy_id") != POLICY_ID:
            raise ValueError("Feature manifest policy identity does not match.")
        selected = item.get("model_feature_names", [])
        if len(selected) != SELECTED_FEATURE_COUNT or len(set(selected)) != len(selected):
            raise ValueError("Feature manifest must contain 250 unique model features.")
        selected_sets.append(set(selected))
    counts = pd.Series(
        [feature for selected in selected_sets for feature in selected]
    ).value_counts()
    small = set(small_features)
    pairwise = [
        {
            "fold_a": fold_a,
            "fold_b": fold_b,
            "shared_features": len(selected_sets[fold_a - 1] & selected_sets[fold_b - 1]),
            "jaccard": float(
                len(selected_sets[fold_a - 1] & selected_sets[fold_b - 1])
                / len(selected_sets[fold_a - 1] | selected_sets[fold_b - 1])
            ),
        }
        for fold_a, fold_b in combinations(range(1, 6), 2)
    ]
    return {
        "shared_all_five": int(counts.eq(5).sum()),
        "selected_only_one_fold": int(counts.eq(1).sum()),
        "unique_selected_across_folds": int(len(counts)),
        "pairwise_jaccard": pairwise,
        "group_membership": [
            {
                "fold_id": fold_id,
                "small": len(selected & small),
                "medium_only": len(selected - small),
            }
            for fold_id, selected in enumerate(selected_sets, start=1)
        ],
    }
