# Numerai Research Log

## Current Benchmark

**v1 — Medium Feature LightGBM**

- Feature set: Medium (780 features)
- Target: Ender-60 (`target`)
- Training eras: 0001–0558
- Purged eras: 0559–0574
- Validation eras: 0575–1221
- Test eras: 1222–1234 (untouched)
- Model: LightGBM
- Parameters:
  - n_estimators: 300
  - learning_rate: 0.05
  - num_leaves: 31

### Validation Results

- Mean CORR: 0.019724
- Std CORR: 0.015247
- Median CORR: 0.019823
- CORR Sharpe-like: 1.294
- Positive Era Rate: 89.80%
- Worst Era CORR: -0.029142
- Best Era CORR: 0.066645

---

## Experiment History

### v0 — Small Feature Baseline

**Change**
- LightGBM baseline using Small feature set (42 features).

**Result**
- Mean CORR: 0.017655
- CORR Sharpe-like: 1.196
- Positive Era Rate: 87.48%

**Conclusion**
- Established the initial baseline.

---

### v1 — Medium Feature Set

**Change**
- Increased feature set from Small (42) to Medium (780).
- Model and validation methodology unchanged.

**Result**
- Mean CORR: 0.019724
- CORR Sharpe-like: 1.294
- Positive Era Rate: 89.80%

**Conclusion**
- Mean CORR improved by approximately 11.7%.
- Medium features contain useful additional predictive information.
- Promoted v1 to the working benchmark.

---

### v2 — Slower Boosting Schedule

**Change**
- n_estimators: 300 → 500
- learning_rate: 0.05 → 0.03
- Medium 780 features unchanged.

**Result**
- Mean CORR: 0.019704
- Std CORR: 0.015050
- CORR Sharpe-like: 1.309
- Positive Era Rate: 90.26%

**Conclusion**
- Average predictive performance did not improve.
- Stability improved slightly.
- v1 remains the benchmark.
- v2 retained as a possible future ensemble candidate.

---

## Pipeline Refactor

Refactored repeated notebook logic into reusable modules:

- `src/paths.py` — project paths
- `src/config.py` — shared configuration and default parameters
- `src/data.py` — feature loading, data loading, purge, dataset preparation
- `src/models.py` — model creation, training, prediction
- `src/evaluation.py` — Numerai CORR and era-level evaluation
- `src/pipeline.py` — end-to-end experiment pipeline

### Validation

The refactored pipeline successfully reproduced the v1 benchmark:

- Mean CORR: 0.01972388
- Std CORR: 0.01524748
- CORR Sharpe-like: 1.29358
- Positive Era Rate: 89.799%

This confirms that the refactor did not change the research logic.

---

## Next Steps

1. Add automatic experiment result persistence.
2. Add robust walk-forward validation.
3. Continue high-value model / feature experiments.
4. Compare candidate models using prediction correlation and stability.
5. Preserve the final test set until candidate selection is complete.

## Expanding Walk-Forward Validation Infrastructure

### Change

Extended the research pipeline with expanding walk-forward validation.

Implemented:

- Chronological research history construction
- Expanding training windows
- 16-era purge windows
- 156-era validation windows
- Fold-level model retraining
- Fold-level CORR evaluation
- Combined out-of-sample CORR evaluation
- Walk-forward experiment persistence

The fixed-validation pipeline was retained for fast experiment screening.

### Validation Design

The walk-forward procedure currently uses five folds:

1. Train 0001–0558 → Validate 0575–0730
2. Train 0001–0714 → Validate 0731–0886
3. Train 0001–0870 → Validate 0887–1042
4. Train 0001–1026 → Validate 1043–1198
5. Train 0001–1182 → Validate 1199–1221

A new LightGBM model is trained independently for each fold.

Earlier validation eras are incorporated into the expanding training history in later folds, while eras 1222–1234 remain untouched as the final test set.

### Decision

Use fixed validation for rapid idea screening and expanding walk-forward validation for stronger candidate evaluation.

The intended research process is:

Idea → Fixed Validation → Walk-Forward Validation → Untouched Test → Live

## Small Feature Set Walk-Forward Pipeline Validation

### Change

Evaluated the 42-feature Small feature set using the expanding walk-forward pipeline and the baseline LightGBM configuration.

### Result

| Fold | Validation Eras | Mean CORR | Sharpe-like | Positive Era Rate |
|---|---|---:|---:|---:|
| 1 | 0575–0730 | 0.02331 | 1.560 | 93.59% |
| 2 | 0731–0886 | 0.02107 | 1.572 | 92.31% |
| 3 | 0887–1042 | 0.01793 | 1.331 | 93.59% |
| 4 | 1043–1198 | 0.01303 | 0.944 | 83.33% |
| 5 | 1199–1221 | 0.00885 | 0.783 | 78.26% |

Combined out-of-sample results across 647 eras:

- Mean CORR: 0.01848
- CORR Std: 0.01443
- Median CORR: 0.01898
- Sharpe-like: 1.281
- Positive Era Rate: 90.26%
- Minimum Era CORR: -0.03001
- Maximum Era CORR: 0.06278

### Conclusion

The walk-forward pipeline successfully produced continuous out-of-sample predictions across all 647 validation eras.

Performance declined across successive folds, with mean CORR decreasing from approximately 0.0233 in Fold 1 to 0.0089 in Fold 5.

This is evidence of time variation in predictive performance, but it does not by itself establish the cause. Possible explanations include feature decay, regime changes, insufficient coverage from the Small feature set, or the effect of using a long expanding training history.

The final fold contains only 23 eras and should therefore be interpreted with additional caution.

### Decision

Retain the Small walk-forward result as an infrastructure validation and reference benchmark.

Next, evaluate the Medium 780-feature benchmark under the same walk-forward framework to test whether the stronger feature set improves performance stability, particularly in later eras.

## Medium Feature Set Walk-Forward Experiment

### Change

Evaluated the Numerai Medium feature set containing 780 features using the same expanding walk-forward framework and baseline LightGBM configuration used for the Small benchmark.

The only major experimental change was:

- Small feature set: 42 features
- Medium feature set: 780 features

Model configuration remained unchanged:

- LightGBM
- `n_estimators = 300`
- `learning_rate = 0.05`
- `num_leaves = 31`
- `random_state = 42`

### Result

Fold-level performance:

| Fold | Validation Eras | Small Mean CORR | Medium Mean CORR | Medium Sharpe-like | Medium Positive Era Rate |
|---|---|---:|---:|---:|---:|
| 1 | 0575–0730 | 0.02331 | 0.02597 | 2.047 | 96.79% |
| 2 | 0731–0886 | 0.02107 | 0.03599 | 2.168 | 99.36% |
| 3 | 0887–1042 | 0.01793 | 0.02830 | 2.324 | 97.44% |
| 4 | 1043–1198 | 0.01303 | 0.01566 | 1.039 | 82.05% |
| 5 | 1199–1221 | 0.00885 | 0.02646 | 2.475 | 100.00% |

Combined out-of-sample performance across all 647 validation eras:

| Metric | Small 42 | Medium 780 |
|---|---:|---:|
| Mean CORR | 0.01848 | 0.02648 |
| CORR Std | 0.01443 | 0.01580 |
| Median CORR | 0.01898 | 0.02601 |
| Sharpe-like | 1.281 | 1.676 |
| Positive Era Rate | 90.26% | 94.13% |
| Minimum CORR | -0.03001 | -0.02512 |
| Maximum CORR | 0.06278 | 0.07267 |

The Medium feature set improved overall Mean CORR by approximately 43% relative to the Small feature benchmark.

### Conclusion

The Medium feature set provides substantially stronger and more robust predictive performance than the Small feature set under expanding walk-forward validation.

The improvement is particularly large in Folds 2, 3, and 5.

Unlike the Small benchmark, which showed a nearly monotonic decline across later folds, the Medium benchmark recovered strongly in Fold 5. This suggests that the deterioration observed with the Small feature set was not simply an unavoidable decline in predictability over time.

However, Fold 4 remains substantially weaker for both Small and Medium feature sets. This suggests the existence of a more difficult historical regime or distribution shift during eras 1043–1198.

The cause has not yet been identified. Possible explanations include:

- market regime changes
- feature distribution shifts
- target relationship changes
- reduced effectiveness of long expanding training history
- changes in which features carry predictive information

No specific real-world market event is currently assigned to Fold 4 because the era-to-calendar mapping has not yet been verified.

### Decision

Promote the Medium 780-feature model to the current walk-forward benchmark.

Current benchmark:

- Feature set: Medium 780
- Model: LightGBM
- Mean walk-forward CORR: 0.02648
- Sharpe-like: 1.676
- Positive era rate: 94.13%

The next research stage will focus on feature research and selection.

The goal is to identify features that are:

- predictive
- stable across eras
- robust across regimes
- non-redundant
- useful across multiple model classes

Candidate feature sets will first be evaluated using fixed validation. Only promising candidates will proceed to the more expensive walk-forward evaluation.

---

## 2026-11-21 — v6 Research Validation Protocol

### Decision

Approved and implemented the leakage-safe data boundaries and fold-generation protocol for the Numerai v6.0 V1 research workflow.

- Explicit target: `target_jupiter_60`
- Time column: weekly `date`, ordered by observed values
- Expanding training history
- Purge: 16 observed dates before every validation window
- Validation window: 78 dates
- Research validation: 2019-01-04–2025-04-25
- Final holdout gap: 2025-05-02–2025-08-15
- Sealed final holdout: 2025-08-22–2026-08-14
- Immature dates: 2026-08-21–2026-11-13

The five research folds contain 78, 78, 78, 78, and 18 validation dates. The final partial fold is retained.

### Rationale and Leakage Safeguards

The 16-date purge prevents training labels immediately before a validation window from overlapping its forward-return period. The same 16-date separation is reserved before the sealed holdout. Ordinary research loaders expose target values only for historical training and approved research-validation dates; gap, holdout, and immature partitions remain outside research history.

The implementation uses explicit, snapshot-fixed date boundaries. Missing dates, unexpected added dates, changed `data_type` assignments, and null Jupiter-60 labels inside approved research history cause verification to fail rather than silently changing the experiment.

Row identity (`numerai_ticker`, `composite_figi`, and `date`) is retained for future out-of-fold prediction alignment.

### Status

No v6 model has been trained, no feature selection has been performed, and no v6 holdout target has been evaluated. Historical v5.3 results remain the only model benchmarks in this log.

---

## 2026-11-21 — Task 17A: Reusable v6 LightGBM Interface

### Implementation

Implemented the reusable model boundary required before v6 walk-forward orchestration:

- Schema-backed resolution of official v6 feature sets
- Initial `medium` policy: 351 metadata entries, excluding `feature_country`, resulting in 350 numeric features
- Preservation of official feature ordering
- Explicit separation of numeric features, `target_jupiter_60`, and row identity
- Validation of missing, nonnumeric, duplicated, or reordered model features
- Validation of null, nonnumeric, or length-mismatched training targets
- Prediction output aligned to the supplied DataFrame index
- Deterministic LightGBM behavior under the configured random seed

The initial model parameters remain:

- `n_estimators = 300`
- `learning_rate = 0.05`
- `num_leaves = 31`
- `random_state = 42`
- `n_jobs = -1`

LightGBM retains native handling of numeric missing values. No imputation, normalization, target-driven feature selection, feature engineering, hyperparameter tuning, or validation-target early stopping was introduced.

### Verification

Permanent tests cover the official Medium resolution, numeric count and ordering, categorical exclusion, feature and target rejection cases, deterministic synthetic fitting, prediction alignment, v6 model-data separation, and legacy v5.3 model-function compatibility.

Only tiny synthetic models were fitted for interface verification. No full v6 model, walk-forward run, OOF predictions, CORR/MMC metrics, model artifacts, or performance results exist yet.

---

## 2026-11-21 — Task 17B: v6 OOF Evaluation and Artifact Contracts

### Official Scoring Verification

Verified current Classic scoring against Numerai documentation, the public round-score configuration API, and the official `numerai-tools` 0.7.2 implementation. Public round 1404 configured:

- NCORR (`neutral_corr` v1): payout, 8×
- NMMC (`neutral_mmc` v1): payout, 32×
- NBMC (`neutral_bmc` v1): informational, 0×
- Canonical neutral companion series and `season_score`: informational, 0×

NCORR requires the official v6 neutralizer matrix. NMMC additionally requires the matching historical `classic_v6_nuswmm_pow1` Meta Model. NBMC requires the exact shared v6 benchmark reference. None of these required reference inputs is currently available locally, and the Meta Model reference is not present in the verified v6 dataset catalog.

No legacy CORR, MMC, benchmark prediction, or proxy has been substituted for a neutral metric.

### Implementation

Implemented a reusable research OOF contract with aligned `date`, `numerai_ticker`, `composite_figi`, fold, prediction, and Jupiter-60 target fields. Validation rejects duplicate identities, missing or nonfinite predictions, null targets, incomplete research-date coverage, incorrect fold assignments, and all gap/holdout/immature dates.

Implemented `NUMERAI_CORR_JUPITER_60` using the exact official `numerai_corr` rank, gaussianization, signed-power, and Pearson-correlation formula. This is recorded only as a non-neutral historical research diagnostic, not as NCORR or a current payout score.

Date-level aggregation provides overall, per-fold, and per-year summaries. Overall metrics operate over individual date scores, preventing the final 18-date fold from receiving the same weight as each 78-date fold.

Implemented validated, non-overwriting experiment artifacts under `outputs/v6.0/experiments/<experiment_id>/`:

- `metadata.json`
- `metrics.json`
- `oof_predictions.parquet`
- `per_date_metrics.parquet`

Metadata retains the ordered feature list, model parameters, target, exact folds and purge, random seed, metric identity, and Git state when available. No raw dataset copy or dummy model file is written.

### Status

All Task 17B verification uses synthetic predictions and targets. No full v6 training, five-fold run, holdout access, real OOF prediction, neutral metric, or performance result exists yet. NCORR, NMMC, and NBMC remain explicitly unavailable pending their exact official reference inputs and supported scoring dependency.

---

## 2026-11-21 — Task 17C: Official v6 Scoring References

### Data acquisition and verification

Downloaded only `v6.0/validation_neutralizer.parquet` through the reusable repository downloader. The 4,699,025,422-byte file has SHA-256 `1ae870b17d501b8d9ec114ab2096eece342812c424a826f44a772087a82c6c6f` and a readable footer containing 2,746,248 rows, 304 columns, and 411 row groups. Its schema has 300 float neutralizer columns plus the string fields `numerai_ticker`, `date`, `data_type`, and `composite_figi`.

Added a reusable, target-free reference audit. It confirmed exact row-order alignment with Validation on `numerai_ticker`, `date`, and `composite_figi`, date coverage from 2019-01-04 through 2026-11-13, and coverage of all 330 approved Research Validation dates. No holdout target or score was read or calculated.

### Reference availability decisions

- NCORR: the official historical Validation neutralizer input is now local and verified; scoring remains unimplemented.
- NMMC: the public v6 catalog has no `classic_v6_nuswmm_pow1` reference, and NumerAPI 3.1.1 exposes no supported public historical Meta Model retrieval method. No substitute is permitted.
- NBMC: Validation benchmark-model data is catalogued, but the exact shared Classic v6 / Signals v3.1 reference column or blend is not documented. It was not downloaded or treated as the official benchmark.
- Dependency: official `numerai-tools` 0.7.2 supports Python 3.12 and is compatible with the installed scientific package versions. It was not installed and project dependencies were not changed in this acquisition-only task.

This entry records reference acquisition and engineering verification, not a model-performance experiment. No model was trained, no walk-forward run occurred, and no CORR, NCORR, NMMC, or NBMC value was calculated.

---

## 2026-11-21 — Task 17D: Official Classic v6 NCORR Scoring

Installed and pinned the official `numerai-tools==0.7.2` dependency. Implemented NCORR with the package's `neutral_correlation` function, preserving the official `NCORR` / `neutral_corr` version-1 identity and explicit `target_jupiter_60` target.

The reference loader reads one approved Research Validation date at a time from the 4.7 GB Validation neutralizer. It uses all 300 columns in verified order and requires exact alignment on `date`, `numerai_ticker`, and `composite_figi`. Missing, duplicate, reordered, unsupported, null, nonfinite, or degenerate inputs are rejected. Gap, sealed holdout, and immature dates are outside the accepted scoring dates.

Synthetic tests compare the repository result directly with the installed official function at absolute tolerance `1e-12`, verify use of every supplied neutralizer column, exercise alignment and invalid-input safeguards, and verify per-date aggregation and experiment metadata. Future artifacts can store NCORR and the separately named Legacy CORR together, including package version, neutralizer version/checksum, and exact scoring coverage.

NMMC remains unavailable without `classic_v6_nuswmm_pow1`; NBMC remains unavailable without an authoritative shared-benchmark mapping. Neither is approximated.

This is scoring-infrastructure verification, not a performance experiment. No real v6 OOF predictions or NCORR values were generated, no five-fold walk-forward was run, and no holdout target was accessed.

---

## 2026-11-21 — Task 17E: Integrated v6 Walk-Forward Pipeline

Integrated the existing v6 data, fold, model, OOF, NCORR, Legacy CORR, and artifact interfaces into one reproducible orchestration path. The pipeline resolves the official 350 numeric Medium features, loads only selected research columns, fits a fresh baseline LightGBM model for each approved expanding fold, preserves row identity and fold IDs, validates complete OOF coverage, and calculates the two metrics under distinct identities.

Memory behavior is bounded by retaining only one fitted model and fold-specific matrices at a time. Models are serialized immediately, large fold objects are released, and the 4.7 GB neutralizer remains date-filtered. Peak memory for the real dataset has not yet been measured.

The CLI separates `--smoke` from real execution. Smoke mode creates only synthetic approved-date data and records `run_kind: synthetic_smoke`; real execution requires both `--run` and `--confirm-full-run`. Artifact loading now requires a final completion marker, so interrupted output cannot be mistaken for a completed experiment. Completed artifacts support five fold-model files, exact configuration and folds, validated OOF predictions, date-level metrics, aggregate/fold summaries, and scoring-reference metadata.

Permanent integration tests cover end-to-end train/predict/OOF/score/save/reload behavior, fold-to-row mapping, expanding training and 16-date purge, exclusion of same-fold validation rows from fitting, official NCORR integration and alignment safeguards, complete 330-date coverage, model persistence, CLI confirmation, and incomplete-run rejection. The full suite passed all 56 tests, including existing v5.3 compatibility coverage.

No real five-fold v6 baseline was executed, no real OOF performance result was created, no feature selection or tuning occurred, and no holdout target was accessed.

---

## 2026-11-21 — First Real v6 Medium LightGBM Baseline

### Hypothesis and configuration

Hypothesis: the unchanged baseline LightGBM trained on the official v6 Medium numeric feature set should provide a reproducible, positive Research Validation reference for later feature and model comparisons.

- Experiment ID: `v6_medium_baseline_001`
- Dataset / target: `v6.0` / `target_jupiter_60`
- Features: official Medium, 350 numeric features in metadata order
- Model: LightGBM with `n_estimators=300`, `learning_rate=0.05`, `num_leaves=31`, `random_state=42`, `n_jobs=-1`
- Validation: five approved expanding folds; 16 observed-date purge; validation windows 78/78/78/78/18 dates
- Research OOF: 2019-01-04 through 2025-04-25, 330 dates and 2,180,295 rows
- Official score: NCORR (`neutral_corr` version 1)
- Diagnostic: `NUMERAI_CORR_JUPITER_60`, reported separately as Legacy CORR

### Aggregate results

| Metric | Mean | Std. dev. | Sharpe-like | Positive dates |
|---|---:|---:|---:|---:|
| NCORR | 0.026121 | 0.017981 | 1.4527 | 93.33% |
| Legacy CORR | 0.022410 | 0.016756 | 1.3374 | 91.21% |

### Fold results

| Fold | Dates | Mean NCORR | Mean Legacy CORR |
|---:|---:|---:|---:|
| 1 | 78 | 0.025914 | 0.022606 |
| 2 | 78 | 0.030107 | 0.025922 |
| 3 | 78 | 0.027457 | 0.022189 |
| 4 | 78 | 0.022474 | 0.019801 |
| 5 | 18 | 0.019773 | 0.018609 |

### Annual results

| Year | Dates | Mean NCORR | NCORR Sharpe-like | Mean Legacy CORR | Legacy Sharpe-like |
|---:|---:|---:|---:|---:|---:|
| 2019 | 52 | 0.029578 | 1.9879 | 0.026000 | 1.9206 |
| 2020 | 52 | 0.015034 | 1.0453 | 0.012424 | 0.9202 |
| 2021 | 53 | 0.039307 | 2.6659 | 0.034121 | 2.7519 |
| 2022 | 52 | 0.026913 | 1.7189 | 0.021799 | 1.5444 |
| 2023 | 52 | 0.034795 | 2.0376 | 0.031205 | 1.8152 |
| 2024 | 52 | 0.013239 | 1.0353 | 0.010261 | 0.8049 |
| 2025 | 17 | 0.018805 | 0.8517 | 0.017599 | 0.8428 |

### Runtime, verification, and interpretation

The run took approximately 18 minutes wall-clock. Exact per-fold and scoring times were not instrumented. LightGBM reported normal row-wise threading selection and expanding training sizes of 4,227,405 to 6,284,208 rows. Peak memory could not be measured reliably because Windows process isolation blocked process sampling; no memory error occurred.

Artifacts reloaded successfully with five models, the completion marker, the official NCORR identity and `numerai-tools==0.7.2`, and the verified neutralizer checksum. OOF verification found exactly 330 dates, no duplicate identities, no missing predictions, and no gap, holdout, or immature dates.

NCORR exceeded Legacy CORR overall and in every fold. Performance was strongest in 2021 and 2023 and materially weaker in 2020, 2024, and the partial 2025 period. Fold means declined after Fold 2, with the short Fold 5 showing the lowest mean and positive-date rate; its 18-date size makes it less stable than the four full folds. The result is positive and reasonably consistent but clearly regime-sensitive.

Limitations: this is repeated Research Validation rather than sealed-holdout evidence; no hyperparameter or feature comparison has yet been performed; the final fold is short; runtime components and peak memory were not instrumented; NMMC and NBMC remain unavailable.

Next research questions should be reviewed before action: model-independent feature quality and stability, alternative baseline model families under the same folds, and whether carefully predeclared ensembling improves later-period robustness. No tuning or follow-on experiment was started in this task.

---

## 2026-11-21 — Task 18A: v6 Small vs Medium Feature Set

### Hypothesis and controls

Hypothesis: the smaller official v6 feature set might improve generalization or temporal stability relative to Medium.

This was a controlled comparison between `v6_medium_baseline_001` and `v6_small_baseline_002`. Only the feature set changed. Small resolved to 100 numeric features, all an ordered subset of Medium's 350. Both experiments used `v6.0`, `target_jupiter_60`, the unchanged baseline LightGBM parameters and seed, identical five expanding folds, a 16-date purge, the same 330 Research Validation dates, the official neutralizer, and NCORR version 1. OOF row identities matched exactly.

### Aggregate results

| Metric | Medium | Small | Small − Medium |
|---|---:|---:|---:|
| Mean NCORR | 0.026121 | 0.015617 | -0.010504 |
| NCORR std. dev. | 0.017981 | 0.015988 | -0.001993 |
| NCORR Sharpe-like | 1.4527 | 0.9768 | -0.4759 |
| NCORR positive dates | 93.33% | 80.91% | -12.42 pp |
| Mean Legacy CORR | 0.022410 | 0.012855 | -0.009555 |
| Legacy CORR Sharpe-like | 1.3374 | 0.8972 | -0.4402 |

Across the 330 paired dates, the NCORR difference had mean `-0.010504`, standard deviation `0.017189`, and Small outperformed Medium on 24.24% of dates. No statistical-significance claim is made because weekly score dependence was not modeled.

### Fold comparison

| Fold | Medium NCORR | Small NCORR | Difference |
|---:|---:|---:|---:|
| 1 | 0.025914 | 0.011938 | -0.013976 |
| 2 | 0.030107 | 0.008531 | -0.021576 |
| 3 | 0.027457 | 0.018721 | -0.008735 |
| 4 | 0.022474 | 0.021814 | -0.000659 |
| 5 | 0.019773 | 0.021968 | +0.002196 |

### Annual NCORR comparison

| Year | Medium | Small | Difference |
|---:|---:|---:|---:|
| 2019 | 0.029578 | 0.015110 | -0.014467 |
| 2020 | 0.015034 | 0.000775 | -0.014259 |
| 2021 | 0.039307 | 0.015223 | -0.024084 |
| 2022 | 0.026913 | 0.016910 | -0.010004 |
| 2023 | 0.034795 | 0.025253 | -0.009542 |
| 2024 | 0.013239 | 0.018378 | +0.005139 |
| 2025 partial | 0.018805 | 0.021926 | +0.003121 |

### Interpretation and limitations

Small did not improve generalization overall. Its disadvantage was broad across 2019–2023 and Folds 1–4, while its advantage was concentrated in 2024, partial 2025, and the short fifth fold. Small also had lower score dispersion, but not enough to offset its much lower mean. This comparison alone does not identify useful or harmful individual features, and no causal explanation is assigned to the recent reversal.

The Small run completed in approximately nine minutes without errors. Exact per-fold timing and peak memory were not instrumented. Artifacts contain 2,180,295 OOF rows, 330 dates, five models, the completion marker, and the same verified NCORR reference checksum as Medium. No holdout target was accessed.

The next research question should focus on model-independent feature quality and temporal stability—especially whether Medium-only features add broad historical signal at the cost of recent robustness—before any feature subset is selected. No additional experiment was started.

---

## 2026-11-23 — Task 18B: All-Feature Memory and Runtime Preflight

### Configuration and scope

Profiled the official v6 `all` set using a bounded 100,000-row Train sample. After excluding `feature_country`, the model matrix contained 1,039 numeric features: 1,017 `int8` and 22 `float32`. The profiling fit used the unchanged baseline LightGBM parameters (`n_estimators=300`, `learning_rate=0.05`, `num_leaves=31`, `random_state=42`, `n_jobs=-1`). This was an engineering feasibility fit only; it did not use Validation, run folds, or calculate NCORR, Legacy CORR, NMMC, or NBMC.

### Measurements

- Sample-frame deep memory: 116,252,335 bytes
- Feature columns: 110,500,000 bytes, or 1,105 bytes per row
- Extracted model-feature copy: 110,500,132 bytes
- Dtype conversions: none
- Process RSS before/after sample load: 166,322,176 / 1,243,070,464 bytes
- Process RSS before/after model-data extraction: 1,243,553,792 / 1,354,715,136 bytes
- LightGBM runtime: 15.35 seconds
- Fit RSS before / sampled peak / after: 1,354,743,808 / 3,079,294,976 / 1,368,403,968 bytes

LightGBM selected col-wise processing, built 8,685 bins, and reported 787 used features for this bounded early-Train sample. This is not a feature-quality result; the sample may not expose variation available elsewhere.

### Extrapolation and pipeline assessment

At 1,105 observed feature bytes per row, one 6,284,208-row Fold 5 pandas feature matrix extrapolates to 6,944,049,840 bytes (about 6.47 GiB). This excludes research-history storage, `.copy()` fold slices, the second model-feature copy, validation matrices, LightGBM native datasets/histograms, temporary arrays, and Windows/application headroom.

The current pipeline can temporarily retain full historical and research frames while concatenating research history, then retain history alongside copied fold and model matrices. The bounded LightGBM peak also cannot be scaled linearly with confidence because fixed overhead, bin construction, feature filtering, threading, and dataset size interact nonlinearly. Training runtime likewise is not extrapolated linearly.

### Decision

Classification: **High risk / not recommended** on the current 32 GB Windows machine. Do not start the full All-feature five-fold experiment with the current copy behavior. First reduce full-frame duplication and fold/model copies, then repeat a bounded and preferably larger staged memory profile. No All-feature research experiment or model-performance result was produced.

---

## 2026-11-23 — Task 18B.1: v6 Pipeline Memory Optimization

### Duplication and implementation

The real pipeline retained caller-owned historical and Research Validation frames while also building a complete concatenated research history. For every fold it then made complete training and validation DataFrame copies before `prepare_v6_model_data` copied feature matrices again. The optimized implementation selects approved dates directly from the two protected source frames into final LightGBM feature/target inputs. It does not build the concatenated history or whole-fold frames. Models continue to be saved and released one fold at a time, OOF parts contain only required output columns, and official neutralizers remain date-bounded.

No feature value/order, target, identity order, fold boundary, purge, model parameter, seed, scoring code, or holdout protection changed. Synthetic comparisons across all five folds matched legacy inputs exactly, including dtypes. A deterministic bounded LightGBM comparison matched predictions at absolute tolerance `1e-12`.

### Before/after bounded profile

Both profiles used the same first 100,000 Train rows, 1,039 numeric All features, and unchanged baseline parameters.

| Measurement | Task 18B | Task 18B.1 |
|---|---:|---:|
| Feature bytes | 110,500,000 | 110,500,000 |
| RSS after input extraction | 1,354,715,136 | 1,246,912,512 |
| RSS before fit | 1,354,743,808 | 1,246,953,472 |
| Sampled peak RSS | 3,079,294,976 | 2,986,553,344 |
| Fit runtime | 15.35 s | 15.02 s |

The optimized bounded copy model estimates 221,000,132 concurrent feature bytes instead of 331,500,264, removing one 110,500,132-byte sample-scale copy. Measured peak RSS fell by 92,741,632 bytes (3.0%). The 6,509,738-row full history feature payload and one Fold 5 feature matrix project to roughly 7.19 GB and 6.94 GB respectively; their removal is a structural projection, not a full-fold measurement, and the benefits cannot safely be added because of pandas and LightGBM allocation behavior.

### Decision

Classification remains **High risk / not recommended** for a complete All-feature five-fold run on 32 GB. The change removes the largest avoidable pandas duplicates, but the bounded profile cannot guarantee sufficient headroom for multi-million-row LightGBM native datasets and temporary structures. Do not launch the All-feature experiment without a larger staged profile; targeted feature research is the preferred next direction if risk reduction is the priority. This engineering task produced no predictive-performance result and accessed no Validation or holdout target.

---

## 2026-11-23 — Task 19B: Leakage-Safe Feature IC Infrastructure

### Methodology

Implemented model-independent feature-quality infrastructure for the official 350 numeric v6 Medium features and `target_jupiter_60`. Cross-sectional IC is calculated independently within each weekly date as Pearson correlation of pairwise-complete average feature ranks and average target ranks—the standard Spearman definition with explicit tie handling. Signed numerical IC, valid-pair counts, row coverage, and status are retained. Valid zero IC is distinct from missing IC; insufficient pairs and constant features or targets produce missing IC with an explicit reason.

Reusable summaries include mean and median IC, sample standard deviation, absolute mean IC, mean-to-standard-deviation ratio when defined, sign consistency, valid-date coverage, rolling 52-date mean IC, and annual summaries. No combined feature-ranking formula or selection threshold was implemented.

### Leakage safeguards

The as-of interface accepts only an exact approved v6 fold and requires complete observations for that fold's `train_dates`. It rejects purge dates, same-fold Validation dates, observations available only to later folds, gap dates, sealed holdout dates, and immature dates. Fold summaries are recomputed independently. The future runner reads the union of approved training dates one date at a time and requires non-null mature targets; earlier-fold summaries cannot consume later observations.

The artifact contract stores per-date IC, fold summaries, rolling and annual tables, feature identity/order, method and missingness policy, exact fold cutoffs, configuration, and Git state beneath a non-overwriting versioned output path. It follows the completed-marker convention used by model experiments but remains a separate model-independent artifact type.

### Verification and status

Permanent synthetic tests compare IC directly with `scipy.stats.spearmanr`, including ties, missing pairs, constants, insufficient observations, sign preservation, and valid zero IC. They also verify rolling-window behavior, exact fold cutoffs, purge/validation/future/holdout rejection, artifact round-trip and overwrite protection, and unchanged v5.3 defaults.

This task implemented infrastructure only. The real Medium audit command was documented but not executed. No real feature IC, ranking, feature selection, LightGBM training, redundancy analysis, ablation, OOF score, or predictive-performance result was produced, and no holdout target was accessed.

---

## 2026-11-23 — Task 19C: Real v6 Medium Feature Quality Audit

### Execution and verification

Executed exactly one audit with analysis ID `v6_medium_ic_001` using the documented `src.feature_quality` command. Runtime was approximately 14 minutes 18 seconds. The completed artifact contains 350 ordered Medium numeric features, 1,131 approved training-history dates from 2003-01-03 through 2024-08-30, and 395,850 unique feature-date rows. All five metadata cutoffs match the approved folds. No purge, same-fold Validation, gap, holdout, or immature date appears in a fold's selection summary.

There are 378,585 valid IC rows (95.6385%) and 17,265 `constant_feature` rows with explicitly missing IC. No duplicate date-feature identity exists. The widest legitimate descriptive view is Fold 5's as-of training history through 2024-08-30, not the later Research Validation or sealed holdout.

### Standalone feature quality

Across the 350 latest-as-of feature summaries, mean signed IC averages `0.001129` with median `0.001001`; 218 features have positive mean IC and 132 have negative mean IC. Absolute mean IC averages `0.002674`, with median `0.002270`, 90th percentile `0.005581`, and maximum `0.009874`. Median sign consistency is `0.565990`, and median valid-date coverage is 100%.

The strongest five features by absolute mean IC are `feature_constitution_30`, `feature_strength_32`, `feature_charisma_152`, `feature_constitution_29`, and `feature_constitution_36`. Their anonymous identifiers are not assigned financial meanings. Weakest features have absolute mean IC near zero and sign consistency close to 50%, including `feature_constitution_337`, `feature_constitution_175`, and `feature_wisdom_47`.

Absolute-IC ranks remain broadly persistent from Fold 1 to Fold 5 (Spearman `0.896282`), but the median feature moves 24 places and several features move more than 100 places. This supports fold-specific recomputation rather than a timeless global ranking.

### Small versus Medium-only and recent behavior

Medium-only features have higher latest-as-of mean/median absolute mean IC (`0.002773` / `0.002403`) than Small (`0.002426` / `0.001865`), along with higher median sign consistency (`0.572060` versus `0.550806`) and coverage. Their median Fold 1-to-5 rank movement is also smaller (23 versus 28).

For the latest 52 eligible dates, 2023-09-08–2024-08-30, mean absolute IC is `0.004766` across features versus `0.002694` for the earlier-history mean. Medium-only and Small recent absolute means are similar (`0.004795` and `0.004693`). Medium-only has more weakening features proportionally (27.6% versus 23.0%) but fewer sign reversals (38.0% versus 41.0%). Short-window absolute values have greater sampling variability, so these comparisons are descriptive rather than equivalent-strength estimates.

These standalone results do not show aggregate Medium-only deterioration sufficient to explain the Small model's 2024–2025 OOF advantage. They also do not refute model-dependent causes: redundancy, interactions, conditional contribution, and regularization behavior were not tested. No causal claim or feature subset decision is made.

### Status

The reusable saved-artifact analyzer writes `analysis_summary.json` without recomputing IC. No LightGBM model, NCORR/NMMC score, redundancy cluster, importance analysis, permutation test, ablation, or feature-selection policy was run. The final holdout remains sealed.

---

## 2026-11-23 — Task 19D: Existing Medium LightGBM Importance Audit

### Inputs and integrity

Loaded the five saved fold boosters from `v6_medium_baseline_001` without retraining. Every model exposes the same ordered 350-feature list as experiment metadata, official Medium metadata, and `v6_medium_ic_001`. The analysis contains 1,750 unique fold-feature rows; normalized gain and split shares each sum to one per fold. Each importance row joins only to the IC summary from that model's exact training-history cutoff.

### Importance and stability

The leading features by mean normalized gain are `feature_endurance_7` (`0.014921`), `feature_endurance_5` (`0.009863`), `feature_endurance_11` (`0.008583`), `feature_strength_32` (`0.007340`), and `feature_wisdom_60` (`0.006799`). All 350 features are used in at least one model. Twenty-one features fall in the top 35 gain ranks in at least four folds; 20 reach the top 35 in exactly one fold.

Mean pairwise fold gain-rank Spearman is `0.794849`, with minimum `0.753403`; Fold 1 versus Fold 5 is `0.769943`. Median absolute Fold 1-to-5 movement is 38 ranks. Importance is meaningfully persistent but more variable than the standalone absolute-IC ranks reported in Task 19C.

### Small versus Medium-only

Small receives 32.11%–34.16% of total fold gain, while Medium-only receives 65.84%–67.89%. Because the groups contain 100 and 250 features, per-feature averages reverse the group-total comparison: Small averages `0.003337` gain share per feature versus `0.002665` for Medium-only. Small's top ten account for an average 19.49% of its group gain versus 10.78% for Medium-only, showing greater concentration. Split-share results are similar, and nearly every feature has nonzero importance in every fold.

These group totals do not demonstrate incremental contribution or explain the OOF reversal. They reflect model usage under unequal feature counts and correlated alternatives.

### IC versus model usage

Fold-specific absolute mean IC versus gain Spearman ranges from `0.230970` to `0.270855`; sign consistency versus gain ranges from `0.241401` to `0.261147`. The relationship is positive but weak. `feature_charisma_110`, `feature_charisma_104`, and `feature_endurance_19` are high-IC/low-gain in all five folds. `feature_intelligence_20`, `feature_constitution_337`, `feature_constitution_259`, and `feature_constitution_77` are low-IC/high-gain in all five.

The divergence is descriptive and plausible because IC measures standalone monotonic association, while gain records conditional split improvement within a correlated nonlinear tree model. It is not evidence that either diagnostic is superior and does not define a feature-selection rule.

### Status

Saved the non-overwriting analysis as `v6_medium_importance_001`, including raw fold importance, cross-fold summaries, group summaries, exact fold-specific IC joins, rank correlations, relationship summaries, metadata, and a completion marker. No model was trained, no prediction or NCORR was recomputed, no validation or holdout target was accessed, and no redundancy, permutation, ablation, or selection analysis was performed.

---

## 2026-11-23 — Task 19E: Fold-Specific Top-250 IC Selection Implementation

### Predeclared policy

Implemented the single policy `medium_top250_abs_mean_ic_v1`. Within each approved fold, the selector ranks the official 350 Medium numeric features by descending absolute mean Spearman IC computed over that fold's exact training dates, requires valid IC observations, resolves ties by feature name, selects exactly 250, and restores official Medium order for LightGBM input. Negative IC is therefore eligible by magnitude; no sign flip or feature transformation is applied.

`K=250` was fixed before a candidate run as one bounded reduction of 100 features from Medium. It was not tuned, and no alternative K or policy was implemented. Gain/split importance and Validation targets are not inputs because the question is specifically whether a predeclared fold-as-of standalone-IC screen can improve the existing Medium baseline.

### Leakage and artifact safeguards

The selector validates `v6.0`, `target_jupiter_60`, the exact ordered Medium universe, all five approved fold identities, each fold's full training-date list and cutoff, and the source analysis identity. It rejects mixed or later-fold summaries, unexpected cutoffs, incomplete feature identities, undefined selections, and inconsistent artifacts. Consequently same-fold Validation, purge, pre-holdout gap, final holdout, and immature dates cannot enter a fold's ranking.

The pipeline now accepts a fold-specific feature manifest while retaining the prior global feature path when selection is disabled. A selected run records one deterministic JSON manifest per fold with its policy, source IC analysis, cutoff, ranks and IC statistics, official-order model features, hashes, model configuration, seed, and Git state. Existing baseline artifacts remain readable and unchanged.

### Future experiment and status

The predeclared future command is:

```powershell
.\.venv\Scripts\python.exe -B -m src.pipeline --version v6.0 --feature-set medium --model-config baseline --selection-policy medium_top250_abs_mean_ic_v1 --feature-quality-id v6_medium_ic_001 --experiment-id v6_medium_top250_ic_002 --run --confirm-full-run
```

This command was not executed in Task 19E. No real model was trained, no prediction or NCORR was calculated, and no performance conclusion exists. The policy remains limited by univariate monotonic association and may discard features whose value is conditional, nonlinear, or interaction-dependent. The final holdout remains sealed.

---

## 2026-11-23 — Task 19F: Real Fold-Specific Top-250 IC Experiment

### Execution and integrity

Executed exactly one predeclared candidate, `v6_medium_top250_ic_002`, with `medium_top250_abs_mean_ic_v1`, baseline LightGBM parameters, `target_jupiter_60`, the approved five expanding folds, and the 16-date purge. The documented command completed in approximately 14 minutes 30 seconds. The artifact contains five readable models, five fold-specific manifests sourced from `v6_medium_ic_001`, 2,180,295 OOF rows, and 330 dates. Canonically sorted row identities match `v6_medium_baseline_001` exactly; there are no duplicates or gap, holdout, or immature dates.

### Aggregate and paired comparison

Mean official NCORR is `0.0263268391`, standard deviation `0.0177141064`, Sharpe-like `1.4862075749`, and positive-date percentage `91.8182%`. Medium records `0.0261214141`, `0.0179810967`, `1.4527152902`, and `93.3333%`. Mean Legacy CORR is `0.0225756881` versus `0.0224102507`.

The paired Top-250-minus-Medium NCORR difference has mean `0.0002054250`, standard deviation `0.0116841383`, median `0.0001086914`, and is positive on 50.6061% of dates. Its 10th/90th percentiles are `-0.01418857` and `+0.01405334`; the largest decline/improvement are `-0.04185361` and `+0.02738511`. The ten largest absolute differences account for 10.05% of total absolute movement. The small mean is not a statistical-significance claim, and weekly dependence has not been modeled.

### Temporal results

Top-250 fold NCORR is `0.02012810`, `0.03160391`, `0.02825315`, `0.02582942`, and `0.02412886`; the corresponding Medium differences are `-0.00578544`, `+0.00149703`, `+0.00079637`, `+0.00335589`, and `+0.00435610`. Top-250 annual NCORR for 2019 through 2025 is `0.02251776`, `0.01386994`, `0.04098789`, `0.02656071`, `0.03547918`, `0.01946007`, and `0.02266710`; differences are `-0.00705978`, `-0.00116433`, `+0.00168056`, `-0.00035242`, `+0.00068384`, `+0.00622100`, and `+0.00386168`. Partial 2025 contains 17 dates. The candidate's aggregate improvement is temporally heterogeneous, with early underperformance and stronger 2024–2025 results. No causal explanation is established.

### Feature-list stability and decision status

There are 212 features shared by all five manifests, 17 selected in only one fold, and 286 selected at least once. Pairwise Jaccard similarity ranges `0.8182`–`0.8939`. Small membership is 65, 65, 66, 67, and 67 by fold; Medium-only membership is 185, 185, 184, 183, and 183. These changes reflect expanding-history IC ranks and are descriptive only.

This was one fixed-K comparison, not a K search. The slight aggregate gain, lower positive-date percentage, Fold 1 loss, and later-period improvement warrant caution. No additional candidate, new IC audit, Gain-based selection, redundancy analysis, NMMC/NBMC proxy, or holdout evaluation was performed. The final holdout remains sealed.

---

## 2026-11-23 — Task 20A: Fixed v6 Ridge Baseline Infrastructure

### Predeclared model

Added one Ridge configuration for the official 350 numeric Medium features using the existing v6 folds, purge, OOF identity, official NCORR, Legacy CORR, and artifact contracts. The objective is `mean squared error + alpha * L2 coefficient norm`, with fixed `alpha=0.001`. This normalized convention is invariant to fold row count and corresponds to unnormalized penalty `n_samples * 0.001`. The value represents modest shrinkage after standardization and was fixed without inspecting Ridge OOF performance. No hyperparameter search or target-aware selection was introduced.

### Leakage-safe preprocessing and solver

Every fold independently learns non-missing training means for imputation and population standard deviations for scaling. These parameters are frozen for that fold's Validation rows. Constant, near-constant, and all-missing training columns are handled without division by zero; categorical features, infinities, wrong feature order, null targets, and misaligned indexes are rejected. The closed-form solver is deterministic, so a random state is not operationally applicable.

Training uses two bounded passes over 20,000-row float32 batches. Only the 350×350 Gram matrix, cross-product vector, and final linear solve use float64. This avoids a full standardized 6.28M×350 matrix; approximate per-batch feature buffers are 28 MB float32 plus 56 MB float64. A 1,024×350 synthetic fit verified the full Medium width and bounded batch contract. This does not prove full-fold runtime or guarantee 32 GB feasibility.

### Pipeline, artifacts, and status

The existing CLI now accepts `--model-config ridge_baseline`. Five fold models will be written as versioned JSON containing ordered feature names, means, scales, coefficients, intercept, alpha, batch configuration, sample count, and observed maximum batch size. Reloaded synthetic models produce identical predictions. LightGBM behavior and its text artifacts remain unchanged, including fold-specific Top-250 compatibility.

The predeclared future command is:

```powershell
.\.venv\Scripts\python.exe -B -m src.pipeline --version v6.0 --feature-set medium --model-config ridge_baseline --experiment-id v6_medium_ridge_003 --run --confirm-full-run
```

The command was not executed. There is no real Ridge NCORR, Legacy CORR, OOF prediction, or complementarity evidence. No Elastic Net, Lasso, feature selection, LightGBM rerun, scoring change, or holdout access occurred.

---

## 2026-11-23 — Task 20B: Real v6 Medium Ridge OOF Experiment

### Execution and artifact integrity

Executed exactly one fixed Ridge experiment, `v6_medium_ridge_003`, with the predeclared normalized alpha `0.001`, all 350 ordered Medium features, training-only imputation/scaling, the approved five folds, and 16-date purge. The exact documented command completed in approximately 6 minutes 48 seconds. The five JSON models reload successfully and reproduce all saved predictions exactly. The OOF artifact has 2,180,295 finite predictions over 330 approved dates, zero duplicate identities, correct fold assignments, canonical identity equality with `v6_medium_baseline_001`, and no gap, holdout, or immature rows.

### Aggregate and paired results

Ridge official NCORR has mean `0.0180930223`, weekly standard deviation `0.0172147902`, mean/std ratio `1.0510161392`, and positive-date percentage `85.7576%`. Medium LightGBM records `0.0261214141`, `0.0179810967`, `1.4527152902`, and `93.3333%`. Ridge mean Legacy CORR is `0.0157882346` versus `0.0224102507`.

The paired Ridge-minus-LightGBM NCORR difference has mean `-0.0080283917`, standard deviation `0.0174471207`, median `-0.0091335188`, and is positive on 34.2424% of dates. Its 10th/90th percentiles are `-0.03031692` and `+0.01486321`; its largest decline/improvement are `-0.06445878` and `+0.03349593`. The ten largest absolute moves account for 9.64% of absolute movement. These weekly mean/std ratios are not annualized investment Sharpe ratios, and no dependence-aware significance test was performed.

### Fold, annual, and training diagnostics

Fold NCORR is `0.01294677`, `0.01982222`, `0.01757355`, `0.02423462`, and `0.00853773`; differences are `-0.01296676`, `-0.01028466`, `-0.00988324`, `+0.00176109`, and `-0.01123502`. Annual NCORR for 2019–2025 is `0.01357569`, `0.00474889`, `0.03054062`, `0.02208528`, `0.01591741`, `0.02485467`, and `0.00768142`; differences are `-0.01600185`, `-0.01028538`, `-0.00876671`, `-0.00482786`, `-0.01887794`, `+0.01161561`, and `-0.01112400`. Only Fold 4 and 2024 outperform Medium; no causal explanation is assigned.

Training sample counts are 4,227,405; 4,715,780; 5,231,777; 5,767,397; and 6,284,208. Recorded model-fit times are 27.09, 30.96, 34.02, 37.10, and 40.54 seconds. Every fold uses at most 20,000 rows per batch, has 350 finite nonzero coefficients, zero constant/near-constant features, and zero entirely missing features. Coefficient norms are `0.016052`, `0.015363`, `0.014383`, `0.013703`, and `0.013355`; JSON files are approximately 37–38 KiB. A reliable process peak-memory reading was unavailable. Nonzero Ridge coefficients are not interpreted as LightGBM Gain importance.

### Research interpretation

The fixed Ridge baseline materially underperforms LightGBM overall, although its 2024 behavior differs. That isolated reversal does not prove complementarity, justify ensembling, or identify a cause. No alpha tuning, additional linear model, feature selection, LightGBM rerun, ensemble experiment, or holdout evaluation occurred. The final holdout remains sealed.

---

## 2026-11-23 — Task 20C: Three-Model Saved-Prediction Diversity Audit

### Scope and integrity

Analyzed saved OOF predictions from `v6_medium_baseline_001` (A), `v6_medium_top250_ic_002` (B), and `v6_medium_ridge_003` (C). The analyzer canonicalized the four row-identity fields rather than trusting physical Parquet order. All experiments have identical 330 dates, 2,180,295 identities, fold assignments, target/scoring metadata, and NCORR references, with zero duplicates, missing/non-finite predictions, or protected dates. No target column was selected, no NCORR was recomputed, and no model or ensemble prediction was created.

The exact command was:

```powershell
.\.venv\Scripts\python.exe -B -m src.analyze_prediction_diversity --version v6.0 --analysis-id v6_three_model_diversity_001 --experiment-a v6_medium_baseline_001 --experiment-b v6_medium_top250_ic_002 --experiment-c v6_medium_ridge_003
```

### Overall agreement

A–B Pearson has mean/median/std/min/max `0.7283`/`0.7309`/`0.0129`/`0.6985`/`0.7534`; Spearman is `0.7275`/`0.7297`/`0.0119`/`0.6916`/`0.7522`. A–C Pearson is `0.5238`/`0.5227`/`0.0159`/`0.4835`/`0.5636`; Spearman is `0.5271`/`0.5261`/`0.0156`/`0.4904`/`0.5672`. B–C Pearson is `0.5145`/`0.5165`/`0.0175`/`0.4686`/`0.5630`; Spearman is `0.5181`/`0.5205`/`0.0161`/`0.4794`/`0.5513`.

No pair exceeds 0.9 on any date. A–B never falls below 0.5; A–C falls below 0.5 on 4.85% Pearson and 3.33% Spearman dates, while B–C does so on 20.61% and 16.06%. Mean percentile-rank disagreement is `0.1532`, `0.2174`, and `0.2198`. Mean top/bottom-decile overlap is 53.43%/52.53%, 33.34%/33.83%, and 32.87%/32.95% for A–B, A–C, and B–C.

### Temporal findings

A–B Spearman is comparatively stable and rises from `0.7140` in 2019 to `0.7356` in 2024 and `0.7324` in partial 2025. A–C is `0.5478` in 2021 but falls to `0.5114` in 2024 and `0.5042` in partial 2025. B–C falls from `0.5302` in 2021 to `0.4977` and `0.4929`. Fold results agree: A–C/B–C end at `0.5042`/`0.4930` in Fold 5, with rank disagreements `0.2233`/`0.2268` and top-decile overlaps `31.41%`/`31.26%`.

The lowest A–C agreement dates are concentrated mainly in early 2025; the lowest B–C dates mainly in late 2024 and early 2025. Still, full-period tree–Ridge Spearman standard deviations are only about `0.016`, so moderate disagreement is persistent, not solely an isolated-period phenomenon.

### Interpretation and status

Ridge predictions are materially less correlated with either LightGBM than the two LightGBMs are with each other, particularly recently. This is a diversity observation only. Low prediction correlation does not establish positive ensemble value, incremental alpha, or a valid weighting rule—especially because the Ridge baseline underperforms. Saved artifacts include 990 pair-date rows, fold/year summaries, rank and decile diagnostics, metadata, research summary, and a completion marker under `v6_three_model_diversity_001`. No new performance result or ensemble was produced, and the final holdout remains sealed.
