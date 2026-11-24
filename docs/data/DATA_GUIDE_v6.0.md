# Numerai v6.0 Data Guide

This document records verified metadata and a selective, memory-bounded audit of the official v6.0 files. It does not include predictive modeling or feature selection.

## Metadata Source

- Dataset version: `v6.0`
- File: `data/v6.0/features.json`
- File size: 47,870 bytes
- SHA-256: `ee6c00bc9d5cb78aaab986630138512945301a3c588865ad61da5d0dea5ff634`
- Top-level keys: `feature_sets`, `targets`

Reproduce the metadata download from the repository root with:

```powershell
python -m src.download_data --version v6.0 --file v6.0/features.json
```

If the file already exists, the downloader reports that it was skipped and leaves its contents unchanged.

## Training and Validation Files

Reproduce the dataset download from the repository root with:

```powershell
.\.venv\Scripts\python.exe -B -m src.download_data --version v6.0 --file train.parquet --file validation.parquet
```

| File | Size (bytes) | SHA-256 | Rows | Columns | Row groups |
|---|---:|---|---:|---:|---:|
| `data/v6.0/train.parquet` | 2,176,263,434 | `ac81951d1ff5925475e015d83d1252483e9c84d60f6963b2fdae7a197210bca0` | 4,329,443 | 1,068 | 22 |
| `data/v6.0/validation.parquet` | 1,410,842,073 | `2f4ee8b9a0a9767286663316d1a89b477d03160f6720d472cd132532a0d52a81` | 2,746,248 | 1,068 | 14 |

Both Parquet footers are readable and report `parquet-cpp-arrow version 23.0.1` as the writer. Their column names and physical schemas are identical.

Verified schema composition:

| Column category | Count | Parquet physical types |
|---|---:|---|
| Features | 1,040 | 1 `BYTE_ARRAY`, 1,017 `INT32`, 22 `FLOAT` |
| Targets | 24 | 24 `FLOAT` |
| Other columns | 4 | 4 `BYTE_ARRAY` |

The four non-feature, non-target columns are:

| Column | Parquet physical type | Logical type |
|---|---|---|
| `numerai_ticker` | `BYTE_ARRAY` | String |
| `composite_figi` | `BYTE_ARRAY` | String |
| `date` | `BYTE_ARRAY` | String |
| `data_type` | `BYTE_ARRAY` | String |

`feature_country` is the one feature stored as `BYTE_ARRAY` with logical type String. The 1,017 `INT32` features carry an 8-bit signed-integer logical annotation, and the other 22 features are stored as `FLOAT`. All 24 targets are stored as `FLOAT`.

The 1,040 feature names and 24 target names in each Parquet schema exactly match the `all` feature set and target list in `features.json`.

## Feature Sets

| Feature set | Feature count |
|---|---:|
| `small` | 101 |
| `medium` | 351 |
| `all` | 1,040 |
| `signals` | 23 |

There are 1,040 unique feature names across all feature sets, with no duplicate names within an individual set.

Verified subset relationships:

- `small` is a subset of `medium` and `all`.
- `medium` is a subset of `all`.
- `signals` is a subset of `all`.
- `small` and `signals` overlap by one feature.
- `medium` and `signals` overlap by one feature.
- `all` contains 1,017 features outside `signals`.

The literal feature name `feature_country` appears in both `signals` and `all`. The metadata file does not provide feature descriptions, tags, categorical annotations, or financial interpretations. The Parquet schema identifies `feature_country` as a string.

## Targets

The metadata lists 24 target names:

- `target`
- `target_raw_return_20`
- `target_raw_return_60`
- `target_factor_neutral_20`
- `target_factor_neutral_60`
- `target_factor_feat_neutral_20`
- `target_factor_feat_neutral_60`
- `target_camille_20`
- `target_camille_60`
- `target_sydney_20`
- `target_sydney_60`
- `target_irina_20`
- `target_irina_60`
- `target_alexandra_20`
- `target_alexandra_60`
- `target_colleen_20`
- `target_colleen_60`
- `target_jean_20`
- `target_jean_60`
- `target_eleven_4`
- `target_eleven_20`
- `target_eleven_60`
- `target_jupiter_20`
- `target_jupiter_60`

Target-name suffix inventory:

| Name pattern | Count |
|---|---:|
| Unsuffixed (`target`) | 1 |
| Suffix `_4` | 1 |
| Suffix `_20` | 11 |
| Suffix `_60` | 11 |

These suffixes are part of the target names; `features.json` does not separately define their units or horizons. Both `target_jupiter_20` and `target_jupiter_60` are explicitly available. The file does not define what the generic `target` aliases, so the V1 research target should not be selected from this metadata alone.

## Confirmed Metadata Differences from v5.3

| Metadata item | v5.3 | v6.0 |
|---|---:|---:|
| Feature sets | 19 | 4 |
| `small` features | 42 | 101 |
| `medium` features | 780 | 351 |
| `all` features | 3,555 | 1,040 |
| Listed targets, including `target` | 41 | 24 |

These comparisons describe only the two versions' `features.json` files. They do not establish performance, distributions, or schema compatibility.

## Reproducible Schema Audit

Run the version-aware audit from the repository root with:

```powershell
.\.venv\Scripts\python.exe -B -m src.audit_data --version v6.0
```

The command writes `outputs/v6.0/data_audit/audit.json` and refuses to overwrite an existing report. Exact statistics stream selected columns by row group. Integer-feature ranges and cardinalities are sampled from the first 2,000 rows of five evenly spaced row groups in each file (20,000 rows total). The report records these scopes and the selected row groups.

## Temporal Structure and Splits

Dates are strings formatted `YYYY-MM-DD`. Both files are chronologically ordered, every observed consecutive date is exactly seven days apart, and Train and Validation share no dates.

| File / partition | Rows | Dates | First date | Last date | Rows per date (min / median / max) |
|---|---:|---:|---|---|---|
| Train / `train` | 4,329,443 | 835 | 2003-01-03 | 2018-12-28 | 2,399 / 5,217 / 6,778 |
| Validation / all | 2,746,248 | 411 | 2019-01-04 | 2026-11-13 | 6,085 / 6,670 / 7,208 |
| Validation / `validation` | 2,653,645 | 398 | 2019-01-04 | 2026-08-14 | — |
| Validation / `test` | 92,603 | 13 | 2026-08-21 | 2026-11-13 | — |

There are no duplicate `numerai_ticker` values within a date in either file. Validation therefore contains an embedded `test` partition comprising its latest 13 dates; this is a structural observation, not a finalized research holdout policy.

## Focused Target Audit

All four inspected targets are Parquet `FLOAT` columns with observed values limited to `0`, `0.25`, `0.5`, `0.75`, and `1`.

| File | Target | Null rows | Null % | Dates available | First / last available date | Observed counts for 0 / .25 / .5 / .75 / 1 |
|---|---|---:|---:|---:|---|---|
| Train | `target` | 0 | 0 | 835 | 2003-01-03 / 2018-12-28 | 217,295 / 865,065 / 2,164,723 / 865,900 / 216,460 |
| Train | `target_jupiter_60` | 0 | 0 | 835 | 2003-01-03 / 2018-12-28 | 217,295 / 865,065 / 2,164,723 / 865,900 / 216,460 |
| Train | `target_jupiter_20` | 0 | 0 | 835 | 2003-01-03 / 2018-12-28 | 217,295 / 865,065 / 2,164,723 / 865,900 / 216,460 |
| Train | `target_eleven_4` | 0 | 0 | 835 | 2003-01-03 / 2018-12-28 | 216,180 / 865,740 / 2,165,671 / 865,540 / 216,312 |
| Validation | `target` | 92,603 | 3.3720 | 398 | 2019-01-04 / 2026-08-14 | 133,084 / 530,337 / 1,326,803 / 530,735 / 132,686 |
| Validation | `target_jupiter_60` | 92,603 | 3.3720 | 398 | 2019-01-04 / 2026-08-14 | 133,084 / 530,337 / 1,326,803 / 530,735 / 132,686 |
| Validation | `target_jupiter_20` | 35,667 | 1.2988 | 406 | 2019-01-04 / 2026-10-09 | 135,938 / 541,716 / 1,355,273 / 542,122 / 135,532 |
| Validation | `target_eleven_4` | 14,227 | 0.5181 | 409 | 2019-01-04 / 2026-10-30 | 136,515 / 546,257 / 1,366,624 / 546,131 / 136,494 |

`target` and `target_jupiter_60` are identical everywhere both are non-null: 4,329,443 compared Train rows and 2,653,645 compared Validation rows, with zero mismatches. The newest Validation dates show horizon-dependent target maturity: the trailing 13 dates are entirely null for `target`/`target_jupiter_60`, five for `target_jupiter_20`, and two for `target_eleven_4`.

## Feature Types and Missingness

The schema contains 1,017 signed 8-bit integer features, 22 floating-point features, and the string-valued `feature_country`. These columns exactly match the 1,040 features listed in `features.json` in both files.

Exact selected-column results:

- `feature_country` has no nulls, with 47 observed categories in Train and 46 in Validation.
- Five of the 22 floating-point features have no nulls in either file; the other 17 have some missing values.
- Train floating-feature missingness ranges from 4 rows (0.000092%) for `feature_adv_20d_factor` to 309,170 rows (7.1411%) for `feature_momentum_52w_less_4w_factor`.
- Validation floating-feature missingness ranges from 117 rows (0.004260%) for `feature_earnings_yield_factor` among features with missing values to 116,570 rows (4.2447%) for `feature_momentum_52w_less_4w_factor`.
- Per-feature exact null counts and percentages are retained in the JSON report.

In the bounded 20,000-row integer-feature sample, every one of the 1,017 integer features had observed range `0`–`4`, five distinct observed values, and the observed set `{0, 1, 2, 3, 4}`. These are sampled statistics and do not prove full-dataset ranges or missingness.

## Initial v6 Model Feature Contract

The first v6 LightGBM baseline resolves the official `medium` feature set directly from `features.json` and verifies every selected name against the Train Parquet schema.

- Official `medium` metadata entries: 351
- Excluded categorical entry: `feature_country`
- Numeric model features: 350
- Ordering: unchanged from the official `medium` list after exclusion
- Feature engineering or target-based selection: none

The resolver is reusable for other explicitly named official feature sets and determines numeric eligibility from the Parquet schema rather than assuming v5.3 names or counts. Selected model columns must be numeric and appear in the exact resolved order. Integer and floating-point NaNs are passed through for LightGBM's native missing-value handling; no global imputation or normalization is introduced.

Model input is separated into:

- Features: the ordered numeric feature matrix only
- Target: explicit `target_jupiter_60`
- Identity: `numerai_ticker`, `composite_figi`, and `date`

Identity columns retain their original row index for later prediction alignment and are never passed to LightGBM as features. Ordinary model preparation consumes only the approved historical-training and research-validation loaders; it does not bypass gap, holdout, or immature-date safeguards.

## Classic v6 Scoring Verification

Scoring was verified on 2026-11-21 against the [official Classic scoring documentation](https://docs.numer.ai/numerai-tournament/scoring/), the public round configuration API, and [`numerai-tools` scoring version 0.7.2](https://github.com/numerai/numerai-tools/blob/master/numerai_tools/scoring.py). Public round 1404 selected `neutral_corr` v1 at 8× and `neutral_mmc` v1 at 32× for payouts. `neutral_bmc` v1, the canonical v0 companion series, and `season_score` had zero payout multipliers.

Metric identities must remain distinct:

| Metric | Official implementation | Required inputs | Round-1404 role | Offline historical status |
|---|---|---|---|---|
| NCORR / `neutral_corr` v1 | `neutral_correlation` | Predictions, v6 neutralizers, Jupiter-60 target | Payout, 8× | Implemented with official `numerai-tools==0.7.2` |
| NMMC / `neutral_mmc` v1 | `neutral_contribution` | Predictions, v6 neutralizers, `classic_v6_nuswmm_pow1`, Jupiter-60 target | Payout, 32× | Exact historical Meta Model is absent from the public dataset catalog and no supported public acquisition API was verified; unavailable |
| NBMC / `neutral_bmc` v1 | `neutral_contribution` | Predictions, v6 neutralizers, exact shared v6 benchmark reference, Jupiter-60 target | Informational | Benchmark-model files are catalogued, but official documentation does not identify the exact column or blend representing the shared benchmark; unavailable |
| `NUMERAI_CORR_JUPITER_60` | `numerai_corr` | Predictions and Jupiter-60 target | Research diagnostic; not a current v6 payout score | Implemented locally |
| Legacy MMC | `correlation_contribution` | Predictions, matching historical Meta Model, configured target | Not selected in round 1404 | No matching reference is local; unavailable |
| Canon neutral companions | Server/API score configurations version 0 | Resolved corresponding live neutral scores | Informational, non-payout | Live/API series; no separate local proxy |
| Season score | Round-configured score aggregation | Resolved payout scores and multipliers | Informational, non-payout in round 1404 | Live-round aggregation only |
| Neutral churn | Five preceding on-time neutralized submissions | Current and prior live submissions plus official neutralization | Live diagnostic/payout adjustment context | Not a historical validation metric |

Official v6 diagnostics also expose neutral correlation, benchmark contribution, and neutral churn. Older general diagnostics such as legacy CORR/MMC, FNC, CWMM, maximum/average model correlation, and historical BMC retain their own identities and input requirements; they are not silently renamed to the current neutral scores.

### Validation neutralizer reference

The official catalog lists `v6.0/train_neutralizer.parquet`, `v6.0/validation_neutralizer.parquet`, and `v6.0/live_neutralizer.parquet`. Task 17C downloaded only the Validation reference using the repository downloader:

```powershell
.\.venv\Scripts\python.exe -B -m src.download_data --version v6.0 --file validation_neutralizer.parquet
```

Verified local file facts:

- Path: `data/v6.0/validation_neutralizer.parquet`
- Size: 4,699,025,422 bytes
- SHA-256: `1ae870b17d501b8d9ec114ab2096eece342812c424a826f44a772087a82c6c6f`
- Parquet: 2,746,248 rows, 304 columns, 411 row groups
- Schema: 300 `float32` neutralizer columns and four string columns: `numerai_ticker`, `date`, `data_type`, and `composite_figi`
- Dates: 411 dates from 2019-01-04 through 2026-11-13
- Alignment: all 2,746,248 rows match Validation in the same order on `numerai_ticker`, `date`, and `composite_figi`
- Research coverage: all 330 approved Research Validation dates are covered

The permanent verification command is:

```powershell
.\.venv\Scripts\python.exe -B -m src.audit_scoring_references --version v6.0
```

It hashes the file, reads its footer, and compares only identity/date columns. It does not read target columns or calculate a score.

### Unresolved contribution references and dependency

The verified v6 catalog contains `train_benchmark_models.parquet`, `validation_benchmark_models.parquet`, and `live_benchmark_models.parquet`, but contains no file named for `classic_v6_nuswmm_pow1`. NumerAPI 3.1.1 exposes no supported public method for retrieving that historical Meta Model series. NMMC therefore cannot be reproduced offline with currently verified inputs.

Official documentation identifies NBMC as contribution against a shared Classic v6 / Signals v3.1 benchmark, but does not map that reference to a specific column or blend in `validation_benchmark_models.parquet`. That file was not downloaded because its exact applicability remains unresolved. A catalog filename is not treated as proof of metric identity.

The official scoring package `numerai-tools==0.7.2` is installed and pinned in `requirements.txt`. Its declared Python requirement is 3.11 or newer and its pandas, NumPy, SciPy, and scikit-learn ranges are compatible with the local Python 3.12 environment. Reproduce the environment with:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Implemented official NCORR

The repository calls `numerai_tools.scoring.neutral_correlation` directly. For each date it tie-kept-ranks and Gaussianizes predictions through the official function, neutralizes them against all 300 official columns, and correlates the residual with `target_jupiter_60`. Metric identity is preserved as display name `NCORR`, API name `neutral_corr`, version `1`.

Scoring is cross-sectional and independent by weekly date. Aggregate NCORR is the arithmetic mean of the per-date scores; fold size does not give the final 18-date fold equal weight with a 78-date fold. Saved metadata records the exact date coverage, `numerai-tools` version, dataset version, and neutralizer SHA-256.

To keep memory bounded, the loader uses a Parquet date filter and requests only the three identity fields plus the ordered 300 neutralizer columns for one date. It requires an exact row-for-row match on `date`, `numerai_ticker`, and `composite_figi`; missing, duplicate, reordered, or truncated rows and null/nonfinite neutralizer values are rejected. Official null `composite_figi` values are retained and must match on both sides; `date` and `numerai_ticker` may not be null. Only approved Research Validation dates are accepted, preventing gap, holdout, and immature dates from entering ordinary scoring.

`NUMERAI_CORR_JUPITER_60` remains a separately named Legacy CORR diagnostic without neutralization. It is never reported as NCORR. NMMC and NBMC remain unimplemented because their exact official reference series are unavailable or unresolved; no proxy is used.

Run the permanent synthetic verification with:

```powershell
.\.venv\Scripts\python.exe -B -m unittest tests.test_ncorr tests.test_evaluation -v
```

The direct comparison to the installed official function uses an absolute tolerance of `1e-12`.

### Implemented Jupiter-60 Research Diagnostic

For each date, `NUMERAI_CORR_JUPITER_60` follows the official `numerai_corr` formula:

1. Tie-kept percentile-rank predictions as `(rank - 0.5) / count`.
2. Gaussianize ranked predictions with the standard normal inverse CDF.
3. Center `target_jupiter_60` on zero.
4. Apply the signed power `sign(x) × abs(x)^1.5` to both series.
5. Calculate their Pearson correlation.

This diagnostic is suitable for historical research because its required inputs are local. It is not neutralized and must never be reported as NCORR.

### OOF Prediction and Aggregation Contract

Research OOF artifacts contain `date`, `numerai_ticker`, `composite_figi`, `fold_id`, `prediction`, and `target_jupiter_60`. Validation requires every one of the 330 approved research dates to appear in its configured fold and rejects dates from the pre-holdout gap, sealed holdout, or immature partition.

Scores are first calculated cross-sectionally per date. Overall mean, standard deviation, Sharpe-like ratio, and positive-date percentage are then computed across the 330 date scores. Per-fold and per-year summaries are derived from the same per-date series; overall results never average the five fold means equally.

The first real v6 OOF baseline is recorded below. No holdout score, NMMC, or NBMC has been generated.

## Approved V1 Research Validation Protocol

The v6 workflow explicitly selects `target_jupiter_60` and uses `date` as the time column. Dates are ordered from the values actually observed in the Parquet files; folds do not use integer or calendar-date subtraction.

| Partition | Date range | Dates | Research use |
|---|---|---:|---|
| Historical training | 2003-01-03–2018-12-28 | 835 | Expanding fold training |
| Research validation | 2019-01-04–2025-04-25 | 330 | Out-of-fold research evaluation |
| Final holdout gap | 2025-05-02–2025-08-15 | 16 | Excluded from fitting and selection |
| Sealed final holdout | 2025-08-22–2026-08-14 | 52 | Reserved for one final evaluation |
| Immature | 2026-08-21–2026-11-13 | 13 | Excluded while Jupiter-60 is null |

Ordinary research loaders expose target values only for historical training and research validation. Gap, holdout, and immature partition inspection is identity-only. The implementation rejects missing approved dates, new unexpected dates, unexpected `data_type` values, and null Jupiter-60 labels inside approved research history rather than silently shifting boundaries or reducing the sample.

### Expanding Walk-Forward Folds

Each validation window is preceded by a purge of 16 observed dates. Training starts on 2003-01-03 and expands through all eligible history before that purge. The final 18-date partial fold is retained.

| Fold | Train end | Purge | Validation | Dates |
|---:|---|---|---|---:|
| 1 | 2018-09-07 | 2018-09-14–2018-12-28 | 2019-01-04–2020-06-26 | 78 |
| 2 | 2020-03-06 | 2020-03-13–2020-06-26 | 2020-07-03–2021-12-24 | 78 |
| 3 | 2021-09-03 | 2021-09-10–2021-12-24 | 2021-12-31–2023-06-23 | 78 |
| 4 | 2023-03-03 | 2023-03-10–2023-06-23 | 2023-06-30–2024-12-20 | 78 |
| 5 | 2024-08-30 | 2024-09-06–2024-12-20 | 2024-12-27–2025-04-25 | 18 |

Verify these boundaries from the local Parquet date metadata without loading features or targets:

```powershell
.\.venv\Scripts\python.exe -B -m src.validation --version v6.0
```

Known limitations:

- The approved boundaries are snapshot-specific and intentionally fail verification when the file date coverage changes.
- The 16-date purge is configured for Jupiter-60 and must be reconsidered for another target horizon.
- The short final fold contributes only 18 dates and should not receive equal fold weight in aggregate evaluation.
- Two controlled real v6 LightGBM runs now compare official Medium and Small feature sets; no feature selection, engineering, tuning, or holdout evaluation has been performed.
- The reusable interfaces have synthetic coverage and two completed real walk-forward experiments; broader model comparisons remain future work.

## Integrated Research Data Flow

The v6 pipeline uses the centralized `v6.0` research configuration and proceeds as follows:

1. Resolve the official Medium metadata set and retain its 350 numeric features in official order; `feature_country` is excluded from the baseline model interface.
2. Load only those features, `target_jupiter_60`, `data_type`, `date`, `numerai_ticker`, and `composite_figi` through the protected historical and Research Validation loaders.
3. Generate the fixed five expanding folds with the approved 16-observed-date purge and 78/78/78/78/18 validation windows.
4. Fit a new baseline LightGBM model on each fold's expanding training dates. Validation targets are not supplied to fitting or early stopping.
5. Preserve source row identity and fold assignment in one validated OOF table covering all 330 research dates.
6. Score each date using official NCORR and separately named Legacy CORR, then save date-level, overall, and fold-level summaries.

The historical and research frames are loaded once with 350 features rather than all 1,040. Only one fold model and its training/validation matrices are retained at a time; each model is serialized before its large intermediates are released. Neutralizers remain date-filtered and are loaded one week at a time. Exact peak memory depends on pandas/LightGBM representations and has not been measured; no unsupported numerical estimate is asserted.

Ordinary loaders never expose gap, sealed-holdout, or immature target data to this pipeline. OOF validation rejects any such date. The real CLI additionally requires the explicit `--run --confirm-full-run` pair, while `--smoke` uses synthetic data only.

Completed experiments contain configuration, ordered features, fold boundaries, OOF predictions, per-date NCORR and Legacy CORR, aggregate/fold metrics, five LightGBM text models, scoring-reference identities, and a final `COMPLETED.json` marker. Existing experiment directories are never overwritten. Interrupted directories without the marker are rejected by the artifact loader.

## First Real v6 Baseline

Experiment `v6_medium_baseline_001` completed on 2026-11-21 using exactly the documented configuration:

- Dataset: `v6.0`
- Features: official Medium set, 350 numeric columns in metadata order
- Target: `target_jupiter_60`
- Model: baseline LightGBM (`n_estimators=300`, `learning_rate=0.05`, `num_leaves=31`, `random_state=42`, `n_jobs=-1`)
- Validation: five approved expanding folds, 16 observed-date purge, 330 OOF dates
- Scoring: NCORR / `neutral_corr` v1 plus separately named Legacy CORR
- Neutralizer SHA-256: `1ae870b17d501b8d9ec114ab2096eece342812c424a826f44a772087a82c6c6f`

Verification found 2,180,295 OOF rows from 2019-01-04 through 2025-04-25, with no duplicate identities, missing predictions, or gap/holdout/immature dates. The five expanding training sets contained 4,227,405; 4,715,780; 5,231,777; 5,767,397; and 6,284,208 rows.

The run took approximately 18 minutes wall-clock. The pipeline did not instrument exact fold or scoring durations. Model serialization timestamps show increasing intervals as training expanded, and the final NCORR plus artifact phase took several minutes. Windows process isolation prevented a reliable peak-memory measurement. The run completed without memory errors; peak usage remains an explicit unknown for future operational profiling.

Headline research results:

| Metric | Mean | Std. dev. | Sharpe-like | Positive dates |
|---|---:|---:|---:|---:|
| NCORR | 0.026121 | 0.017981 | 1.4527 | 93.33% |
| Legacy CORR | 0.022410 | 0.016756 | 1.3374 | 91.21% |

These are Research Validation results, not sealed-holdout results. They must not be interpreted as evidence from the final holdout.

## Small Feature-Set Experiment

Experiment `v6_small_baseline_002` verified the official `small` metadata set against the v6 Train schema:

- Numeric Small features: 100
- Numeric Medium features: 350
- Relationship: every Small feature is contained in Medium and appears in the same relative official order
- Excluded categorical feature: `feature_country` is not part of the numeric model matrix
- OOF rows and dates: identical to Medium, 2,180,295 rows across 330 dates

Only the feature list changed. Dataset version, `target_jupiter_60`, baseline LightGBM parameters, random seed, expanding folds, purge, neutralizer checksum, and NCORR implementation were identical. Artifact validation found no duplicate identities or gap/holdout/immature rows.

Small completed in approximately nine minutes versus approximately eighteen minutes for Medium. Exact peak memory and component timings were not instrumented; the reduced feature matrix and LightGBM bin count were observed, but no unsupported memory estimate is made.

Small's overall NCORR was 0.015617 with Sharpe-like ratio 0.9768, versus Medium's 0.026121 and 1.4527. Small's NCORR standard deviation was lower (0.015988 versus 0.017981), but the reduction in mean was substantially larger. Small performed better in 2024 and partial 2025 while underperforming in the five earlier calendar periods. These observations describe temporal outcomes only and do not identify causal feature importance.

## All-Feature Memory Profile

The official `all` metadata set contains 1,040 entries. Excluding string-valued `feature_country` leaves 1,039 numeric model features. A bounded 100,000-row Train sample verified:

| Observation | Result |
|---|---:|
| Integer features | 1,017 × `int8` |
| Floating features | 22 × `float32` |
| Full sample-frame deep memory | 116,252,335 bytes |
| Feature-column memory | 110,500,000 bytes |
| Extracted model-feature copy | 110,500,132 bytes |
| Feature bytes per row | 1,105 |
| Observed dtype conversions | None |

Selected-column Parquet reads preserve the compact dtypes. However, the current real pipeline loads full historical and Research Validation frames, concatenates them into research history, makes `.copy()` fold slices, and `prepare_v6_model_data` copies model features again. These objects can overlap in memory. LightGBM receives the compact pandas dtypes but constructs additional native bins, datasets, histograms, and temporary arrays.

Multiplying the observed 1,105 feature bytes per row by the actual 6,284,208-row Fold 5 training set gives 6,944,049,840 bytes (about 6.47 GiB) for one pandas feature matrix. This is an extrapolation only and excludes index/representation changes, research-history and fold copies, validation matrices, LightGBM internal memory, and the operating system.

The bounded baseline-parameter LightGBM fit took 15.35 seconds. Process RSS was 1,354,743,808 bytes before fitting, peaked at 3,079,294,976 bytes under 20 ms `psutil` sampling, and was 1,368,403,968 bytes afterward. LightGBM reported 787 used features in this early bounded sample; no predictive interpretation is assigned. Small-sample runtime and peak memory are not assumed to scale linearly.

Feasibility classification: **High risk / not recommended** for the current unoptimized All-feature walk-forward on a 32 GB Windows machine. The full experiment should not proceed until full-frame duplication is reduced and the optimized path is re-profiled. The profile read Train only and performed no OOF, NCORR, Legacy CORR, or holdout operation.

### Task 18B.1 memory optimization

The former real fold path retained the original historical and Research Validation frames while constructing a second full concatenated research-history frame. Each fold then copied all selected columns into training and validation frames, and `prepare_v6_model_data` copied the feature matrices again. The optimized path keeps the two protected source frames, selects approved train and validation dates directly into the final model matrices, and never constructs the full research-history or complete fold frames. Feature order, compact dtypes, targets, identities, folds, purge, LightGBM parameters, and scoring are unchanged.

The complete research history has 6,509,738 rows. At the sampled 1,105 feature bytes per row, its feature payload alone projects to about 7.19 GB (6.70 GiB); removing it is a projected full-run benefit, not a measured full-fold peak. Similarly, avoiding the second Fold 5 feature matrix projects to another 6.94 GB reduction. Pandas representation, copy-on-write behavior, LightGBM native allocation, and temporary objects make these projections non-additive.

| Bounded 100,000-row measurement | Before | After |
|---|---:|---:|
| Feature-column memory | 110,500,000 | 110,500,000 |
| Estimated concurrent feature bytes in profiled extraction path | 331,500,264 | 221,000,132 |
| RSS after model-input extraction | 1,354,715,136 | 1,246,912,512 |
| RSS immediately before LightGBM fit | 1,354,743,808 | 1,246,953,472 |
| Sampled peak RSS | 3,079,294,976 | 2,986,553,344 |
| Fit runtime | 15.35 s | 15.02 s |

The measured peak reduction was about 92.7 MB (3.0%); extraction RSS fell by about 107.8 MB (8.0%). Runtime was essentially unchanged and is not treated as a predictive-performance result. Deterministic synthetic regression tests verified identical feature values/dtypes, target values, identities/order, fold assignments, and LightGBM predictions within absolute tolerance `1e-12`.

Feasibility remains **High risk / not recommended** on 32 GB. The bounded result verifies removal of one sample-scale copy but cannot guarantee that multi-million-row LightGBM native allocations fit. A larger staged profile would be required before a full All-feature run; otherwise research should proceed with targeted feature analysis. Neither profile used Validation targets, OOF scoring, or holdout data.

## Feature Quality IC Contract

The initial model-independent feature-quality scope is the official Medium set after excluding `feature_country`: 350 numeric features against explicit `target_jupiter_60`, grouped by observed weekly `date`. No NCORR neutralization, Gaussianization, feature interpretation, model fitting, redundancy clustering, or selection formula is part of this diagnostic.

For one date and feature, the implementation:

1. Drops rows where either the feature or target is missing.
2. Requires at least three valid pairs by default.
3. Rejects constant feature or target values as undefined.
4. Assigns average ranks to tied feature and target values.
5. Calculates Pearson correlation between those two rank vectors.

The resulting Spearman IC retains its sign. Every row records the date, feature, IC, valid-pair count, total row count, coverage, and status. A valid numerical zero remains `0.0`; insufficient, constant, or otherwise undefined IC remains missing and carries a non-valid status.

### Fold-specific as-of restrictions

Feature summaries are constructed separately for every approved fold. The strict aggregation interface requires complete date/feature rows for exactly `fold.train_dates`. It rejects purge dates, same-fold Validation dates, observations that become eligible only in a later fold, the pre-holdout gap, sealed holdout, and immature dates. The real audit runner reads only the union of approved training dates; a Research Validation date can contribute only after it becomes part of a later fold's expanding training history. Approved training targets must be mature and non-null.

Available summaries are mean and median IC, sample IC standard deviation, absolute mean IC, mean-to-standard-deviation ratio when defined, sign consistency relative to the mean sign, valid-date coverage, rolling 52-date mean IC, and annual mean/median IC with coverage. These are descriptive statistics; no final ranking or selection rule is implemented.

### Memory and artifacts

The runner reads one date at a time from Train or date-filtered Validation columns, calculates 350 feature rows, and releases the source frame before the next date. It never loads the full Medium history or constructs a dense feature-to-feature correlation matrix. Gap and holdout targets are never requested.

A completed report uses:

```text
outputs/v6.0/feature_quality/<analysis_id>/
├── COMPLETED.json
├── metadata.json
├── per_date_ic.parquet
├── fold_summaries.parquet
├── rolling_ic.parquet
└── annual_ic.parquet
```

Metadata records the ordered feature list, dataset and target, exact IC and missingness policies, minimum pair count, rolling window, exact fold training cutoffs, configuration, and Git state. Writers refuse to overwrite an existing analysis. Task 19B created and tested this contract.

### Task 19C Medium audit

The single real audit `v6_medium_ic_001` completed in approximately 14 minutes 18 seconds. It covers the union of approved fold training dates, which equals the widest Fold 5 training history: 1,131 weekly dates from 2003-01-03 through 2024-08-30. It contains 395,850 unique feature-date rows for the 350 ordered Medium numeric features. Exactly 378,585 rows (95.6385%) have valid IC; 17,265 rows record `constant_feature`. There are no duplicate identities or unauthorized dates.

For the latest Fold 5 as-of summaries:

- Mean signed IC across features: `0.001129`; median: `0.001001`.
- Mean absolute mean IC: `0.002674`; median: `0.002270`.
- Median sign consistency: `0.565990`.
- Median valid-date coverage: `1.0`; mean coverage: `0.956385`.
- Positive-mean features: 218; negative-mean features: 132.
- Fold 1-to-5 absolute-IC rank Spearman: `0.896282`; median absolute rank change: 24.

The five strongest standalone features by latest-as-of absolute mean IC are `feature_constitution_30` (`0.009874`), `feature_strength_32` (`0.009855`), `feature_charisma_152` (`0.009369`), `feature_constitution_29` (`0.008719`), and `feature_constitution_36` (`0.008582`). These anonymous names are identifiers only and carry no inferred financial interpretation.

Small-versus-Medium-only descriptive results are:

| Latest Fold 5 training-history statistic | Small (100) | Medium-only (250) |
|---|---:|---:|
| Mean absolute mean IC | 0.002426 | 0.002773 |
| Median absolute mean IC | 0.001865 | 0.002403 |
| Median sign consistency | 0.550806 | 0.572060 |
| Mean valid-date coverage | 0.940124 | 0.962889 |
| Mean absolute recent 52-date IC | 0.004693 | 0.004795 |
| Recent weakening percentage | 23.0% | 27.6% |
| Recent sign-reversal percentage | 41.0% | 38.0% |
| Median absolute Fold 1-to-5 rank change | 28 | 23 |

The recent window is 2023-09-08 through 2024-08-30. Across all features, mean absolute recent 52-date IC is `0.004766` versus `0.002694` for the earlier-history mean; 92 features weaken and 136 reverse mean sign. Absolute estimates from a short window are more variable than long-history estimates, so their magnitudes are not directly equivalent evidence. Aggregate Medium-only standalone quality did not deteriorate relative to Small in a way that explains the known recent model-performance reversal. IC is descriptive and unconditional; interactions, redundancy, and LightGBM behavior remain untested.

The sealed holdout, pre-holdout gap, immature dates, same-fold Validation dates, and every fold's purge dates remain excluded from its selection summaries. No feature subset, model result, NCORR, redundancy, importance, permutation, or ablation result was generated.

## Saved Medium Model Importance Audit

Task 19D loaded the five existing `v6_medium_baseline_001` fold boosters and joined them to `v6_medium_ic_001`; it did not fit or score a model. Every booster contains the same ordered 350-feature list as experiment metadata, official Medium metadata, and the IC audit. Gain and split values are normalized independently to sum to one within each fold. Average ranks preserve ties, including a shared bottom rank for zero-importance features.

The completed non-overwriting artifact is:

```text
outputs/v6.0/model_importance/v6_medium_importance_001/
├── COMPLETED.json
├── metadata.json
├── per_fold_importance.parquet
├── cross_fold_summary.parquet
├── group_summary.parquet
├── ic_importance.parquet
├── rank_correlations.parquet
├── ic_relationships.parquet
└── research_summary.json
```

It contains 1,750 unique fold-feature rows, 350 cross-fold feature summaries, ten Small/Medium-only fold summaries, and 1,750 exact fold-matched IC joins. Normalized gain and split shares sum to one in every fold. No feature is unused across all five models; one Medium-only feature has zero gain/split in Fold 3 and one in Fold 5. Using a predeclared top-35 threshold, 21 features rank in the top group in at least four folds and 20 rank there in exactly one fold.

The highest mean normalized gain shares are:

| Feature | Group | Mean gain share |
|---|---|---:|
| `feature_endurance_7` | Small | 0.014921 |
| `feature_endurance_5` | Medium-only | 0.009863 |
| `feature_endurance_11` | Medium-only | 0.008583 |
| `feature_strength_32` | Medium-only | 0.007340 |
| `feature_wisdom_60` | Medium-only | 0.006799 |
| `feature_wisdom_76` | Medium-only | 0.006759 |
| `feature_wisdom_71` | Medium-only | 0.006661 |
| `feature_constitution_172` | Small | 0.006505 |
| `feature_constitution_175` | Medium-only | 0.006065 |
| `feature_intelligence_20` | Medium-only | 0.005711 |

Mean pairwise fold gain-rank Spearman is `0.794849`, the minimum pair is `0.753403`, and Fold 1 versus Fold 5 is `0.769943`. Median absolute Fold 1-to-5 rank movement is 38. Importance is therefore persistent but less stable than standalone absolute-IC ranks.

| Group statistic across folds | Small | Medium-only |
|---|---:|---:|
| Total gain-share range | 0.321091–0.341620 | 0.658380–0.678909 |
| Mean gain share per feature | 0.003337 | 0.002665 |
| Mean top-ten within-group concentration | 0.194901 | 0.107751 |

The group totals reflect unequal group sizes: Small is 28.6% of Medium features but receives roughly one-third of gain, whereas Medium-only contains 71.4% of features and receives roughly two-thirds. Small has higher per-feature average gain and greater within-group concentration. These observations do not attribute the Small/Medium OOF difference to either group.

Each fold's importance is joined only to IC calculated through that model's training cutoff. Absolute mean IC versus normalized gain Spearman ranges from `0.230970` to `0.270855`; sign consistency versus gain ranges from `0.241401` to `0.261147`. Examples repeatedly classified as high-IC/low-gain include `feature_charisma_110`, `feature_charisma_104`, and `feature_endurance_19`. Repeated low-IC/high-gain examples include `feature_intelligence_20`, `feature_constitution_337`, `feature_constitution_259`, and `feature_constitution_77`. This weak positive relationship is expected to differ because gain reflects conditional tree use, interactions, and substitute features, while IC measures standalone monotonic association.

Gain and split importance are model-dependent usage diagnostics. They are biased by tree construction and correlated alternatives and do not establish causality, standalone alpha, or incremental contribution. No validation target, holdout target, NCORR, permutation, redundancy, ablation, or selection rule was used.

## Fold-Specific Top-250 IC Selection Contract

Task 19E adds the single policy `medium_top250_abs_mean_ic_v1`. Its universe is the ordered 350 numeric features in the official Medium set. Independently for each approved research fold, it ranks features by descending absolute training-history mean Spearman IC from `v6_medium_ic_001`, breaks ties by feature name, places missing or undefined statistics last, and requires at least 250 features with valid observations. It selects exactly 250 and restores official Medium order before model input.

The selector verifies the dataset (`v6.0`), target (`target_jupiter_60`), complete Medium feature identity, approved fold identity, exact training-date list, and training cutoff. A fold can consume only its own as-of summary; later-fold rows, Validation-derived rankings, gap/holdout dates, mixed folds, incomplete feature sets, and inconsistent metadata are rejected. Gain and split importance are not selection inputs.

Each selected experiment will persist `feature_manifests/fold_<n>.json`. A manifest records:

- policy ID, fixed `K=250`, fold number, approved train start/end, and source IC analysis ID;
- source-universe identity and hashes of the ranked and model-ordered feature lists;
- selection rank, feature name, signed mean IC, absolute mean IC, and valid-date count;
- the 250 features in official model-input order;
- model configuration, random seed, repository configuration, and Git state.

`K=250` was predeclared as one bounded 100-feature reduction from Medium. It was not optimized against OOF results, and the implementation does not compare other values or policies. The Task 19F command was:

```powershell
.\.venv\Scripts\python.exe -B -m src.pipeline --version v6.0 --feature-set medium --model-config baseline --selection-policy medium_top250_abs_mean_ic_v1 --feature-quality-id v6_medium_ic_001 --experiment-id v6_medium_top250_ic_002 --run --confirm-full-run
```

Task 19F executed this command exactly once. `v6_medium_top250_ic_002` contains five fold models, five manifests, 2,180,295 OOF rows, and all 330 approved dates. Canonically sorted OOF identities match the Medium baseline exactly; physical Parquet row order differs. There are no duplicate identities or dates from the pre-holdout gap, sealed holdout, or immature partition. The observed wall time was approximately 14 minutes 30 seconds.

| Fold | Train cutoff | Small | Medium-only | Top-250 NCORR | Medium NCORR | Difference |
|---:|---|---:|---:|---:|---:|---:|
| 1 | 2018-09-07 | 65 | 185 | 0.020128 | 0.025914 | -0.005785 |
| 2 | 2020-03-06 | 65 | 185 | 0.031604 | 0.030107 | +0.001497 |
| 3 | 2021-09-03 | 66 | 184 | 0.028253 | 0.027457 | +0.000796 |
| 4 | 2023-03-03 | 67 | 183 | 0.025829 | 0.022474 | +0.003356 |
| 5 | 2024-08-30 | 67 | 183 | 0.024129 | 0.019773 | +0.004356 |

Overall mean NCORR is `0.02632684` versus `0.02612141`; NCORR standard deviation is `0.01771411` versus `0.01798110`; Sharpe-like is `1.4862` versus `1.4527`; positive-date percentage is `91.82%` versus `93.33%`. Mean Legacy CORR is `0.02257569` versus `0.02241025`. The paired NCORR difference has mean `0.00020542`, standard deviation `0.01168414`, median `0.00010869`, and 50.61% positive dates.

Top-250 annual NCORR for 2019–2025 is `0.022518`, `0.013870`, `0.040988`, `0.026561`, `0.035479`, `0.019460`, and `0.022667`. The corresponding differences are `-0.007060`, `-0.001164`, `+0.001681`, `-0.000352`, `+0.000684`, `+0.006221`, and `+0.003862`; 2025 contains 17 dates. The ten largest absolute paired moves contribute 10.05% of total absolute movement. This indicates broad temporal variation rather than a result driven solely by a few dates, but is not a dependence-aware significance test.

There are 212 features selected in all folds, 17 in only one fold, and 286 selected at least once. Pairwise Jaccard similarity ranges from `0.8182` to `0.8939`, reflecting gradual changes across expanding training cutoffs. This one predeclared comparison does not tune K or establish that standalone IC selection generalizes. Univariate IC can miss nonlinear, interaction, and conditional contribution, and repeated research validation creates selection bias. The sealed holdout remains outside selection and evaluation.

## Fixed Ridge Baseline Contract

Task 20A integrates one untuned Ridge baseline into the existing v6 pipeline. Its input is the ordered 350 numeric Medium features, target `target_jupiter_60`, and the unchanged five expanding folds with the 16-date purge. It uses no target-aware feature selection. The objective is normalized mean squared error plus `alpha * ||coefficients||²`, with fixed `alpha=0.001`; under an unnormalized sum-of-squares convention the equivalent penalty is `n_samples * 0.001`. The choice was declared before real Ridge training and is not based on OOF performance.

Preprocessing is fitted separately within each fold. Non-missing training means provide imputation, and training population standard deviations provide scaling. Validation receives those parameters unchanged. Constant or near-constant training columns use scale 1 after centering; an entirely missing training column uses mean 0 and scale 1 and therefore receives a zero fitted coefficient. Only numeric finite-or-missing features are accepted, and exact feature order is enforced.

The deterministic solver streams two passes in 20,000-row float32 batches. The first calculates training means and scales; the second accumulates float64 `XᵀX` and `Xᵀy`, adds `n_samples * alpha` to the coefficient diagonal, and solves the 350-dimensional system. It does not construct a full standardized matrix. The approximate peak Ridge-specific batch buffers are 28 MB float32 plus 56 MB float64, in addition to the compact fold DataFrames and ordinary scoring data. A bounded 1,024-row, 350-feature synthetic fit passed; this verifies batch behavior but not full-fold runtime or guaranteed machine feasibility.

Ridge artifacts use `numerai_streaming_ridge_v1` JSON with the feature identity, preprocessing parameters, coefficients, intercept, objective parameters, training sample count, runtime, constant/all-missing counts, and maximum batch size. Existing LightGBM `.txt` artifacts retain their prior format. The Task 20B command was:

```powershell
.\.venv\Scripts\python.exe -B -m src.pipeline --version v6.0 --feature-set medium --model-config ridge_baseline --experiment-id v6_medium_ridge_003 --run --confirm-full-run
```

Task 20B executed this command once as `v6_medium_ridge_003`. It completed in approximately 6 minutes 48 seconds. All five models contain the exact Medium order, alpha `0.001`, finite preprocessing and coefficients, expected fold sample counts, and 20,000-row maximum batches. Reloaded models reproduce all 2,180,295 saved predictions exactly. Canonical OOF identities match `v6_medium_baseline_001`; all 330 approved dates and fold assignments are present, with no duplicates, protected dates, or non-finite predictions.

| Fold | Training rows | Training seconds | Coefficient norm | Ridge NCORR | Difference vs LightGBM |
|---:|---:|---:|---:|---:|---:|
| 1 | 4,227,405 | 27.09 | 0.016052 | 0.012947 | -0.012967 |
| 2 | 4,715,780 | 30.96 | 0.015363 | 0.019822 | -0.010285 |
| 3 | 5,231,777 | 34.02 | 0.014383 | 0.017574 | -0.009883 |
| 4 | 5,767,397 | 37.10 | 0.013703 | 0.024235 | +0.001761 |
| 5 | 6,284,208 | 40.54 | 0.013355 | 0.008538 | -0.011235 |

All folds have 350 nonzero coefficients, zero constant/near-constant features, and zero entirely missing features. Model JSON sizes are 38,424; 38,344; 38,320; 38,262; and 38,345 bytes. Nonzero coefficients reflect a jointly regularized linear fit and are not LightGBM-style Gain importance. A reliable process peak-memory sample was unavailable.

Overall Ridge NCORR is `0.01809302`, standard deviation `0.01721479`, mean/std ratio `1.0510`, and positive-date percentage `85.76%`; Medium LightGBM records `0.02612141`, `0.01798110`, `1.4527`, and `93.33%`. Ridge Legacy CORR is `0.01578823` versus `0.02241025`. These are raw weekly research summaries; the ratio is not an annualized investment Sharpe.

Ridge annual NCORR for 2019–2025 is `0.013576`, `0.004749`, `0.030541`, `0.022085`, `0.015917`, `0.024855`, and `0.007681`. Differences from LightGBM are `-0.016002`, `-0.010285`, `-0.008767`, `-0.004828`, `-0.018878`, `+0.011616`, and `-0.011124`; 2025 contains 17 dates. The paired weekly difference has mean `-0.00802839`, standard deviation `0.01744712`, median `-0.00913352`, and is positive on 34.24% of dates. Its ten largest absolute moves contribute 9.64% of total absolute movement.

Ridge underperforms overall and is unstable across periods despite outperforming in Fold 4 and 2024. This does not establish statistical significance, causality, or useful ensemble complementarity. Its additive form may underfit nonlinear interactions; alpha remains untuned, and the sealed holdout remains outside evaluation.

## Saved-Prediction Diversity Audit

Task 20C reads the completed OOF artifacts for Medium LightGBM (A: `v6_medium_baseline_001`), fold-specific Top-250 LightGBM (B: `v6_medium_top250_ic_002`), and Medium Ridge (C: `v6_medium_ridge_003`). All have 330 dates and 2,180,295 exact canonical identities with identical targets, folds, metric identities, and NCORR reference metadata. The analyzer selects identity and prediction columns only; metadata records `target_columns_accessed: []`.

For each pair and date, Pearson correlation uses raw predictions and Spearman uses average within-date ranks. Mean absolute percentile-rank disagreement is scale-free. Top and bottom deciles contain `ceil(0.10 * date_rows)` observations using stable prediction order, and overlap is reported relative to that decile count. Missing/non-finite and constant-prediction diagnostics are explicit; the real inputs contain none.

| Pair | Pearson mean / median / std / min / max | Spearman mean / median / std / min / max | Rank disagreement | Top / bottom overlap |
|---|---|---|---:|---:|
| A vs B | 0.7283 / 0.7309 / 0.0129 / 0.6985 / 0.7534 | 0.7275 / 0.7297 / 0.0119 / 0.6916 / 0.7522 | 0.1532 | 53.43% / 52.53% |
| A vs C | 0.5238 / 0.5227 / 0.0159 / 0.4835 / 0.5636 | 0.5271 / 0.5261 / 0.0156 / 0.4904 / 0.5672 | 0.2174 | 33.34% / 33.83% |
| B vs C | 0.5145 / 0.5165 / 0.0175 / 0.4686 / 0.5630 | 0.5181 / 0.5205 / 0.0161 / 0.4794 / 0.5513 | 0.2198 | 32.87% / 32.95% |

No pair has a date above 0.9 Pearson or Spearman. A–B has no date below 0.5. A–C has 4.85%/3.33% of dates below 0.5 Pearson/Spearman; B–C has 20.61%/16.06%. A–B mean Spearman by fold is `0.7138`, `0.7252`, `0.7346`, `0.7352`, and `0.7328`. A–C is `0.5271`, `0.5396`, `0.5325`, `0.5146`, and `0.5042`; B–C is `0.5227`, `0.5236`, `0.5277`, `0.5042`, and `0.4930`.

Calendar results show the same recent divergence. A–B mean Spearman is `0.7356` in 2024 and `0.7324` in partial 2025. A–C declines to `0.5114` and `0.5042`; B–C declines to `0.4977` and `0.4929`. B–C top-decile overlap is 32.07% in 2024 and 31.46% in partial 2025, with bottom overlap 30.76% and 30.55%. The lowest A–C dates are mostly in early 2025; the lowest B–C dates are mostly in late 2024 and early 2025. However, moderate tree–Ridge disagreement exists throughout the history rather than on only a few dates.

Artifacts are saved under `outputs/v6.0/prediction_diversity/v6_three_model_diversity_001/` with metadata, completion marker, per-date correlations, fold and annual summaries, rank disagreement, decile overlap, and compact JSON summary. The analysis does not use targets, recalculate NCORR, train a model, or form an ensemble. Correlation alone does not demonstrate incremental alpha, useful complementarity, or an appropriate ensemble weight. The final holdout remains sealed.

## Fixed-Weight Rank-Ensemble Contract

Task 21A predeclares exactly two policies. `medium_top250_rank_equal_v1` weights `v6_medium_baseline_001` and `v6_medium_top250_ic_002` at 0.50 each. `medium_top250_ridge_rank_442_v1` weights those experiments and `v6_medium_ridge_003` at 0.40, 0.40, and 0.20. These values are fixed infrastructure inputs, not optimized weights. The first tests whether two related tree models benefit from equal rank averaging; the second gives a bounded 20% allocation to the less-correlated but weaker Ridge model without making a complementarity claim.

Normalization occurs independently for every component and Research Validation date. Average ranks handle ties and the normalized value is `(rank - 0.5) / n`, where `n` is that date's row count. A constant component date maps every row to 0.5 and is recorded diagnostically. Construction selects only `date`, `numerai_ticker`, `composite_figi`, `fold_id`, and `prediction`; it rejects missing/non-finite predictions, duplicate or mismatched identities, mismatched metadata/scoring references, incorrect folds, and any date outside 2019-01-04 through 2025-04-25. Targets enter only the separately invoked approved scoring workflow.

Future artifacts live under `outputs/v6.0/ensembles/<ensemble_id>/` and contain canonical ensemble predictions, metadata, component IDs and weights, normalization identity, source metadata/OOF hashes, Git state, and a completion marker. Scored runs additionally contain per-date and aggregate metrics. They explicitly contain no trained fold models and refuse overwrite. The existing comparison loader supports scored ensemble artifacts; unscored ensembles are rejected from performance comparison.

Real execution requires both `--run` and `--confirm-real-scoring` and calls the existing official NCORR implementation, date-bounded neutralizers, and separately named Legacy CORR without changing formulas. It does not implement NMMC or NBMC. Task 21A ran only synthetic integration tests, so no ensemble performance result exists. Both policies are exposed to repeated Research Validation selection bias because their design followed prior OOF and diversity observations. The sealed holdout remains untouched.

## Fixed-Rank Ensemble Evaluation

Task 21B evaluated exactly the two predeclared policies. Ensemble A (`v6_medium_top250_rank_equal_001`) has mean NCORR `0.02812597`, std `0.01820897`, mean/std `1.5446`, 94.24% positive dates, mean Legacy CORR `0.02353895`, Legacy mean/std `1.4065`, worst weekly NCORR `-0.01409069`, and P10 `0.00488538`. Ensemble B (`v6_medium_top250_ridge_rank_442_002`) is `0.02795070`, `0.01801763`, `1.5513`, 93.03%, `0.02322010`, `1.4285`, `-0.01581489`, and `0.00517175` respectively.

```powershell
.\.venv\Scripts\python.exe -B -m src.ensemble --version v6.0 --policy medium_top250_rank_equal_v1 --ensemble-id v6_medium_top250_rank_equal_001 --run --confirm-real-scoring
.\.venv\Scripts\python.exe -B -m src.ensemble --version v6.0 --policy medium_top250_ridge_rank_442_v1 --ensemble-id v6_medium_top250_ridge_rank_442_002 --run --confirm-real-scoring
.\.venv\Scripts\python.exe -B -m src.analyze_ensembles --version v6.0
```

Ensemble A fold NCORR is `0.024286`, `0.033355`, `0.029916`, `0.025992`, and `0.023600`; annual NCORR for 2019–2025 is `0.027564`, `0.015281`, `0.043431`, `0.028616`, `0.037776`, `0.017699`, and `0.022296`. Ensemble B fold values are `0.022892`, `0.033225`, `0.029386`, `0.027763`, and `0.021608`; annual values are `0.025942`, `0.013976`, `0.044016`, `0.029258`, `0.035330`, `0.021374`, and `0.020306`. The weakest year for both is 2020, and the weakest fold is Fold 5 for both B and A, although A's Fold 1 is only slightly higher than Fold 5.

Paired A-minus-Medium mean/median/std is `+0.002005`/`+0.001872`/`0.005762`, with 63.94% wins, P10/P90 `-0.004799`/`+0.009568`, largest improvement/decline `+0.015586`/`-0.019197`, and top-ten absolute-difference share 9.38%. A-minus-Top-250 is `+0.001799`/`+0.001441`/`0.006464`, 58.79%, `-0.006172`/`+0.009800`, `+0.022657`/`-0.014418`, and 10.53%. B-minus-A is `-0.000175`/`-0.000088`/`0.003762`, 49.09%, `-0.004929`/`+0.004824`, `+0.010165`/`-0.011076`, and 8.93%.

The A-minus-Medium fold differences are `-0.001628`, `+0.003248`, `+0.002459`, `+0.003518`, `+0.003828`; A-minus-Top-250 is `+0.004158`, `+0.001751`, `+0.001663`, `+0.000162`, `-0.000528`. B-minus-A is `-0.001393`, `-0.000130`, `-0.000529`, `+0.001771`, `-0.001993`. Ridge's allocation helps 2024 (`+0.003675`) and modestly 2021–2022, but hurts 2019, 2020, 2023, and partial 2025, providing no full-period mean improvement over A.

Both artifacts have exact source hashes, 2,180,295 finite canonical predictions over 330 dates, official neutralizer hash `1ae870b1…c6c6f`, and no model files or protected dates. Runs took about 4m04s and 3m43s and occupy 23.18 MB and 24.14 MB; peak memory was not measured reliably. No statistical-significance claim or final policy selection is made. Repeated use of the same Research Validation history creates selection bias, and the sealed holdout remains untouched.

## Frozen V1 Policy and Final Holdout Protocol

The machine-readable freeze is `docs/manifests/v6.0_v1_policy.json`, policy `numerai_classic_v6_v1`. It freezes two fresh LightGBM components: official Medium 350 and a 250-feature subset recomputed at the final training cutoff using `medium_top250_abs_mean_ic_v1`. Selection ranks descending absolute mean per-date training-history Spearman IC, places undefined statistics last, breaks ties by feature name, requires 250 valid features, and restores official Medium ordering for model input. It does not reuse any historical fold list.

Both components use `n_estimators=300`, `learning_rate=0.05`, `num_leaves=31`, `random_state=42`, `n_jobs=-1`, LightGBM native NaN handling, and no learned imputation. Per-date average-rank normalization is `(rank - 0.5) / n`; component weights are 0.50/0.50. Primary scoring is official NCORR v1 through `numerai-tools==0.7.2` and neutralizer SHA-256 `1ae870b1…c6c6f`; Legacy CORR is a separately named diagnostic. Ridge is excluded.

The final boundaries were verified without selecting a target column: eligible training ends 2025-04-25; the gap has 16 observed dates and 107,987 rows from 2025-05-02 through 2025-08-15; the sealed holdout has 52 mature dates and 365,363 rows from 2025-08-22 through 2026-08-14; the 13 immature dates from 2026-08-21 through 2026-11-13 remain excluded. The gap preserves the approved Jupiter-60 16-date maturity separation.

Future execution is deliberately two phase. First, a target-safe preparation phase must load only eligible labeled history through 2025-04-25, recompute IC through that cutoff, select Top-250, train both fresh models, generate all component and ensemble holdout predictions, validate identities, and persist immutable models/manifests/predictions before holdout targets are authorized. Second, a separate explicit scoring phase may open the 52 holdout targets and calculate official NCORR and Legacy CORR. Models, features, weights, normalization, and parameters remain fixed throughout all dates and cannot respond to holdout results.

Task 22A therefore needs: a final-training loader distinct from ordinary Research OOF loaders; extension of feature-quality coverage beyond the existing 2024-08-30 Fold-5 cutoff to 2025-04-25; one as-of final-selection manifest; two-model full-history training; a pre-target prediction artifact contract; canonical holdout alignment; and a separate scoring authorization boundary. Existing ordinary loaders correctly exclude holdout targets and must retain that default. No holdout-driven feature selection, tuning, weight search, or model replacement is permitted.

Final reporting was predeclared as mean/std/mean-to-std NCORR, positive-date percentage, worst weekly NCORR, P10 NCORR, separately labeled Legacy CORR, both component diagnostics, exact date/row integrity, and a period-qualified comparison with Research OOF. There is no arbitrary pass threshold. Task 22C completed this evaluation once; the holdout is evaluated and no longer untouched. Historical NMMC and NBMC references remain unavailable.

## Sealed Final-Holdout Infrastructure

Task 22A implements, but does not run, dedicated final-training and holdout primitives. Final training combines only mature labeled history ending 2025-04-25 and rejects the gap or any later row. Holdout preparation reads only identity plus frozen feature columns; any target-prefixed column is rejected. Existing Research OOF loaders and neutralizer access remain unchanged and continue rejecting protected dates.

The final IC interface accepts reusable per-date observations only when dataset, target, exact ordered Medium list, and IC definition match. It merges additional authorized observations, rejects duplicates or dates after 2025-04-25, requires the complete date-feature Cartesian coverage, and then applies the unchanged absolute-mean-IC Top-250 ranking, deterministic feature-name tie break, and official-order restoration. No real missing coverage was computed in Task 22A.

Preparation artifacts live under `final_holdout/<evaluation_id>/preparation/`: copied frozen policy, selection manifest, two models, Medium/Top-250/ensemble predictions, metadata, SHA-256 for every artifact, Git identity, target-access record `[]`, and `PREPARATION_COMPLETED.json`. Scoring is a distinct `scoring/` state and cannot begin without explicit target authorization, complete preparation, policy and artifact hash verification, exact 52-date/365,363-row canonical coverage, finite predictions, and no previous final marker. It stores per-date metrics, aggregate reporting, component diagnostics, official reference identity, and `FINAL_EVALUATION_COMPLETED.json`.

The dedicated holdout neutralizer function accepts only the frozen 52 dates and performs the same schema, identity, order, and finiteness checks as Research OOF. NCORR and Legacy CORR formulas are unchanged. The reporting contract adds worst-week and P10 NCORR to the existing mean/std/ratio and positive-date summaries and forbids automatic model selection.

```powershell
.\.venv\Scripts\python.exe -B -m src.final_holdout prepare --version v6.0 --evaluation-id v6_v1_final_001 --run --confirm-final-preparation
.\.venv\Scripts\python.exe -B -m src.final_holdout score --version v6.0 --evaluation-id v6_v1_final_001 --run --authorize-holdout-target-access --confirm-final-scoring
```

Task 22C enables real scoring only with both explicit authorization flags and successful verification of the committed preparation seal. Local safeguards are auditable accident-prevention controls, not a security boundary against deliberate direct filesystem modification.

## Frozen V1 Final Preparation

Task 22B executed the preparation command exactly once with evaluation ID `v6_v1_final_001`. Target-free date validation found 1,165 eligible weekly training dates and 6,509,738 rows from 2003-01-03 through 2025-04-25. No gap, holdout, or immature row entered training.

The compatible `v6_medium_ic_001` source contains 395,850 rows for 350 features over 1,131 dates through 2024-08-30. The runner calculated 11,900 rows for the 34 missing dates from 2024-09-06 through 2025-04-25 in 34.07 seconds. The combined artifact has 407,750 unique date-feature rows over 1,165 dates. Provenance records source hashes, the exact IC definition, missingness policy, feature order, and Validation file identity. The final manifest selects exactly 250 unique features in official Medium order. It shares 240 features with Fold 5 (Jaccard `0.9231`), with 10 additions and 10 removals.

The Medium-350 and Top-250 models used the frozen 300-tree, 0.05 learning-rate, 31-leaf, seed-42 configuration and native missing handling. They trained once on the identical 6,509,738 rows in 180.70 and 131.06 seconds. Model sizes are 992,015 and 984,389 bytes. Both reload checks have maximum absolute prediction difference `0.0`. Peak process memory was not measured reliably; the runner released each trained model before fitting the next component. Runtime before artifact persistence was 362.38 seconds.

All three target-free prediction artifacts contain the same 365,363 canonical identities, 52 dates from 2025-08-22 through 2026-08-14, no duplicates, and only finite predictions. Component files retain raw predictions; the ensemble separately applies per-date average ranks and fixed 0.50/0.50 weights. Metadata records `target_columns_accessed: []`.

| Artifact | SHA-256 |
| --- | --- |
| Frozen policy | `4ccddd457707232feb7391389b4a387e57dcd1871af898aeefd82a1f8fa7381e` |
| Final feature manifest | `6af69e4f0862434c7d5bbcb318ad70a83bc65037f0f88b6af9fb3cee38cb4a38` |
| Medium model | `1d3994ce989b4e3607f8e81d922abd70053f595842161ceba7de140bbdd10d6f` |
| Top-250 model | `370912bc83235e1883626f47ae861517aa0956e6e303e060fe4ab22c7a9d1479` |
| Medium predictions | `b92626933618b3bec31de2a73af18dc413c0510342965a951bc974eca93d5035` |
| Top-250 predictions | `255ef598733f4983e7552a3c8a907eff3b1f2bd606905e018c48228c2aa16613` |
| Ensemble predictions | `7f0b0d194b7eafbb55d7f5183509c5c938fad97c39caf3b9ce09ecc2a5c63669` |
| IC extension | `f66492f6abff6773a09e55c80f826bc5a1460fad2ad48e2698d669254eea76b0` |
| Combined per-date IC | `0ee40d632e3c3a27cfb188d1e3e9d0d30f9b0564df86cc4fae01ca0f7a3de952` |
| Final IC summary | `a5d61db276de65f87e5da2357ba8e60e0d9ca255ab1fb4709a75a4802ebabd54` |
| IC provenance | `d5e7c0a13f4bdc1e720beb99b3c6098671a0ad36f094eb8b2ae56dcbacb6a9c9` |

`PREPARATION_COMPLETED.json` exists and all recorded hashes continue to verify. Task 22C created a separate scoring directory and final marker without modifying the preparation artifacts.

The preparation is sealed in version control at `docs/manifests/v6.0_v1_final_001_preparation_seal.json`, SHA-256 `dbb1d968687125abd915be88e9b864c0c005e324e14f053f92d470c854b9a297`. The seal includes all 11 full preparation artifact hashes, metadata hash, completion-marker hash, recorded Git identity, empty target-access record, and explicit `prepared` / `unscored` / `sealed` status. The immutable completion marker did not record a timestamp, so the seal records that timestamp as unavailable rather than inferring one from filesystem metadata.

## Independent Final-Holdout Evaluation

Task 22C executed exactly one authorized evaluation. Before target access it verified policy SHA-256 `4ccddd457707232feb7391389b4a387e57dcd1871af898aeefd82a1f8fa7381e`, preparation-seal SHA-256 `dbb1d968687125abd915be88e9b864c0c005e324e14f053f92d470c854b9a297`, all 11 preparation hashes, 52 dates, 365,363 identities per prediction, exact component alignment, and the frozen ensemble. It persisted `SCORING_STARTED.json` and `AUTHORIZATION.json` before the first target read. `TARGET_ACCESS.json` records a single projected access to `date`, `numerai_ticker`, `composite_figi`, and `target_jupiter_60` over 2025-08-22–2026-08-14; no other target column was accessed and raw target values were not saved.

Official NCORR uses `numerai-tools==0.7.2` and the exact 300-column neutralizer with SHA-256 `1ae870b17d501b8d9ec114ab2096eece342812c424a826f44a772087a82c6c6f`. Legacy CORR remains separately labeled. NMMC and NBMC were not calculated.

| Candidate | Dates / rows | Mean NCORR | Std | Mean/std | Positive | Median | Worst | Best | P10 | Mean Legacy CORR | Legacy std | Legacy ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Medium | 52 / 365,363 | 0.015918 | 0.016749 | 0.9504 | 82.69% | 0.017320 | -0.026575 | 0.042976 | -0.005573 | 0.013501 | 0.014833 | 0.9103 |
| Top-250 | 52 / 365,363 | 0.014903 | 0.020868 | 0.7142 | 73.08% | 0.016389 | -0.035146 | 0.050011 | -0.012579 | 0.012291 | 0.018602 | 0.6607 |
| Frozen ensemble | 52 / 365,363 | 0.016431 | 0.019656 | 0.8359 | 80.77% | 0.014883 | -0.032016 | 0.050741 | -0.008534 | 0.013294 | 0.017073 | 0.7787 |

Ensemble-minus-Medium weekly NCORR has mean/median `+0.000513`/`+0.000212`, std `0.005773`, wins 51.92%, P10/P90 `-0.006106`/`+0.007351`, and extremes `-0.013487`/`+0.013952`. Ensemble-minus-Top-250 has mean/median `+0.001528`/`+0.001383`, std `0.004427`, wins 57.69%, P10/P90 `-0.003431`/`+0.007471`, and extremes `-0.009236`/`+0.013330`. These are descriptive diagnostics, not evidence for changing the frozen policy.

The ensemble's 19 partial-2025 dates have mean NCORR `-0.002327`, std `0.013680`, and 52.63% positive dates. Its 33 partial-2026 dates have mean `0.027232`, std `0.013464`, and 96.97% positive dates. Best and worst weeks are 2026-05-22 (`0.050741`) and 2025-11-14 (`-0.032016`). Against 330-date Research OOF, final mean NCORR is lower by `0.011695` (41.58% relative), mean/std is lower by `0.7087`, positive dates by 13.47 percentage points, and mean Legacy CORR by `0.010245`. The periods and sample sizes differ, so this is an honest descriptive comparison rather than an exchangeable-sample inference.

Scoring took 103.22 seconds before artifact persistence. RSS rose from approximately 272 MiB to 454 MiB; peak RSS was not measured. Reports and their hashes are under `outputs/v6.0/final_holdout/v6_v1_final_001/scoring/`. The target-free reporting command is:

```powershell
.\.venv\Scripts\python.exe -B -m src.final_holdout report --version v6.0 --evaluation-id v6_v1_final_001
```

The final holdout has been evaluated and is no longer untouched. Any subsequent research informed by these results is a new development phase and cannot be presented as independent final-holdout validation.

The immutable evaluation summary is versioned at `docs/manifests/v6.0_v1_final_001_evaluation_record.json`. It records all verified identities and hashes, exact saved aggregate and temporal metrics, paired diagnostics, and the Research OOF comparison without raw targets, predictions, models, credentials, or datasets. The ensemble has the highest holdout mean NCORR among the three predeclared candidates, while Medium has the better holdout mean/std ratio and downside statistics. The evidence is temporally heterogeneous, its cause remains unresolved, and no statistical-significance, causal, or profitability conclusion is supported. The frozen policy remains unchanged and is retained for initial live deployment without holdout-driven tuning; this deployment state is distinct from future research.

## Remaining Validation-Design Questions

The following research decisions remain open:

- Appropriate categorical encoding for `feature_country`
- Full-dataset integer-feature ranges, cardinalities, and missingness
- Benchmark, example-prediction, and neutralizer file schemas
- Model-selection metrics and experiment budget

The approved split protocol establishes data boundaries only; it does not establish any v6 model result.
