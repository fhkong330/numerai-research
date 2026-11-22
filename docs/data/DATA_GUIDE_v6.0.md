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

No real v6 OOF predictions, neutral scores, or performance metrics have been generated.

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
- No v6 model, feature selection, or holdout evaluation has been run.
- The reusable LightGBM, OOF evaluation, and artifact interfaces have been tested only on synthetic data; full v6 walk-forward fitting and populated experiment artifacts remain future work.

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

## Remaining Validation-Design Questions

The following research decisions remain open:

- Appropriate categorical encoding for `feature_country`
- Full-dataset integer-feature ranges, cardinalities, and missingness
- Benchmark, example-prediction, and neutralizer file schemas
- Model-selection metrics and experiment budget

The approved split protocol establishes data boundaries only; it does not establish any v6 model result.
