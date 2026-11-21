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

## Remaining Validation-Design Questions

The following research decisions remain open:

- Appropriate categorical encoding for `feature_country`
- Full-dataset integer-feature ranges, cardinalities, and missingness
- Benchmark, example-prediction, and neutralizer file schemas
- Model-selection metrics and experiment budget

The approved split protocol establishes data boundaries only; it does not establish any v6 model result.
