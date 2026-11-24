# Numerai Research

A research project for developing and evaluating machine learning models on the Numerai dataset.

The project focuses on building a reproducible research workflow for cross-sectional prediction, including data exploration, feature-set experiments, model experiments, and time-aware validation.

## Research Workflow

The current research process is:

**Data Exploration → Baseline → Fixed Validation → Walk-Forward Validation → Final Test → Live**

Two complementary validation workflows are supported:

- **Fixed Validation** — used for fast screening of new features, models, and research ideas.
- **Expanding Walk-Forward Validation** — used for more robust evaluation across different historical periods.

The final test set is kept untouched during model development and is reserved for evaluating selected candidates before live deployment.

## Project Structure

```text
numerai-research/
├── data/
│   └── v5.3/
│       ├── train.parquet
│       ├── validation.parquet
│       └── features.json
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_model.ipynb
│   ├── 03_feature_set_experiment.ipynb
│   └── 04_model_experiment.ipynb
│
├── src/
│   ├── config.py
│   ├── data.py
│   ├── download_data.py
│   ├── evaluation.py
│   ├── models.py
│   ├── paths.py
│   ├── pipeline.py
│   └── validation.py
│
├── outputs/
│   └── v5.3/
│       ├── experiments/
│       └── model_experiments/
│
├── tests/
│
├── docs/
│   ├── data/
│   │   ├── DATA_GUIDE_v5.3.md
│   │   └── DATA_GUIDE_v6.0.md
│   └── manifests/
│       └── v5.3_preservation.json
├── RESEARCH_LOG.md
├── README.md
└── requirements.txt
```

## Data

The historical benchmark uses Numerai dataset version **v5.3**. The explicitly selectable v6.0 research workflow is configured for split and fold verification, but no v6 model result exists yet.

Main local data files:

- `data/v5.3/train.parquet` — historical training data
- `data/v5.3/validation.parquet` — validation and test data
- `data/v5.3/features.json` — Numerai feature-set definitions

Large datasets and generated outputs are stored locally and are not tracked by Git.

Dataset selection is centralized in `src/config.py`. Active v5.3 paths now resolve to `data/v5.3/` and `outputs/v5.3/`. The verified original files remain temporarily in the flat `data/` and `outputs/` locations; other versions, including v6.0, resolve only to their own versioned directories and never fall back to those legacy files.

Download files explicitly from the repository root by selecting both a version and one or more files:

```powershell
python -m src.download_data --version v6.0 --file v6.0/features.json
```

Repeat `--file` to request multiple files. The downloader accepts only files in the verified catalog for the selected version, stores them under `data/<version>/`, and skips existing destinations without overwriting them. It never downloads every available file implicitly.

Audit the local v6.0 schema and temporal structure without loading every feature and target at once:

```powershell
.\.venv\Scripts\python.exe -B -m src.audit_data --version v6.0
```

The audit saves a machine-readable report under `outputs/v6.0/data_audit/` and refuses to overwrite an existing report.

Verify the approved v6 research partitions and five walk-forward folds using only local date and split columns:

```powershell
.\.venv\Scripts\python.exe -B -m src.validation --version v6.0
```

The v6 configuration selects `target_jupiter_60`, weekly `date` values, a 16-observed-date purge, and 78-date validation windows. Selection is explicit through `--version v6.0`; the historical v5.3 default remains unchanged for compatibility.

See the [`v5.3 Data Guide`](docs/data/DATA_GUIDE_v5.3.md) for detailed information about the dataset, feature sets, targets, eras, and validation splits.

## Modeling

The current baseline model is **LightGBM**.

The reusable v6 model interface now provides:

- `resolve_v6_numeric_feature_set(...)` for schema-checked official feature sets
- `prepare_v6_model_data(...)` to separate numeric features, the explicit Jupiter-60 target, and identity columns
- `train_validated_lightgbm(...)` for validated fitting with an explicit ordered feature list
- `predict_validated_lightgbm(...)` for order-checked predictions returned with the input row index

The initial v6 baseline policy resolves the official `medium` set, excludes the string-valued `feature_country`, and uses the remaining 350 numeric features in official metadata order. Numeric NaNs are left for LightGBM to handle natively; no global imputation, normalization, target-based selection, or early stopping is applied.

Run the model-interface tests with:

```powershell
.\.venv\Scripts\python.exe -B -m unittest tests.test_models -v
```

Only tiny synthetic models are trained by these tests. Full v6 walk-forward training has not been performed; the evaluation and artifact contracts described below are implemented but contain no real v6 results yet.

### v6 OOF Evaluation and Artifacts

The reusable evaluation layer validates research-only OOF rows with this contract:

```text
date, numerai_ticker, composite_figi, fold_id,
prediction, target_jupiter_60
```

It rejects duplicate identities, missing or nonfinite predictions, null targets, incorrect fold assignments, incomplete research-date coverage, and any gap, holdout, or immature date. Overall summaries are computed over per-date scores, so the 18-date final fold is not given the same weight as a 78-date fold.

Official Classic v6 NCORR (`neutral_corr` version 1) is implemented with `numerai-tools==0.7.2` and all 300 verified v6 neutralizer columns. It is calculated independently for each weekly date and remains distinct from `NUMERAI_CORR_JUPITER_60`, the preserved non-neutral Legacy CORR research diagnostic. The exact historical Meta Model required for NMMC and the exact shared benchmark series required for NBMC are not available through a verified public source, so neither contribution metric is implemented or approximated.

Acquire and verify the official Validation neutralizer reproducibly with:

```powershell
.\.venv\Scripts\python.exe -B -m src.download_data --version v6.0 --file validation_neutralizer.parquet
.\.venv\Scripts\python.exe -B -m src.audit_scoring_references --version v6.0
```

The audit reads only Parquet metadata and identity/date columns from Validation; it does not read targets. NCORR scoring loads only one requested date and its 300 neutralizer columns at a time, checks exact `date`/`numerai_ticker`/`composite_figi` row alignment, then calls the official scoring function directly.

Install the pinned dependencies and run the official-scoring tests with:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -B -m unittest tests.test_ncorr tests.test_evaluation -v
```

No real v6 OOF prediction or NCORR result exists yet; only synthetic numerical and artifact-contract verification has been run.

### v6 Walk-Forward Pipeline

The integrated v6 pipeline resolves the official 350-feature numeric Medium set, loads only those features plus protected identity and Jupiter-60 fields, generates the approved five folds, and fits a fresh baseline LightGBM model per fold. Validation labels are used only after prediction for scoring; they are not passed to model fitting or early stopping. Fold predictions retain their original identity and are validated before official NCORR and separately named Legacy CORR are calculated.

Run the complete test suite:

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

Run the bounded synthetic smoke pipeline:

```powershell
.\.venv\Scripts\python.exe -B -m src.pipeline --version v6.0 --feature-set medium --model-config baseline --experiment-id v6_smoke_001 --smoke
```

Smoke mode constructs synthetic approved-date data and cannot invoke the real-data loader. Its artifacts are marked `run_kind: synthetic_smoke` and must not be interpreted as research performance.

The future real baseline command is shown for reproducibility but has **not** been executed:

```powershell
.\.venv\Scripts\python.exe -B -m src.pipeline --version v6.0 --feature-set medium --model-config baseline --experiment-id v6_medium_baseline_001 --run --confirm-full-run
```

Both `--run` and `--confirm-full-run` are required. Outputs are non-overwriting, and only directories containing `COMPLETED.json` can be loaded as completed experiments.

The first real v6 baseline, `v6_medium_baseline_001`, completed on 2026-11-21 with 2,180,295 OOF rows across all 330 Research Validation dates. Headline results were:

| Metric | Mean | Std. dev. | Sharpe-like | Positive dates |
|---|---:|---:|---:|---:|
| NCORR (`neutral_corr` v1) | 0.02612 | 0.01798 | 1.4527 | 93.33% |
| Legacy CORR (`NUMERAI_CORR_JUPITER_60`) | 0.02241 | 0.01676 | 1.3374 | 91.21% |

Reproduce it with the real baseline command above. Local ignored artifacts are available at [`outputs/v6.0/experiments/v6_medium_baseline_001/`](outputs/v6.0/experiments/v6_medium_baseline_001/), including [`metadata.json`](outputs/v6.0/experiments/v6_medium_baseline_001/metadata.json) and [`metrics.json`](outputs/v6.0/experiments/v6_medium_baseline_001/metrics.json). These outputs are intentionally not committed.

Future completed experiments can be saved without overwriting under:

```text
outputs/v6.0/experiments/<experiment_id>/
├── metadata.json
├── metrics.json
├── oof_predictions.parquet
└── per_date_metrics.parquet
```

Run the synthetic evaluation and artifact tests with:

```powershell
.\.venv\Scripts\python.exe -B -m unittest tests.test_evaluation -v
```

The Medium run above established the first real v6 benchmark. No feature selection, tuning, or holdout evaluation has been performed.

### Small vs Medium controlled comparison

Experiment `v6_small_baseline_002` changed only the official feature set, using 100 numeric Small features that are an ordered subset of Medium's 350. Model parameters, folds, target, purge, OOF dates, neutralizer, and scoring were unchanged.

| Feature set | Mean NCORR | NCORR Sharpe-like | Positive dates | Mean Legacy CORR |
|---|---:|---:|---:|---:|
| Medium | 0.026121 | 1.4527 | 93.33% | 0.022410 |
| Small | 0.015617 | 0.9768 | 80.91% | 0.012855 |

Small minus Medium mean paired NCORR was `-0.010504` with paired-date standard deviation `0.017189`; Small won 24.24% of dates. Small underperformed overall and in Folds 1–4, but exceeded Medium in 2024, partial 2025, and the short fifth fold. This is a temporal comparison, not a feature-importance conclusion or a dependence-aware significance test.

Reproduce the saved-artifact comparison without retraining:

```powershell
.\.venv\Scripts\python.exe -B -m src.compare_experiments --version v6.0 --baseline v6_medium_baseline_001 --candidate v6_small_baseline_002
```

Both experiment directories remain local and Git-ignored. No holdout evaluation has been performed.

### All-feature memory preflight

The official `all` set resolves to 1,039 numeric features after excluding `feature_country`: 1,017 `int8` and 22 `float32` columns. Run the bounded Train-only engineering profile with:

```powershell
.\.venv\Scripts\python.exe -B -m src.profile_feature_memory --version v6.0 --feature-set all --max-sample-rows 100000 --fit-lightgbm --output outputs/v6.0/profiling/all_100000_rows_baseline.json
```

The 100,000-row feature matrix used 110.5 MB (decimal) with no dtype conversion. A single Fold 5 feature matrix extrapolates to 6.94 GB before research-history copies, fold/model copies, validation data, LightGBM native structures, temporary arrays, and OS overhead. The bounded baseline fit took 15.35 seconds and peaked at 3.08 GB process RSS. Runtime must not be extrapolated linearly from this sample.

Classification: **High risk / not recommended** for the current unoptimized five-fold pipeline on a 32 GB Windows machine. The full All-feature experiment has not been run; loading and copy behavior should be optimized and re-profiled first. This profile does not calculate predictive-performance metrics or access Validation/holdout targets.

The Task 18B.1 pipeline now constructs fold model inputs directly from the historical and Research Validation source frames. It no longer creates a complete concatenated research-history frame, complete copied fold frames, and a second feature copy. Fold models are still persisted one at a time and released before the next fold; OOF accumulation retains only identity, prediction, target, and fold columns. NCORR neutralizers remain loaded one date at a time.

Reproduce the post-optimization bounded profile with:

```powershell
.\.venv\Scripts\python.exe -B -m src.profile_feature_memory --version v6.0 --feature-set all --max-sample-rows 100000 --fit-lightgbm --output outputs/v6.0/profiling/all_100000_rows_baseline_optimized.json
```

The optimized profile preserved the 110.5 MB feature matrix and compact dtypes. RSS after model-input extraction fell from 1,354,715,136 to 1,246,912,512 bytes; sampled LightGBM peak RSS fell from 3,079,294,976 to 2,986,553,344 bytes; runtime changed from 15.35 to 15.02 seconds. The bounded copy model removes one 110,500,132-byte feature copy per 100,000 rows. These measurements do not establish full-fold peak memory. Classification remains **High risk / not recommended** until a larger staged profile demonstrates sufficient 32 GB headroom; targeted feature research is safer than launching the full All-feature run now.

### Feature Quality infrastructure

`src.feature_quality` provides model-independent, per-date cross-sectional Spearman IC diagnostics for the official 350 numeric Medium features. It uses pairwise-complete average ranks, preserves signed IC, records valid-pair counts and coverage, and stores undefined IC as missing with an explicit status rather than zero. Fold summaries are rebuilt independently from each approved fold's exact training dates; purge, same-fold validation, later research, gap, holdout, and immature dates are rejected by the as-of interface.

Run the permanent synthetic verification with:

```powershell
.\.venv\Scripts\python.exe -B -m unittest tests.test_feature_quality -v
```

The future real Medium audit command is:

```powershell
.\.venv\Scripts\python.exe -B -m src.feature_quality --version v6.0 --feature-set medium --analysis-id v6_medium_ic_001 --run --confirm-feature-audit
```

The audit streams one approved training date at a time and writes a non-overwriting report beneath `outputs/v6.0/feature_quality/<analysis_id>/`. Task 19C executed the command once as `v6_medium_ic_001`: 1,131 approved Fold 5 training-history dates (2003-01-03–2024-08-30), 350 features, and 395,850 feature-date rows. Of those rows, 378,585 (95.64%) have valid IC and 17,265 explicitly record `constant_feature` with missing IC.

Analyze the completed compact artifacts without recomputing IC:

```powershell
.\.venv\Scripts\python.exe -B -m src.analyze_feature_quality --version v6.0 --analysis-id v6_medium_ic_001
```

The widest approved as-of summary has median absolute mean IC `0.002270`, median sign consistency `0.5660`, and 218 positive-mean versus 132 negative-mean features. Fold 1-to-5 absolute-IC ranks have Spearman correlation `0.8963`, although the median feature moved 24 rank positions. Medium-only features show higher latest-as-of median absolute mean IC (`0.002403`) than Small (`0.001865`); this standalone association does not establish why the Small model outperformed Medium in 2024 and partial 2025. No feature subset was selected and no predictive-performance result was produced. Redundancy, model importance, permutation importance, and ablation remain out of scope.

### Saved-model feature importance

Task 19D reads the five existing `v6_medium_baseline_001` LightGBM boosters without retraining and joins each fold's gain/split importance only to the matching as-of IC summary from `v6_medium_ic_001`:

```powershell
.\.venv\Scripts\python.exe -B -m src.analyze_model_importance --version v6.0 --experiment-id v6_medium_baseline_001 --feature-quality-id v6_medium_ic_001 --analysis-id v6_medium_importance_001
```

All 350 features were used in at least one fold. The strongest mean normalized-gain features were `feature_endurance_7` (1.4921%), `feature_endurance_5` (0.9863%), `feature_endurance_11` (0.8583%), `feature_strength_32` (0.7340%), and `feature_wisdom_60` (0.6799%). Pairwise fold gain-rank Spearman averaged `0.7948`; Fold 1 versus Fold 5 was `0.7699`, with median absolute movement of 38 ranks.

Small captured 32.11%–34.16% of total fold gain with 100 features, while the 250 Medium-only features captured 65.84%–67.89%. Average gain per feature was higher for Small (`0.003337`) than Medium-only (`0.002665`), and Small importance was more concentrated in its top ten features. Fold-specific absolute IC versus gain correlations were positive but weak (`0.2310`–`0.2709`), confirming that standalone association and conditional model usage are distinct diagnostics. These results are not causal, do not measure incremental value, and define no selection policy.

### Fold-specific Top-250 IC policy

Task 19E implements one predeclared candidate policy, `medium_top250_abs_mean_ic_v1`, without running it on a real model. For each approved fold it ranks the official 350 numeric Medium features by descending absolute mean Spearman IC calculated only over that fold's exact training dates. It requires valid IC observations, sends missing or undefined values to the end, resolves ties by feature name, selects exactly 250, and then restores official Medium order for the model matrix. It never uses gain importance, same-fold Validation targets, later-fold summaries, the pre-holdout gap, or holdout targets.

`K=250` is fixed in advance as one bounded reduction of 100 features from Medium, not a value selected by comparing candidate results. Each future experiment stores a per-fold manifest with the policy and source-analysis identities, approved cutoff, selection ranks and IC statistics, ordered model features, integrity hashes, configuration, and code state. Existing Small and Medium experiments continue to use their original global feature lists.

The real candidate command was:

```powershell
.\.venv\Scripts\python.exe -B -m src.pipeline --version v6.0 --feature-set medium --model-config baseline --selection-policy medium_top250_abs_mean_ic_v1 --feature-quality-id v6_medium_ic_001 --experiment-id v6_medium_top250_ic_002 --run --confirm-full-run
```

Task 19F executed this command exactly once as `v6_medium_top250_ic_002`, completing in approximately 14 minutes 30 seconds. Its 2,180,295 OOF rows and 330 dates have the same canonical identities as `v6_medium_baseline_001`, with no duplicate, gap, holdout, or immature identity.

| OOF metric | Medium 350 | Fold Top-250 |
|---|---:|---:|
| Mean NCORR | 0.026121 | 0.026327 |
| NCORR standard deviation | 0.017981 | 0.017714 |
| NCORR Sharpe-like | 1.4527 | 1.4862 |
| Positive NCORR dates | 93.33% | 91.82% |
| Mean Legacy CORR | 0.022410 | 0.022576 |

The paired Top-250-minus-Medium mean is `0.00020542` with standard deviation `0.01168414`; Top-250 wins 50.61% of dates. Fold 1 declines by `-0.00578544`, while Folds 2–5 improve by `0.00149703`, `0.00079637`, `0.00335589`, and `0.00435610`. The strongest annual differences are later: 2024 is `+0.00622100` and partial 2025 is `+0.00386168`; 2019 is `-0.00705978`. The top ten absolute paired moves account for 10.05% of total absolute movement, so the result is not explained by only a few dates, but its small mean and temporal heterogeneity do not establish statistical significance.

Across manifests, 212 features appear in all five folds, 17 appear in only one, and 286 appear at least once. Pairwise Jaccard similarity ranges `0.8182`–`0.8939`. Small membership rises from 65 features in Folds 1–2 to 66 in Fold 3 and 67 in Folds 4–5; the remainder are Medium-only. IC is a standalone monotonic-association diagnostic, so the policy may remove conditionally useful or interacting features. The sealed holdout remains untouched.

### Fixed Ridge baseline infrastructure

Task 20A adds a Ridge option to the same v6 fold, scoring, and artifact pipeline. The predeclared model uses all 350 official numeric Medium features and minimizes:

```text
mean((y - intercept - X_standardized @ coefficients) ** 2)
    + 0.001 * sum(coefficients ** 2)
```

The normalized `alpha=0.001` convention is independent of fold sample count and is equivalent to an unnormalized scikit-learn-style penalty of `n_samples * 0.001`. It was chosen as modest shrinkage for standardized features before observing any Ridge OOF result. The closed-form sufficient-statistics solver is deterministic, so no stochastic random state is consumed.

For every fold, training-only non-missing means impute missing numeric values and training-only population standard deviations scale features. Constant, near-constant (scale at most `1e-6`), and entirely missing training columns receive scale 1; their centered training values are zero. Categorical columns, infinities, feature-order changes, null targets, and target-index misalignment are rejected.

The implementation makes two 20,000-row passes, retaining float32 feature batches and only a float64 350×350 Gram matrix, cross-product vector, and solve. A batch requires roughly 28 MB as float32 and 56 MB for its temporary float64 sufficient-statistics view, rather than an approximately 8.8 GB full standardized Fold 5 matrix. Each fold is released before the next. A bounded 1,024×350 synthetic fit verifies width and batch limits, but does not establish full-fold runtime or guaranteed 32 GB headroom.

Each fold model is saved as `models/fold_<n>.json` using `numerai_streaming_ridge_v1`, including ordered features, imputation means, scales, coefficients, intercept, alpha, batch settings, sample count, training runtime, constant/all-missing counts, and observed maximum batch size. Reloaded models reproduce predictions exactly.

The real experiment command was:

```powershell
.\.venv\Scripts\python.exe -B -m src.pipeline --version v6.0 --feature-set medium --model-config ridge_baseline --experiment-id v6_medium_ridge_003 --run --confirm-full-run
```

Task 20B executed this command exactly once. `v6_medium_ridge_003` completed in approximately 6 minutes 48 seconds and produced five reloadable fold models, 330 OOF dates, and 2,180,295 rows with canonical identities matching LightGBM Medium. No duplicate, gap, holdout, immature, or non-finite prediction exists.

| OOF metric | Ridge Medium | LightGBM Medium |
|---|---:|---:|
| Mean NCORR | 0.018093 | 0.026121 |
| NCORR standard deviation | 0.017215 | 0.017981 |
| NCORR mean/std ratio | 1.0510 | 1.4527 |
| Positive NCORR dates | 85.76% | 93.33% |
| Mean Legacy CORR | 0.015788 | 0.022410 |

The paired Ridge-minus-LightGBM NCORR mean is `-0.00802839` with standard deviation `0.01744712`; Ridge wins 34.24% of dates. It underperforms in four folds and every year except 2024. Fold 4 is `+0.00176109`, and 2024 is `+0.01161561`; partial 2025 falls by `-0.01112400`. The top ten absolute weekly differences account for 9.64% of total absolute movement. These raw weekly mean/std ratios are research stability summaries, not annualized investment Sharpe ratios, and no dependence-aware significance test or ensemble experiment was performed.

Fold training times are 27.1, 31.0, 34.0, 37.1, and 40.5 seconds for 4.23–6.28 million rows. Every fold uses a 20,000-row maximum batch, has 350 finite nonzero coefficients, no constant/all-missing feature, and a 37–38 KiB JSON model. Coefficient norms decline from `0.016052` to `0.013355`. Nonzero linear coefficients are not comparable to LightGBM Gain importance. Reliable process peak memory was unavailable; only the bounded buffer design is documented. The sealed holdout remains untouched.

The modeling pipeline separates reusable functionality into dedicated modules:

- `data.py` — data loading and research-history construction
- `models.py` — model creation, training, and prediction
- `evaluation.py` — Numerai CORR and performance summaries
- `validation.py` — walk-forward fold generation and data splitting
- `pipeline.py` — experiment orchestration and result persistence
- `config.py` — shared research and model configuration
- `paths.py` — centralized, version-aware project paths

This keeps the notebooks focused on research and analysis while reusable logic remains in `src/`.

## Validation

### v6.0 Research Validation

The approved v6.0 V1 protocol uses:

- Explicit target: `target_jupiter_60`
- Research validation: 2019-01-04–2025-04-25
- Expanding training windows
- 16 observed weekly dates purged before every validation window
- Four 78-date validation folds plus one final 18-date partial fold
- Final holdout gap: 2025-05-02–2025-08-15
- Sealed final holdout: 2025-08-22–2026-08-14
- Immature dates excluded: 2026-08-21–2026-11-13

Ordinary research history excludes the gap, holdout, and immature dates. Approved date coverage is fixed to the current snapshot and verification fails rather than silently shifting boundaries.

The sections below describe the preserved historical v5.3 workflow.

### Fixed Validation

Fixed validation is used for fast experiment screening.

Current historical split:

- Training: eras `0001–0558`
- Purge: eras `0559–0574`
- Validation: eras `0575–1221`
- Final test: eras `1222–1234`

A **16-era purge window** is used for the current 60-day target.

### Expanding Walk-Forward Validation

Promising ideas can be evaluated using expanding walk-forward validation.

The current configuration uses:

- Expanding training history
- 16-era purge window
- 156-era validation windows
- Independent model retraining for each fold
- Combined out-of-sample evaluation across validation eras

Historical validation eras are incorporated into the training history once they become part of the available past.

The final test eras `1222–1234` remain excluded from research and walk-forward model selection.

## Experiments

Research currently includes:

1. Data exploration and dataset structure analysis
2. Small-feature LightGBM baseline
3. Small vs. Medium feature-set comparison
4. LightGBM parameter experiment
5. Expanding walk-forward validation infrastructure

Experiment results and research decisions are documented in [`RESEARCH_LOG.md`](RESEARCH_LOG.md).

Generated machine-readable experiment artifacts are stored under `outputs/`.

## Experiment Pipeline

The project provides two main experiment interfaces:

```python
run_experiment(...)
```

Used for fast fixed-validation experiments.

```python
run_walk_forward_experiment(...)
```

Used for expanding walk-forward evaluation of stronger candidates.

The intended research loop is:

```text
Idea
  ↓
Hypothesis
  ↓
Fixed Validation
  ↓
Promising?
  ├── No  → Discard / Revise
  └── Yes
       ↓
Walk-Forward Validation
       ↓
Candidate Selection
       ↓
Untouched Final Test
       ↓
Live
```

## Reproducibility

Shared settings such as model parameters, purge length, target configuration, and walk-forward window length are centralized in `src/config.py`.

Project paths are centralized in `src/paths.py`, with isolated resolution for each supported dataset version.

Experiment outputs can include:

- Era-level CORR results
- Fold-level summaries
- Overall performance metrics
- Model and experiment metadata

This allows experiments to be compared without relying only on notebook state.

## Documentation

- [`README.md`](README.md) — project overview and workflow
- [`DATA_GUIDE_v5.3.md`](docs/data/DATA_GUIDE_v5.3.md) — v5.3 dataset structure, features, targets, and validation design
- [`DATA_GUIDE_v6.0.md`](docs/data/DATA_GUIDE_v6.0.md) — verified v6.0 metadata, schema, temporal structure, splits, target availability, and selected feature missingness
- [`RESEARCH_LOG.md`](RESEARCH_LOG.md) — experiment results, conclusions, and research decisions

Each dataset version has a separate Data Guide so its historical schema and research assumptions remain preserved.

## Historical v5.3 Benchmark

The historical v5.3 walk-forward benchmark uses the Numerai Medium feature set with 780 features and the baseline LightGBM configuration.

Across 647 out-of-sample validation eras, the expanding walk-forward evaluation achieved:

- Mean CORR: 0.02648
- Sharpe-like ratio: 1.676
- Positive era rate: 94.13%

Compared with the 42-feature Small benchmark, the Medium feature set improved overall walk-forward Mean CORR from 0.01848 to 0.02648.

The next research stage focuses on feature quality, stability, redundancy, and model-feature interactions.

## Current Direction

The current focus is improving cross-sectional predictive performance and robustness through:

- Feature-set research
- Feature importance and stability analysis
- Model experiments
- Walk-forward robustness testing
- Ensemble research

The final goal is to select a robust model using historical research data, evaluate it once on the untouched test set, and then move to Numerai live submissions.
