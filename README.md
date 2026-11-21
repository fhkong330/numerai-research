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
