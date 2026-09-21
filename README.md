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
│   ├── train.parquet
│   ├── validation.parquet
│   └── features.json
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
│   ├── experiments/
│   └── model_experiments/
│
├── tests/
│
├── DATA_GUIDE.md
├── RESEARCH_LOG.md
├── README.md
└── requirements.txt
```

## Data

The project currently uses Numerai dataset version **v5.3**.

Main local data files:

- `train.parquet` — historical training data
- `validation.parquet` — validation and test data
- `features.json` — Numerai feature-set definitions

Large datasets and generated outputs are stored locally and are not tracked by Git.

See [`DATA_GUIDE.md`](DATA_GUIDE.md) for detailed information about the dataset, feature sets, targets, eras, and validation splits.

## Modeling

The current baseline model is **LightGBM**.

The modeling pipeline separates reusable functionality into dedicated modules:

- `data.py` — data loading and research-history construction
- `models.py` — model creation, training, and prediction
- `evaluation.py` — Numerai CORR and performance summaries
- `validation.py` — walk-forward fold generation and data splitting
- `pipeline.py` — experiment orchestration and result persistence
- `config.py` — shared research and model configuration
- `paths.py` — centralized project paths

This keeps the notebooks focused on research and analysis while reusable logic remains in `src/`.

## Validation

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

Project paths are centralized in `src/paths.py`.

Experiment outputs can include:

- Era-level CORR results
- Fold-level summaries
- Overall performance metrics
- Model and experiment metadata

This allows experiments to be compared without relying only on notebook state.

## Documentation

- [`README.md`](README.md) — project overview and workflow
- [`DATA_GUIDE.md`](DATA_GUIDE.md) — dataset structure, features, targets, and validation design
- [`RESEARCH_LOG.md`](RESEARCH_LOG.md) — experiment results, conclusions, and research decisions

## Current Direction

The current focus is improving cross-sectional predictive performance and robustness through:

- Feature-set research
- Feature importance and stability analysis
- Model experiments
- Walk-forward robustness testing
- Ensemble research

The final goal is to select a robust model using historical research data, evaluate it once on the untouched test set, and then move to Numerai live submissions.