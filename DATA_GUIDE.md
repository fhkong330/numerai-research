# Numerai Data Guide

This document provides a quick reference for the Numerai v5.3 dataset used in this project.

The goal is to understand the dataset structure without needing to rerun the exploratory notebook.

---

## 1. Dataset Overview

**Dataset version:** v5.3

Main local files:

| File | Purpose |
|---|---|
| `train.parquet` | Historical training data |
| `validation.parquet` | Validation and final test data |
| `features.json` | Feature-set metadata |

The dataset is organized by **era**.

Each era represents a weekly cross-sectional snapshot containing many anonymous securities.

---

## 2. Training Data

File:

`data/train.parquet`

### Basic Information

- Rows: **2,746,268**
- Columns: **3,599**
- Features: **3,555**
- Named target columns: **40**
- Number of eras: **574**
- Era range: **0001–0574**

Important non-feature columns include:

- `id`
- `era`
- `data_type`
- `target`

### Era Size

Number of observations per era:

| Statistic | Rows |
|---|---:|
| Mean | ~4,784 |
| Median | ~4,923 |
| Minimum | ~2,314 |
| Maximum | ~6,009 |

### Simplified Structure

The full dataset contains thousands of feature columns, but conceptually it looks like:

| era | feature_1 | feature_2 | feature_3 | ... | target |
|---|---:|---:|---:|---|---:|
| 0001 | ... | ... | ... | ... | 0.50 |
| 0001 | ... | ... | ... | ... | 0.75 |
| 0001 | ... | ... | ... | ... | 0.25 |
| 0002 | ... | ... | ... | ... | 1.00 |
| 0002 | ... | ... | ... | ... | 0.50 |

Each row represents one anonymous security within one era.

---

## 3. Validation Data

File:

`data/validation.parquet`

### Full File

- Rows: **4,128,060**
- Columns: **3,599**
- Number of eras: **660**
- Era range: **0575–1234**

The file contains both validation and final test observations.

### Validation Portion

`data_type == "validation"`

- Rows: **4,036,034**
- Number of eras: **647**
- Era range: **0575–1221**

This portion is used for model development and out-of-sample evaluation.

### Final Test Portion

`data_type == "test"`

- Rows: **92,026**
- Number of eras: **13**
- Era range: **1222–1234**

This is reserved as the final holdout and should not be used during normal model development.

---

## 4. Current Train / Validation / Test Split

The current project predicts the **Ender-60** target.

Because this target has a relatively long forward horizon, a **16-era purge** is applied between training and validation.

Current split:

```text
Train               Purge          Validation                    Test
0001 ─────── 0558   0559 ─ 0574   0575 ─────────────── 1221   1222 ─ 1234
```

Therefore:

| Dataset | Era Range | Number of Eras |
|---|---|---:|
| Training | 0001–0558 | 558 |
| Purge | 0559–0574 | 16 |
| Validation | 0575–1221 | 647 |
| Final Test | 1222–1234 | 13 |

The purge prevents overlap between the forward target windows near the training/validation boundary.

### Expanding Walk-Forward Validation

In addition to the fixed validation split, the research pipeline supports expanding walk-forward validation.

The historical research data consists of:

- Training eras: 0001–0574
- Historical validation eras: 0575–1221

These datasets are combined into a chronological research history covering eras 0001–1221.

The final test eras 1222–1234 are excluded from the research history and remain untouched.

Walk-forward configuration:

- Purge window: 16 eras
- Validation window: 156 eras
- Training window: expanding
- Final fold: retained even if shorter than 156 eras

Current folds:

| Fold | Training | Purge | Validation |
|---|---|---|---|
| 1 | 0001–0558 | 0559–0574 | 0575–0730 |
| 2 | 0001–0714 | 0715–0730 | 0731–0886 |
| 3 | 0001–0870 | 0871–0886 | 0887–1042 |
| 4 | 0001–1026 | 1027–1042 | 1043–1198 |
| 5 | 0001–1182 | 1183–1198 | 1199–1221 |

For later folds, validation eras from earlier periods become part of the historical training data. This allows the expanding-window procedure to simulate periodic model retraining using all information that would have been historically available.

Overall walk-forward metrics are calculated by combining all out-of-sample era correlations before computing summary statistics. Fold means are not equally averaged because the final fold contains only 23 eras.

---

## 5. features.json

`features.json` defines predefined groups of Numerai features.

Its structure can be simplified as:

```python
{
    "feature_sets": {
        "small": [...],
        "medium": [...],
        "all": [...],

        "v2_equivalent_features": [...],
        "v3_equivalent_features": [...],
        "fncv3_features": [...],

        "intelligence": [...],
        "charisma": [...],
        "strength": [...],
        "dexterity": [...],
        "constitution": [...],
        "wisdom": [...],
        "agility": [...],
        "serenity": [...],
        "sunshine": [...],
        "rain": [...],
        "midnight": [...],
        "faith": [...],
        "quantum": [...]
    }
}
```

For example:

```python
feature_sets["small"]
```

returns a list containing the names of the 42 Small features.

Similarly:

```python
feature_sets["medium"]
```

returns the 780 Medium feature names.

### Feature-Set Sizes

| Feature Set | Number of Features |
|---|---:|
| small | 42 |
| medium | 780 |
| all | 3,555 |
| v2_equivalent_features | 304 |
| v3_equivalent_features | 1,000 |
| fncv3_features | 400 |
| intelligence | 35 |
| charisma | 290 |
| strength | 135 |
| dexterity | 51 |
| constitution | 335 |
| wisdom | 140 |
| agility | 145 |
| serenity | 95 |
| sunshine | 325 |
| rain | 666 |
| midnight | 244 |
| faith | 372 |
| quantum | 807 |

These feature sets can overlap.

Therefore, their feature counts should **not** be added together to obtain the total number of features.

### Current Feature Set

The current benchmark uses:

```text
Feature set: Medium
Number of features: 780
```

---

## 6. Features

Numerai features are anonymized.

Example feature names:

```text
feature_antistrophic_striate_conscriptionist
feature_bicameral_showery_wallaba
feature_bridal_fingered_pensioner
...
```

The names do not reveal the underlying economic meaning of the features.

In the observed v5.3 dataset, feature values are discrete and typically take values such as:

```text
0
1
2
3
4
```

Therefore, feature quality must primarily be studied statistically rather than interpreted from the feature names.

Future feature research may include:

- Feature-target IC
- IC stability across eras
- Positive / negative IC consistency
- Feature importance
- Feature correlation and redundancy
- Regime stability
- Feature-set membership
- Model dependence

---

## 7. Targets

The dataset contains multiple target columns.

Named targets generally follow patterns such as:

```text
target_[codename]_20
target_[codename]_60
```

The `20` and `60` represent different prediction horizons.

In Numerai v5.3, the generic:

```text
target
```

corresponds to:

```text
target_ender_60
```

This is the current target used by the project.

### Target Values

Typical target values are:

```text
0.00
0.25
0.50
0.75
1.00
```

Approximate distribution within an era:

| Target | Approximate Share |
|---:|---:|
| 0.00 | ~5% |
| 0.25 | ~20% |
| 0.50 | ~50% |
| 0.75 | ~20% |
| 1.00 | ~5% |

The target is a processed cross-sectional future outcome.

It is **not raw stock return**.

For example:

```text
target = 1.0
```

does **not** mean that the security returned 100%.

Instead, it represents a relatively strong future outcome according to Numerai's processed target construction.

---

## 8. Era

`era` is the main time variable in the dataset.

Example:

```text
0001
0002
0003
...
0574
0575
...
1234
```

Each era represents a weekly cross-sectional observation period.

The ordering is chronological:

```text
smaller era number
        ↓
earlier time

larger era number
        ↓
later time
```

This allows us to perform time-aware validation such as:

- Chronological train / validation splits
- Purging
- Expanding walk-forward validation
- Rolling walk-forward validation

---

## 9. IDs

The dataset contains anonymized security IDs.

An ID identifies an observation within the Numerai dataset, but it should not be treated as a persistent real-world ticker.

Therefore, we cannot use the dataset like:

```text
AAPL through time
MSFT through time
NVDA through time
```

Instead, the main research structure is:

```text
Era 0001
    many anonymous securities

Era 0002
    many anonymous securities

Era 0003
    many anonymous securities
...
```

The project therefore focuses on **cross-sectional prediction within each era**.

---

## 10. Evaluation Structure

The model generates one prediction for every row in the validation set.

Predictions are then evaluated separately within each era:

```text
Era 0575
Predictions vs Targets
        ↓
      CORR

Era 0576
Predictions vs Targets
        ↓
      CORR

Era 0577
Predictions vs Targets
        ↓
      CORR

...
```

This produces a time series of **647 validation-era CORRs**.

The project currently summarizes these using:

- Mean CORR
- Std CORR
- Median CORR
- Minimum CORR
- Maximum CORR
- CORR Sharpe-like ratio
- Positive Era Rate

CORR measures predictive ranking quality.

**CORR is not portfolio return.**

Therefore:

```text
CORR = 0.02
```

does not mean:

```text
Portfolio return = 2%
```

The CORR Sharpe-like ratio:

```text
Mean CORR / Std CORR
```

is a measure of prediction consistency, not a true portfolio Sharpe Ratio.

---

## 11. Current Benchmark Data Configuration

Current working benchmark:

| Setting | Value |
|---|---|
| Dataset | Numerai v5.3 |
| Feature Set | Medium |
| Number of Features | 780 |
| Target | Ender-60 |
| Training Eras | 0001–0558 |
| Purge Eras | 0559–0574 |
| Validation Eras | 0575–1221 |
| Final Test Eras | 1222–1234 |

The final test set remains untouched during normal research.

---

## 12. Future Feature Research

As feature research progresses, this document can be expanded with observed feature characteristics.

Because Numerai features are anonymized, this section should describe **observed statistical behavior**, not unsupported economic interpretations.

Possible future feature records:

```text
Feature: feature_xxx

Feature Sets:
- medium
- all

Mean IC:
...

IC Std:
...

IC Sign:
Positive / Negative

Era Stability:
...

Feature Importance:
...

Redundancy:
...

Regime Behavior:
...

Research Notes:
...
```

This section will be expanded when systematic feature analysis begins.