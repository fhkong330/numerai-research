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