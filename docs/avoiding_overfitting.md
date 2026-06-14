# Avoiding Overfitting in Genetic Algorithm Backtests

## The Problem

Genetic Algorithms are powerful optimizers — which is exactly what makes them dangerous in quantitative finance. A GA with enough generations and a large enough population **will** find a strategy that looks excellent in-sample, even if that strategy is pure noise.

This is called the **Multiple Testing Problem** or **Backtest Overfitting**: when you evaluate 50,000+ candidate strategies on the same historical data, you are almost guaranteed to find strategies that appear to have high Sharpe Ratios by chance.

## Defense Strategy: Three Layers

This framework uses three complementary techniques to combat overfitting.

### Layer 1: Walk-Forward Validation

The 9-year dataset is split into multiple **folds**. The GA evolves on the training partition; the best chromosome is tested on the unseen out-of-sample partition.

```
Full Dataset (2015–2024)
│
├── Fold 1: Train [2015–2017] → Test [2017–2018]
├── Fold 2: Train [2015–2019] → Test [2019–2020]
├── Fold 3: Train [2015–2021] → Test [2021–2022]
├── Fold 4: Train [2015–2022] → Test [2022–2023]
└── Fold 5: Train [2015–2023] → Test [2023–2024]
```

**Anchored (expanding window)** rather than rolling window — the training set always starts at 2015 to maximize data efficiency.

A strategy is considered **robust** if it performs well (Sharpe > 1.0) on the average of all out-of-sample folds, not just the training data.

### Layer 2: Deflated Sharpe Ratio

When many strategies are tested, the expected maximum Sharpe Ratio rises even for white noise. The **Deflated Sharpe Ratio (DSR)** adjusts for this:

```
DSR = (SR - E[max(SR_i)]) / σ(SR)
```

where `E[max(SR_i)]` is estimated from the number of trials (50,000+ in our case) and the Sharpe distribution's skewness.

A DSR > 0 means the strategy's Sharpe is better than what random selection would produce.

### Layer 3: Combinatorial Purged Cross-Validation (CPCV)

Introduced by López de Prado (2018). Key ideas:

1. **Purging**: Remove training observations whose labels (returns) overlap with test observations in time. This prevents look-ahead leakage from overlapping holding periods.

2. **Embargoing**: Add a gap between train and test sets equal to the maximum holding period (60 days) to prevent carry-over effects.

3. **Combinatorial sampling**: Instead of a single train/test split, all possible combinations of folds are used to compute performance variance.

## Practical Recommendations

| Technique | This Framework | Alternative |
|---|---|---|
| Data splitting | Walk-forward (anchored) | Rolling window |
| Multiple testing correction | DSR | Bonferroni |
| Leakage prevention | Embargo (60-day gap) | CPCV |
| Regime robustness | Test on 5 market regimes | Random subsets |
| Parameter stability | Plot sensitivity heatmaps | Single-point estimate |

## Red Flags for Overfitting

Watch out for these signals in your results:

- **In-sample >> Out-of-sample Sharpe**: A gap > 0.5 suggests overfitting.
- **Trade frequency too low**: Fewer than 30 trades gives noisy Sharpe estimates.
- **Holdout Sharpe variance high**: The OOS Sharpe is inconsistent across folds.
- **Strategy sensitivity**: Small gene perturbations cause large Sharpe changes.
- **All trades in one regime**: Strategy only works in 2021 bull market, for example.

## References

- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- Bailey, D., Borwein, J., López de Prado, M., & Zhu, Q. J. (2014). Pseudo-mathematics and financial charlatanism. *Notices of the AMS*, 61(5).
- Harvey, C. R., Liu, Y., & Zhu, H. (2016). And the cross-section of expected returns. *Review of Financial Studies*, 29(1).
