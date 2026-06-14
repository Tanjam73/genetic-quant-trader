# Chromosome Design

## Overview

This document describes the encoding scheme used to represent a mean-reversion trading strategy as a **chromosome** that the Genetic Algorithm can evolve.

## Design Principles

A good chromosome encoding satisfies three criteria:

1. **Completeness** — every strategy in the feasible search space is representable.
2. **Non-redundancy** — ideally, each valid strategy maps to exactly one chromosome. (Near-redundancy is acceptable; exact injectivity is often too restrictive.)
3. **Locality** — small gene perturbations produce strategies with similar behavior. This is critical for the gradient-free GA search to work efficiently.

## Gene Layout

Each chromosome is a vector of **6 genes**:

```
Index  Name           Type    Bounds        Meaning
─────  ─────────────  ──────  ────────────  ──────────────────────────────────────
  0    pair_idx       int     [0, 49]       Index into pre-screened pair universe
  1    entry_z        float   [1.0, 3.0]    z-score threshold to enter a position
  2    exit_z         float   [0.0, 1.5]    z-score threshold to exit a position
  3    holding_min    int     [1, 5]        Minimum holding period (days)
  4    holding_max    int     [5, 60]       Maximum holding period (days / forced exit)
  5    lookback       int     [20, 252]     Rolling window for spread mean/std (days)
```

## Constraints

The following logical constraints must hold for a chromosome to be valid:

| Constraint | Formula | Reason |
|---|---|---|
| Entry above exit | `entry_z > exit_z` | Otherwise the exit signal fires before a position is ever opened |
| Min holding | `holding_min < holding_max` | Prevents degenerate case |
| Sufficient lookback | `lookback ≥ 5` | Prevents statistically meaningless windows |

Constraints are enforced in two places:
- `Chromosome.random()` — ensures constraints are satisfied at initialization.
- `Chromosome.repair()` — called after every crossover and mutation to fix violations.

## Pair Universe

**Gene 0 (`pair_idx`)** does not directly encode ticker symbols. Instead, it is an index into a **pre-screened universe of 50 cointegrated pairs** that is computed once before evolution begins.

This design choice:
- Reduces the search space significantly (50 pairs vs 500C2 ≈ 125,000 possible pairs)
- Ensures all encoded pairs have statistical merit (cointegration p-value < 0.05, half-life < 30 days)
- Maintains locality: adjacent indices are similarly-scored pairs

## Spread Model

Given a decoded pair `(A, B)`, the **spread** is computed using a rolling OLS hedge ratio:

```
β(t) = rolling_cov(A, B, w) / rolling_var(B, w)
spread(t) = log(A(t)) - β(t) × log(B(t))
```

The **z-score** is then:

```
z(t) = (spread(t) - μ(t)) / σ(t)
```

where `μ(t)` and `σ(t)` are the rolling mean and standard deviation over the lookback window.

## Example Chromosome

```
genes = [12, 2.1, 0.4, 3, 21, 60]

Decoded:
  pair:         ("AAPL", "MSFT")   # pair_universe[12]
  entry_z:      2.1                # enter when |z| > 2.1 sigma
  exit_z:       0.4                # exit when |z| < 0.4 sigma
  holding_min:  3 days             # hold at least 3 days
  holding_max:  21 days            # forced exit at 21 days
  lookback:     60 days            # 60-day rolling window
```

## Fitness Landscape Considerations

The fitness landscape for this chromosome space is **non-convex, multi-modal, and noisy** due to:

- Discrete pair index gene creates discontinuities
- Sharpe Ratio is non-smooth in the holding period and z-score parameters
- Transaction costs introduce flat regions at low trade frequency
- Market regime changes cause fitness to vary across time periods

This makes gradient-free evolutionary search well-suited for this problem.
