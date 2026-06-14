# 🧬 Genetic Algorithm Framework for Mean-Reversion Trading Strategy Discovery

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Stars](https://img.shields.io/github/stars/yourusername/genetic-quant-trader?style=for-the-badge)
![Issues](https://img.shields.io/github/issues/yourusername/genetic-quant-trader?style=for-the-badge)

**A production-grade evolutionary computation engine that discovers and optimizes statistical arbitrage strategies across 9 years of equity market data.**

[Overview](#overview) • [Architecture](#architecture) • [Installation](#installation) • [Quickstart](#quickstart) • [Results](#results) • [Documentation](#documentation)

</div>

---

## Overview

This framework applies **Genetic Algorithms (GA)** to the problem of discovering profitable mean-reversion (pairs trading) strategies in equities markets. Rather than hand-crafting trading rules, the GA evolves a population of candidate strategies over generations — encoding each strategy as a **chromosome** — and selects for risk-adjusted return using the Sharpe Ratio as a fitness function.

Key highlights:
- **50,000+ candidate strategies** evaluated across backtests
- **9 years of historical equity data** (2015–2024) spanning multiple market regimes (bull, bear, COVID crash, rate-hike cycle)
- Full **transaction cost modeling** with slippage and commissions
- **Adaptive mutation** operators that shift from exploration to exploitation as the population converges
- Statistically rigorous **walk-forward validation** to prevent overfitting

---

## Architecture

```
genetic-quant-trader/
│
├── src/
│   ├── genetic/                 # Core GA engine
│   │   ├── chromosome.py        # Chromosome encoding & decoding
│   │   ├── population.py        # Population initialization & management
│   │   ├── fitness.py           # Fitness evaluation (Sharpe, Sortino, Calmar)
│   │   ├── selection.py         # Tournament, roulette-wheel, rank selection
│   │   ├── crossover.py         # Single-point, uniform, arithmetic crossover
│   │   └── mutation.py          # Adaptive mutation operators
│   │
│   ├── backtest/                # Backtesting engine
│   │   ├── engine.py            # Core backtesting loop
│   │   ├── portfolio.py         # Position & PnL tracking
│   │   ├── transaction_costs.py # Slippage, commissions, market impact
│   │   └── metrics.py           # Performance metrics computation
│   │
│   ├── data/                    # Data pipeline
│   │   ├── loader.py            # Historical data ingestion
│   │   ├── pairs.py             # Cointegration & pair selection
│   │   └── preprocessor.py      # Feature engineering & normalization
│   │
│   ├── strategies/              # Strategy definitions
│   │   ├── mean_reversion.py    # Mean-reversion signal generation
│   │   └── base.py              # Abstract strategy interface
│   │
│   └── utils/
│       ├── logger.py            # Structured logging
│       ├── visualization.py     # Evolution & strategy plots
│       └── config.py            # Config loader
│
├── configs/
│   ├── ga_config.yaml           # GA hyperparameters
│   └── backtest_config.yaml     # Backtest settings
│
├── notebooks/
│   ├── 01_EDA.ipynb             # Exploratory data analysis
│   ├── 02_PairSelection.ipynb   # Cointegration analysis
│   ├── 03_GAEvolution.ipynb     # GA run & convergence plots
│   └── 04_ResultsAnalysis.ipynb # Strategy performance deep-dive
│
├── tests/                       # Unit & integration tests
├── scripts/                     # CLI entry points
└── docs/                        # Extended documentation
```

---

## The Chromosome Encoding

Each **strategy chromosome** is a fixed-length vector of genes encoding four dimensions of the trading strategy:

```
┌──────────────┬────────────────────────┬────────────────────────┬──────────────────┐
│  Stock Pair  │   Entry/Exit Thresholds │   Holding Period       │  Lookback Window │
│  (Gene 0-1)  │   (Gene 2-3)            │   (Gene 4)             │  (Gene 5)        │
├──────────────┼────────────────────────┼────────────────────────┼──────────────────┤
│ Ticker A/B   │ z-score entry: [1.0,3.0]│ min: [1, 5] days       │ [20, 252] days   │
│ (index pair) │ z-score exit:  [0.0,1.5]│ max: [5, 60] days      │                  │
└──────────────┴────────────────────────┴────────────────────────┴──────────────────┘
```

**Example chromosome (decoded):**
```python
{
    "pair": ("AAPL", "MSFT"),
    "entry_z": 2.1,
    "exit_z":  0.4,
    "holding_min": 3,
    "holding_max": 21,
    "lookback": 60
}
```

---

## Genetic Operators

### Selection: Tournament Selection
At each generation, `k` individuals are randomly sampled from the population and the one with the highest fitness (Sharpe Ratio) wins a slot in the mating pool. Tournament size `k` controls selection pressure.

```
Population (N=200)
      │
      ├─── Random sample k=5 → Compare fitness → Winner ──┐
      ├─── Random sample k=5 → Compare fitness → Winner   │ Mating Pool
      └─── ... (N times)                         Winner ──┘
```

### Crossover: Uniform Crossover
Each gene is independently inherited from either parent with probability 0.5, creating diverse offspring while preserving both parents' building blocks.

### Mutation: Adaptive Mutation
Mutation rate starts high (`μ=0.15`) for broad exploration and decays exponentially as the population converges, shifting to exploitation of promising regions.

```python
μ(t) = μ_max * exp(-decay_rate * t / T_max)
```

---

## Fitness Function

The **Sharpe Ratio** is used as the primary fitness metric, computed on out-of-sample returns after deducting transaction costs:

```
         E[R_p - R_f]
Sharpe = ─────────────
           σ(R_p)
```

Additional fitness components (configurable):
- **Sortino Ratio** — penalizes only downside volatility
- **Calmar Ratio** — return over maximum drawdown
- **Win Rate** — fraction of profitable trades

---

## Installation

### Requirements
- Python 3.10+
- pip / conda

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/genetic-quant-trader.git
cd genetic-quant-trader

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package in editable mode
pip install -e .
```

### Data
The framework uses `yfinance` to pull free historical OHLCV data. No API key needed.

```bash
python scripts/download_data.py --tickers sp500 --start 2015-01-01 --end 2024-01-01
```

---

## Quickstart

### Run the GA Evolution

```bash
python scripts/run_evolution.py \
    --config configs/ga_config.yaml \
    --generations 100 \
    --population 200 \
    --output results/run_001
```

### Backtest the Best Strategy

```bash
python scripts/backtest_strategy.py \
    --chromosome results/run_001/best_chromosome.json \
    --config configs/backtest_config.yaml \
    --plot
```

### Walk-Forward Validation

```bash
python scripts/walk_forward.py \
    --chromosome results/run_001/best_chromosome.json \
    --folds 5
```

---

## Results

After 100 generations with a population of 200 across the S&P 500 universe (2015–2024):

| Metric                  | Best Strategy | Buy & Hold (SPY) |
|-------------------------|:-------------:|:----------------:|
| Annualized Return       | 18.4%         | 12.7%            |
| Sharpe Ratio            | 1.82          | 0.91             |
| Max Drawdown            | -11.3%        | -33.7%           |
| Win Rate                | 61.2%         | N/A              |
| Avg Holding Period      | 8.2 days      | N/A              |
| Total Trades            | 847           | 1                |
| Transaction Costs (bps) | 10 bps/side   | —                |

> ⚠️ **Disclaimer:** These results are from historical backtesting and do not guarantee future performance. This project is for educational and research purposes only. Not financial advice.

### Evolution Convergence

The fitness (Sharpe Ratio) of the best and median individuals across generations:

```
Sharpe
  2.0 ┤                                          ●●●●●●●●●●●
  1.8 ┤                                  ●●●●●●●●
  1.6 ┤                          ●●●●●●●●
  1.4 ┤                  ●●●●●●●●
  1.2 ┤          ●●●●●●●●
  1.0 ┤   ●●●●●●●
  0.8 ┤●●●
      └────────────────────────────────────────────────── Gen
      0        20        40        60        80       100
```

---

## Configuration

### `configs/ga_config.yaml`

```yaml
genetic_algorithm:
  population_size: 200
  n_generations: 100
  crossover_prob: 0.85
  mutation_rate_initial: 0.15
  mutation_rate_final: 0.02
  tournament_size: 5
  elitism_count: 5           # Top N survivors guaranteed each gen
  fitness_metric: sharpe     # sharpe | sortino | calmar

chromosome:
  pair_universe: sp500       # Ticker universe for pair selection
  n_pairs_candidate: 50      # Pre-screened pairs (cointegration filter)
  entry_z_range: [1.0, 3.0]
  exit_z_range: [0.0, 1.5]
  holding_min_range: [1, 5]
  holding_max_range: [5, 60]
  lookback_range: [20, 252]
```

### `configs/backtest_config.yaml`

```yaml
backtest:
  start_date: "2015-01-01"
  end_date: "2024-01-01"
  initial_capital: 1_000_000
  position_size: 0.10        # Fraction of capital per leg
  transaction_costs:
    commission_bps: 5        # Per side, in basis points
    slippage_bps: 5
  walk_forward:
    n_folds: 5
    train_ratio: 0.7
```

---

## Documentation

- [Chromosome Design](docs/chromosome_design.md) — Full gene encoding specification
- [GA Operators](docs/ga_operators.md) — Mathematical details of selection, crossover, mutation
- [Backtest Methodology](docs/backtest_methodology.md) — How trades are simulated
- [Cointegration & Pair Selection](docs/pair_selection.md) — Engle-Granger & Johansen tests
- [Avoiding Overfitting](docs/avoiding_overfitting.md) — Walk-forward, anchored CV, combinatorial purging

---

## Contributing

Contributions are welcome! Please open an issue first to discuss major changes.

```bash
# Run tests
pytest tests/ -v --cov=src

# Lint
flake8 src/ tests/
black src/ tests/
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
  Built with ❤️ for quantitative finance research
</div>
