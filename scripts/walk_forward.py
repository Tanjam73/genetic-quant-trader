"""
walk_forward.py
───────────────
Walk-forward validation for a given chromosome.

Splits 9 years of data into expanding training windows and
evaluates the best chromosome on each unseen out-of-sample fold.

Usage:
    python scripts/walk_forward.py \
        --chromosome results/run_001/best_chromosome.json \
        --folds 5 \
        --plot
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.genetic.chromosome import Chromosome
from src.backtest.engine import BacktestEngine
from src.backtest.metrics import PerformanceAnalyzer
from src.data.loader import DataLoader
from src.utils.visualization import plot_walk_forward


def run_walk_forward(
    chromosome: Chromosome,
    pair_panel: pd.DataFrame,
    strategy_spec: dict,
    config: dict,
    n_folds: int = 5,
    train_ratio: float = 0.70,
) -> list:
    """
    Run walk-forward validation.

    Returns list of per-fold result dicts.
    """
    n = len(pair_panel)
    fold_size = n // n_folds
    results = []
    analyzer = PerformanceAnalyzer()
    engine = BacktestEngine(config=config)

    for fold in range(n_folds):
        # Expanding window: train always starts from day 0
        train_end = int(n * train_ratio) + fold * (fold_size // n_folds)
        test_start = train_end
        test_end   = min(test_start + fold_size, n)

        if test_end - test_start < 60:
            continue  # too few test days

        train_data = pair_panel.iloc[:train_end]
        test_data  = pair_panel.iloc[test_start:test_end]

        # In-sample
        is_result = engine.run(prices=train_data, strategy_spec=strategy_spec)
        is_report = analyzer.analyze(is_result.daily_returns, is_result.trades)

        # Out-of-sample
        oos_result = engine.run(prices=test_data, strategy_spec=strategy_spec)
        oos_report = analyzer.analyze(oos_result.daily_returns, oos_result.trades)

        results.append({
            "fold":       fold + 1,
            "train_days": len(train_data),
            "test_days":  len(test_data),
            "is_sharpe":  is_report.sharpe,
            "oos_sharpe": oos_report.sharpe,
            "is_cagr":    is_report.cagr,
            "oos_cagr":   oos_report.cagr,
            "is_maxdd":   is_report.max_drawdown,
            "oos_maxdd":  oos_report.max_drawdown,
            "oos_trades": oos_result.n_trades,
        })
        print(
            f"Fold {fold+1}: IS Sharpe={is_report.sharpe:.3f}  "
            f"OOS Sharpe={oos_report.sharpe:.3f}  "
            f"OOS Trades={oos_result.n_trades}"
        )

    avg_oos_sharpe = sum(r["oos_sharpe"] for r in results) / max(len(results), 1)
    print(f"\nAverage OOS Sharpe: {avg_oos_sharpe:.3f}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chromosome", required=True)
    parser.add_argument("--config", default="configs/backtest_config.yaml")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--pair-universe", default="results/run_001/pair_universe.json")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    chrom = Chromosome.load(args.chromosome)
    with open(args.config) as f:
        config = yaml.safe_load(f)["backtest"]

    with open(args.pair_universe) as f:
        pair_universe = [tuple(p) for p in json.load(f)]

    spec = chrom.decode(pair_universe)
    print(f"Strategy: {spec}")

    loader = DataLoader(
        start_date=config["start_date"],
        end_date=config["end_date"],
    )
    pair_panel = loader.build_pair_panel(*spec["pair"])

    results = run_walk_forward(
        chromosome=chrom,
        pair_panel=pair_panel,
        strategy_spec=spec,
        config=config,
        n_folds=args.folds,
    )

    if args.plot:
        fig = plot_walk_forward(results, save_path="results/walk_forward.png")
        fig.show()


if __name__ == "__main__":
    main()
