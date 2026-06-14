"""
fitness.py
──────────
Fitness evaluation for the Genetic Algorithm.

The fitness function runs a lightweight backtest for a single chromosome
and returns a scalar fitness score. Multiple metrics are supported:
  - Sharpe Ratio  (primary default)
  - Sortino Ratio (penalizes only downside vol)
  - Calmar Ratio  (return / max drawdown)
  - Composite     (weighted combination)

Parallelism
-----------
evaluate_population() uses ProcessPoolExecutor to evaluate all individuals
in parallel across CPU cores. Each chromosome evaluation is stateless and
embarrassingly parallel.
"""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .chromosome import Chromosome

logger = logging.getLogger(__name__)

# Risk-free rate (annualized, used in Sharpe/Sortino)
RISK_FREE_RATE = 0.04
TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Metric functions (operate on daily return series)
# ---------------------------------------------------------------------------

def sharpe_ratio(returns: np.ndarray, rf: float = RISK_FREE_RATE) -> float:
    """Annualized Sharpe Ratio."""
    if len(returns) < 2:
        return -99.0
    daily_rf = rf / TRADING_DAYS
    excess = returns - daily_rf
    std = excess.std()
    if std == 0:
        return 0.0
    return float(excess.mean() / std * np.sqrt(TRADING_DAYS))


def sortino_ratio(returns: np.ndarray, rf: float = RISK_FREE_RATE) -> float:
    """Annualized Sortino Ratio (downside deviation in denominator)."""
    if len(returns) < 2:
        return -99.0
    daily_rf = rf / TRADING_DAYS
    excess = returns - daily_rf
    downside = excess[excess < 0]
    if len(downside) == 0:
        return 10.0    # no losing days → cap at 10
    downside_std = np.sqrt((downside ** 2).mean())
    if downside_std == 0:
        return 0.0
    return float(excess.mean() / downside_std * np.sqrt(TRADING_DAYS))


def calmar_ratio(returns: np.ndarray, rf: float = RISK_FREE_RATE) -> float:
    """Calmar Ratio: annualized return / max drawdown."""
    if len(returns) < 2:
        return -99.0
    cum = (1 + returns).cumprod()
    rolling_max = cum.cummax()
    drawdowns = (cum - rolling_max) / rolling_max
    max_dd = abs(drawdowns.min())
    if max_dd == 0:
        return 10.0
    ann_return = (cum.iloc[-1] ** (TRADING_DAYS / len(returns))) - 1
    return float(ann_return / max_dd)


def compute_fitness(
    returns: np.ndarray,
    metric: str = "sharpe",
    weights: Optional[dict] = None,
) -> float:
    """
    Dispatch to the requested fitness metric.

    Parameters
    ----------
    returns  : daily strategy returns (numpy array)
    metric   : 'sharpe' | 'sortino' | 'calmar' | 'composite'
    weights  : for 'composite', e.g. {'sharpe': 0.5, 'sortino': 0.3, 'calmar': 0.2}
    """
    series = pd.Series(returns)
    if metric == "sharpe":
        return sharpe_ratio(series.values)
    elif metric == "sortino":
        return sortino_ratio(series.values)
    elif metric == "calmar":
        return calmar_ratio(series)
    elif metric == "composite":
        w = weights or {"sharpe": 0.5, "sortino": 0.3, "calmar": 0.2}
        s = sharpe_ratio(series.values)
        so = sortino_ratio(series.values)
        ca = calmar_ratio(series)
        return w.get("sharpe", 0) * s + w.get("sortino", 0) * so + w.get("calmar", 0) * ca
    else:
        raise ValueError(f"Unknown fitness metric: {metric!r}")


# ---------------------------------------------------------------------------
# Single chromosome evaluation (designed to be pickle-able for multiprocessing)
# ---------------------------------------------------------------------------

def _evaluate_one(
    args: Tuple[Chromosome, dict, dict]
) -> Tuple[Chromosome, float]:
    """
    Worker function: evaluate a single chromosome.

    Parameters (packed as tuple for pool.map compatibility)
    ----------
    args : (chromosome, price_data, config)
        price_data : dict[tuple[str,str], pd.DataFrame]  — pre-loaded price panels
        config     : backtest/fitness configuration dict

    Returns
    -------
    (chromosome, fitness_score)
    """
    from src.backtest.engine import BacktestEngine   # local import to avoid circular

    chromosome, price_data, config = args
    pair_universe = config["pair_universe"]
    strategy_spec = chromosome.decode(pair_universe)
    pair = strategy_spec["pair"]

    if pair not in price_data:
        return chromosome, -99.0

    prices = price_data[pair]
    engine = BacktestEngine(config=config)
    result = engine.run(prices=prices, strategy_spec=strategy_spec)
    fitness = compute_fitness(
        result["daily_returns"].values,
        metric=config.get("fitness_metric", "sharpe"),
    )
    return chromosome, fitness


# ---------------------------------------------------------------------------
# Population-level batch evaluation
# ---------------------------------------------------------------------------

def evaluate_population(
    population: List[Chromosome],
    price_data: dict,
    config: dict,
    n_workers: int = 4,
    skip_evaluated: bool = True,
) -> List[Chromosome]:
    """
    Evaluate all chromosomes in the population in parallel.

    Parameters
    ----------
    population      : list of Chromosome instances
    price_data      : dict mapping pair → price DataFrame
    config          : backtest + GA config dict
    n_workers       : number of parallel worker processes
    skip_evaluated  : if True, skip chromosomes that already have fitness set

    Returns
    -------
    List of chromosomes with .fitness updated in-place (well, we rebuild).
    """
    to_evaluate = (
        [c for c in population if c.fitness is None]
        if skip_evaluated
        else population
    )

    if not to_evaluate:
        logger.debug("All chromosomes already evaluated — skipping.")
        return population

    logger.info("Evaluating %d chromosomes with %d workers...", len(to_evaluate), n_workers)

    args_list = [(c, price_data, config) for c in to_evaluate]

    if n_workers == 1:
        # Single-threaded fallback (easier to debug)
        results = [_evaluate_one(args) for args in args_list]
    else:
        results = []
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_evaluate_one, args): args[0] for args in args_list}
            for future in as_completed(futures):
                try:
                    chrom, score = future.result()
                    results.append((chrom, score))
                except Exception as exc:
                    logger.warning("Evaluation failed: %s", exc)
                    chrom = futures[future]
                    results.append((chrom, -99.0))

    # Write fitness back
    fitness_map = {id(chrom): score for chrom, score in results}
    for c in to_evaluate:
        c.fitness = fitness_map.get(id(c), -99.0)

    n_valid = sum(1 for c in population if c.fitness is not None and c.fitness > -99.0)
    logger.info("Evaluation complete. %d / %d valid.", n_valid, len(population))
    return population
