"""
visualization.py
────────────────
Plotting utilities for GA evolution diagnostics and strategy performance.

Charts:
  1. Evolution convergence (best & median fitness per generation)
  2. Population diversity over time
  3. Adaptive mutation rate schedule
  4. Strategy cumulative return curve
  5. Drawdown plot
  6. Trade P&L distribution
  7. Spread and z-score time series
  8. Walk-forward validation results
"""

from __future__ import annotations

from typing import List, Optional

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

COLORS = {
    "primary":   "#1f77b4",
    "secondary": "#ff7f0e",
    "green":     "#2ca02c",
    "red":       "#d62728",
    "gray":      "#7f7f7f",
    "purple":    "#9467bd",
}

def _set_style():
    plt.rcParams.update({
        "font.family":        "sans-serif",
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.grid":          True,
        "grid.alpha":         0.3,
        "figure.dpi":         120,
    })


# ---------------------------------------------------------------------------
# 1. Evolution convergence
# ---------------------------------------------------------------------------

def plot_evolution(history: List[dict], save_path: Optional[str] = None):
    """
    Plot best and median fitness across generations.

    Parameters
    ----------
    history : list of dicts with keys: generation, best_fitness, median_fitness
    """
    _set_style()
    gens    = [h["generation"] for h in history]
    best    = [h["best_fitness"] for h in history]
    median  = [h["median_fitness"] for h in history]
    div     = [h.get("diversity", 0) for h in history]
    mu      = [h.get("mutation_rate", 0) for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("Genetic Algorithm Evolution", fontsize=14, fontweight="bold")

    # Fitness convergence
    ax = axes[0, 0]
    ax.plot(gens, best, color=COLORS["primary"], label="Best", linewidth=2)
    ax.plot(gens, median, color=COLORS["secondary"], label="Median", linewidth=1.5, linestyle="--")
    ax.fill_between(gens, median, best, alpha=0.15, color=COLORS["primary"])
    ax.set_xlabel("Generation"); ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Fitness Convergence"); ax.legend()

    # Diversity
    ax = axes[0, 1]
    ax.plot(gens, div, color=COLORS["purple"], linewidth=2)
    ax.set_xlabel("Generation"); ax.set_ylabel("Gene-space Diversity")
    ax.set_title("Population Diversity")

    # Mutation rate
    ax = axes[1, 0]
    ax.plot(gens, mu, color=COLORS["red"], linewidth=2)
    ax.set_xlabel("Generation"); ax.set_ylabel("Mutation Rate μ")
    ax.set_title("Adaptive Mutation Schedule")

    # Best fitness histogram (final generation)
    ax = axes[1, 1]
    ax.hist(best, bins=20, color=COLORS["green"], alpha=0.7, edgecolor="white")
    ax.axvline(max(f for f in best if f), color=COLORS["red"],
               linestyle="--", label=f"Best = {max(f for f in best if f):.3f}")
    ax.set_xlabel("Sharpe Ratio"); ax.set_ylabel("Count")
    ax.set_title("Best Fitness Distribution"); ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 2. Strategy performance dashboard
# ---------------------------------------------------------------------------

def plot_strategy_performance(
    daily_returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    trades: Optional[list] = None,
    title: str = "Strategy Performance",
    save_path: Optional[str] = None,
):
    """
    Full performance dashboard: cumulative return, drawdown, trade PnL.
    """
    _set_style()
    cum_ret    = (1 + daily_returns).cumprod()
    rolling_max = cum_ret.cummax()
    drawdown   = (cum_ret - rolling_max) / rolling_max

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(title, fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4)

    # Cumulative return
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(cum_ret.index, cum_ret.values, color=COLORS["primary"], linewidth=2, label="Strategy")
    if benchmark_returns is not None:
        bm_cum = (1 + benchmark_returns).cumprod()
        ax1.plot(bm_cum.index, bm_cum.values, color=COLORS["gray"],
                 linewidth=1.5, linestyle="--", label="Benchmark (SPY)")
    ax1.set_ylabel("Portfolio Value (normalized)")
    ax1.set_title("Cumulative Return"); ax1.legend()

    # Drawdown
    ax2 = fig.add_subplot(gs[1, :])
    ax2.fill_between(drawdown.index, drawdown.values, 0,
                     color=COLORS["red"], alpha=0.5)
    ax2.plot(drawdown.index, drawdown.values, color=COLORS["red"], linewidth=0.8)
    ax2.set_ylabel("Drawdown"); ax2.set_title("Drawdown")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))

    # Daily return distribution
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.hist(daily_returns.values, bins=50,
             color=COLORS["primary"], alpha=0.7, edgecolor="white")
    ax3.axvline(0, color="black", linewidth=0.8)
    ax3.axvline(daily_returns.mean(), color=COLORS["green"],
                linestyle="--", label=f"Mean = {daily_returns.mean():.4f}")
    ax3.set_xlabel("Daily Return"); ax3.set_ylabel("Frequency")
    ax3.set_title("Return Distribution"); ax3.legend()

    # Rolling Sharpe (63-day)
    ax4 = fig.add_subplot(gs[2, 1])
    rolling_sharpe = (
        daily_returns.rolling(63).mean() /
        (daily_returns.rolling(63).std() + 1e-10)
    ) * np.sqrt(252)
    ax4.plot(rolling_sharpe.index, rolling_sharpe.values,
             color=COLORS["purple"], linewidth=1.5)
    ax4.axhline(0, color="black", linewidth=0.8)
    ax4.axhline(1, color=COLORS["green"], linestyle="--", alpha=0.7, label="Sharpe = 1")
    ax4.set_ylabel("Rolling Sharpe"); ax4.set_title("63-Day Rolling Sharpe"); ax4.legend()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 3. Walk-forward validation results
# ---------------------------------------------------------------------------

def plot_walk_forward(fold_results: List[dict], save_path: Optional[str] = None):
    """
    Plot in-sample vs out-of-sample Sharpe across walk-forward folds.

    Parameters
    ----------
    fold_results : list of dicts with keys: fold, is_sharpe, oos_sharpe
    """
    _set_style()
    folds = [r["fold"] for r in fold_results]
    is_  = [r["is_sharpe"] for r in fold_results]
    oos  = [r["oos_sharpe"] for r in fold_results]

    x = np.arange(len(folds))
    fig, ax = plt.subplots(figsize=(10, 5))
    w = 0.35
    ax.bar(x - w / 2, is_,  w, label="In-Sample",     color=COLORS["primary"],   alpha=0.8)
    ax.bar(x + w / 2, oos, w, label="Out-of-Sample", color=COLORS["secondary"], alpha=0.8)
    ax.axhline(1.0, color=COLORS["green"], linestyle="--", alpha=0.7, label="Sharpe = 1")
    ax.axhline(0.0, color="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {f}" for f in folds])
    ax.set_ylabel("Sharpe Ratio"); ax.set_title("Walk-Forward Validation: IS vs OOS")
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    return fig
