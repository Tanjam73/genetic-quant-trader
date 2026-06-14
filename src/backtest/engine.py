"""
engine.py
─────────
Core backtesting engine for mean-reversion (pairs trading) strategies.

Simulates trading in a spread (long leg A / short leg B, or vice versa)
based on z-score signals derived from a rolling spread model.

Transaction cost model
──────────────────────
Every trade deducts:
  cost_per_trade = 2 * (commission_bps + slippage_bps) / 10_000
  (factor of 2 because we trade both legs)

Market regime handling
──────────────────────
The backtest spans multiple regimes (bull, COVID crash, rate-hike cycle).
No regime filtering is applied here — the GA is expected to discover
strategies that are robust across regimes via fitness selection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .transaction_costs import TransactionCostModel
from .portfolio import Portfolio

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trade record
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    entry_date: str
    exit_date:  Optional[str]
    direction:  int          # +1 long spread, -1 short spread
    entry_z:    float
    exit_z:     Optional[float]
    pnl:        float = 0.0
    holding_days: int = 0
    forced_exit:  bool = False


# ---------------------------------------------------------------------------
# Backtest result container
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    daily_returns:  pd.Series = field(default_factory=pd.Series)
    cumulative_pnl: pd.Series = field(default_factory=pd.Series)
    trades:         list = field(default_factory=list)
    n_trades:       int = 0
    win_rate:       float = 0.0
    avg_holding:    float = 0.0
    total_cost:     float = 0.0


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class BacktestEngine:
    """
    Vectorized event-driven backtester for pairs mean-reversion strategies.

    Parameters
    ----------
    config : dict
        Must contain:
          - initial_capital  (float)
          - position_size    (float, fraction of capital per leg)
          - transaction_costs.commission_bps
          - transaction_costs.slippage_bps
    """

    def __init__(self, config: dict):
        self.config = config
        self.initial_capital = config.get("initial_capital", 1_000_000)
        self.position_size = config.get("position_size", 0.10)
        tc_cfg = config.get("transaction_costs", {})
        self.cost_model = TransactionCostModel(
            commission_bps=tc_cfg.get("commission_bps", 5),
            slippage_bps=tc_cfg.get("slippage_bps", 5),
        )

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def run(
        self,
        prices: pd.DataFrame,
        strategy_spec: dict,
    ) -> BacktestResult:
        """
        Execute a backtest for one strategy specification.

        Parameters
        ----------
        prices : pd.DataFrame
            Columns ['close_a', 'close_b'], DatetimeIndex.
        strategy_spec : dict
            Decoded chromosome: pair, entry_z, exit_z, holding_min,
            holding_max, lookback.

        Returns
        -------
        BacktestResult
        """
        entry_z   = strategy_spec["entry_z"]
        exit_z    = strategy_spec["exit_z"]
        hold_min  = strategy_spec["holding_min"]
        hold_max  = strategy_spec["holding_max"]
        lookback  = strategy_spec["lookback"]

        if len(prices) < lookback + 20:
            # Not enough data to compute meaningful statistics
            return BacktestResult(
                daily_returns=pd.Series(dtype=float),
            )

        # Step 1: Build the spread
        spread = self._compute_spread(prices, lookback)

        # Step 2: Compute rolling z-score
        z_scores = self._compute_zscore(spread, lookback)

        # Step 3: Simulate trades
        result = self._simulate_trades(
            prices=prices,
            z_scores=z_scores,
            entry_z=entry_z,
            exit_z=exit_z,
            hold_min=hold_min,
            hold_max=hold_max,
        )
        return result

    # ------------------------------------------------------------------ #
    #  Spread model                                                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _compute_spread(prices: pd.DataFrame, lookback: int) -> pd.Series:
        """
        OLS hedge ratio estimated on a rolling window.
        spread(t) = close_a(t) - β(t) * close_b(t)
        """
        close_a = prices["close_a"]
        close_b = prices["close_b"]

        betas = pd.Series(index=prices.index, dtype=float)
        for i in range(lookback, len(prices)):
            window_a = close_a.iloc[i - lookback: i].values
            window_b = close_b.iloc[i - lookback: i].values
            # OLS: β = Cov(a, b) / Var(b)
            b = np.cov(window_a, window_b)[0, 1] / (np.var(window_b) + 1e-10)
            betas.iloc[i] = b

        spread = close_a - betas * close_b
        return spread.dropna()

    @staticmethod
    def _compute_zscore(spread: pd.Series, lookback: int) -> pd.Series:
        """Rolling z-score of the spread."""
        mu    = spread.rolling(lookback).mean()
        sigma = spread.rolling(lookback).std()
        z = (spread - mu) / (sigma + 1e-10)
        return z.dropna()

    # ------------------------------------------------------------------ #
    #  Trade simulation                                                    #
    # ------------------------------------------------------------------ #

    def _simulate_trades(
        self,
        prices: pd.DataFrame,
        z_scores: pd.Series,
        entry_z: float,
        exit_z: float,
        hold_min: int,
        hold_max: int,
    ) -> BacktestResult:
        """
        Event-driven trade simulation over the z-score series.

        Signal logic:
          z >  +entry_z  → short spread (sell A, buy B)
          z <  -entry_z  → long  spread (buy A, sell B)
          |z| <  exit_z  → close position
          holding_days ≥ hold_max → forced exit
          holding_days <  hold_min → no exit allowed
        """
        portfolio = Portfolio(
            initial_capital=self.initial_capital,
            position_size=self.position_size,
        )
        trades = []
        total_cost = 0.0

        position    = 0         # +1 long spread, -1 short spread, 0 flat
        entry_price_a = entry_price_b = 0.0
        entry_date  = None
        holding_days = 0

        common_idx = z_scores.index.intersection(prices.index)
        z_series   = z_scores.loc[common_idx]
        price_data = prices.loc[common_idx]

        daily_pnl = pd.Series(0.0, index=common_idx)

        for i, date in enumerate(common_idx):
            z = z_series.loc[date]
            pa = price_data.loc[date, "close_a"]
            pb = price_data.loc[date, "close_b"]

            if position != 0:
                holding_days += 1

                # Mark-to-market daily PnL
                if position == 1:    # long A, short B
                    daily_pnl.loc[date] = (
                        (pa - entry_price_a) / (entry_price_a + 1e-10)
                        - (pb - entry_price_b) / (entry_price_b + 1e-10)
                    ) * self.position_size
                else:                # short A, long B
                    daily_pnl.loc[date] = (
                        (pb - entry_price_b) / (entry_price_b + 1e-10)
                        - (pa - entry_price_a) / (entry_price_a + 1e-10)
                    ) * self.position_size

                # Exit conditions
                should_exit = (
                    abs(z) < exit_z and holding_days >= hold_min
                ) or (holding_days >= hold_max)

                if should_exit:
                    cost = self.cost_model.cost_fraction()
                    daily_pnl.loc[date] -= cost
                    total_cost += cost
                    trade_pnl = daily_pnl.loc[entry_date: date].sum()
                    trades.append(
                        Trade(
                            entry_date=str(entry_date),
                            exit_date=str(date),
                            direction=position,
                            entry_z=float(z_series.loc[entry_date]) if entry_date in z_series.index else 0.0,
                            exit_z=float(z),
                            pnl=trade_pnl,
                            holding_days=holding_days,
                            forced_exit=(holding_days >= hold_max),
                        )
                    )
                    position = 0
                    holding_days = 0

            else:
                # Flat — check for entry signals
                if z > entry_z:
                    position = -1    # short spread
                    entry_price_a = pa
                    entry_price_b = pb
                    entry_date = date
                    holding_days = 0
                    cost = self.cost_model.cost_fraction()
                    daily_pnl.loc[date] -= cost
                    total_cost += cost

                elif z < -entry_z:
                    position = 1     # long spread
                    entry_price_a = pa
                    entry_price_b = pb
                    entry_date = date
                    holding_days = 0
                    cost = self.cost_model.cost_fraction()
                    daily_pnl.loc[date] -= cost
                    total_cost += cost

        # Compute cumulative PnL
        cum_pnl = (1 + daily_pnl).cumprod()

        # Aggregate trade statistics
        n_trades = len(trades)
        win_rate = sum(1 for t in trades if t.pnl > 0) / max(n_trades, 1)
        avg_holding = sum(t.holding_days for t in trades) / max(n_trades, 1)

        return BacktestResult(
            daily_returns=daily_pnl,
            cumulative_pnl=cum_pnl,
            trades=trades,
            n_trades=n_trades,
            win_rate=win_rate,
            avg_holding=avg_holding,
            total_cost=total_cost,
        )
