"""
portfolio.py
────────────
Tracks capital, positions, and P&L throughout a backtest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class Position:
    """A single open position (one side of a spread trade)."""
    ticker: str
    direction: int        # +1 long, -1 short
    entry_price: float
    shares: float
    entry_date: str


class Portfolio:
    """
    Tracks cash, open positions, and running P&L.

    Parameters
    ----------
    initial_capital : float   Starting capital in dollars.
    position_size   : float   Fraction of capital allocated per leg.
    """

    def __init__(self, initial_capital: float = 1_000_000, position_size: float = 0.10):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position_size = position_size
        self.positions: List[Position] = []
        self.pnl_history: List[float] = []
        self._total_trades = 0
        self._total_costs = 0.0

    def open_position(
        self,
        ticker: str,
        price: float,
        direction: int,
        date: str,
        cost_fraction: float = 0.0,
    ) -> None:
        notional = self.capital * self.position_size
        shares = notional / (price + 1e-10)
        cost = notional * cost_fraction
        self.capital -= cost
        self._total_costs += cost
        self.positions.append(
            Position(
                ticker=ticker,
                direction=direction,
                entry_price=price,
                shares=shares,
                entry_date=date,
            )
        )

    def close_position(
        self,
        ticker: str,
        price: float,
        cost_fraction: float = 0.0,
    ) -> Optional[float]:
        for i, pos in enumerate(self.positions):
            if pos.ticker == ticker:
                pnl = pos.direction * pos.shares * (price - pos.entry_price)
                cost = pos.shares * price * cost_fraction
                self.capital += pnl - cost
                self._total_costs += cost
                self._total_trades += 1
                self.pnl_history.append(pnl)
                self.positions.pop(i)
                return pnl
        return None

    @property
    def total_value(self) -> float:
        return self.capital

    @property
    def total_return(self) -> float:
        return (self.capital - self.initial_capital) / self.initial_capital

    @property
    def n_trades(self) -> int:
        return self._total_trades

    @property
    def win_rate(self) -> float:
        if not self.pnl_history:
            return 0.0
        return sum(1 for p in self.pnl_history if p > 0) / len(self.pnl_history)

    def __repr__(self) -> str:
        return (
            f"Portfolio(capital=${self.capital:,.0f}, "
            f"return={self.total_return:.2%}, trades={self.n_trades})"
        )
