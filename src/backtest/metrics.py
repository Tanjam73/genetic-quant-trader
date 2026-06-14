"""
metrics.py
──────────
Comprehensive performance analytics for backtested strategies.

Computes the full suite of quantitative metrics used in professional
portfolio management contexts:
  - Return metrics: CAGR, total return
  - Risk metrics: volatility, max drawdown, VaR, CVaR
  - Risk-adjusted: Sharpe, Sortino, Calmar, Omega
  - Trade statistics: win rate, avg P&L per trade, profit factor
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

TRADING_DAYS = 252
RISK_FREE_RATE = 0.04


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class PerformanceReport:
    # Returns
    total_return:       float = 0.0
    cagr:               float = 0.0
    ann_volatility:     float = 0.0

    # Risk-adjusted
    sharpe:             float = 0.0
    sortino:            float = 0.0
    calmar:             float = 0.0
    omega:              float = 0.0

    # Drawdown
    max_drawdown:       float = 0.0
    avg_drawdown:       float = 0.0
    max_dd_duration:    int   = 0     # days in max drawdown

    # Value at Risk
    var_95:             float = 0.0
    cvar_95:            float = 0.0

    # Trade statistics
    n_trades:           int   = 0
    win_rate:           float = 0.0
    avg_pnl_per_trade:  float = 0.0
    profit_factor:      float = 0.0
    avg_holding_days:   float = 0.0
    total_costs:        float = 0.0

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    def __str__(self) -> str:
        lines = [
            "=" * 50,
            "  PERFORMANCE REPORT",
            "=" * 50,
            f"  Total Return        : {self.total_return:>8.2%}",
            f"  CAGR                : {self.cagr:>8.2%}",
            f"  Ann. Volatility     : {self.ann_volatility:>8.2%}",
            "-" * 50,
            f"  Sharpe Ratio        : {self.sharpe:>8.3f}",
            f"  Sortino Ratio       : {self.sortino:>8.3f}",
            f"  Calmar Ratio        : {self.calmar:>8.3f}",
            f"  Omega Ratio         : {self.omega:>8.3f}",
            "-" * 50,
            f"  Max Drawdown        : {self.max_drawdown:>8.2%}",
            f"  Avg Drawdown        : {self.avg_drawdown:>8.2%}",
            f"  Max DD Duration     : {self.max_dd_duration:>8} days",
            "-" * 50,
            f"  VaR (95%)           : {self.var_95:>8.2%}",
            f"  CVaR (95%)          : {self.cvar_95:>8.2%}",
            "-" * 50,
            f"  # Trades            : {self.n_trades:>8}",
            f"  Win Rate            : {self.win_rate:>8.1%}",
            f"  Avg P&L / Trade     : {self.avg_pnl_per_trade:>8.4f}",
            f"  Profit Factor       : {self.profit_factor:>8.2f}",
            f"  Avg Holding (days)  : {self.avg_holding_days:>8.1f}",
            f"  Total Costs         : {self.total_costs:>8.4f}",
            "=" * 50,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core analytics
# ---------------------------------------------------------------------------

class PerformanceAnalyzer:
    """Compute comprehensive performance metrics from a daily return series."""

    def __init__(self, rf: float = RISK_FREE_RATE):
        self.rf = rf

    def analyze(
        self,
        daily_returns: pd.Series,
        trades: Optional[list] = None,
        total_costs: float = 0.0,
    ) -> PerformanceReport:
        r = daily_returns.dropna().values.astype(float)
        n = len(r)

        if n < 5:
            return PerformanceReport()

        report = PerformanceReport()
        cum = (1 + r).cumprod()

        # Returns
        report.total_return = cum[-1] - 1
        report.cagr = (cum[-1] ** (TRADING_DAYS / n)) - 1
        report.ann_volatility = r.std() * np.sqrt(TRADING_DAYS)

        # Risk-adjusted
        daily_rf = self.rf / TRADING_DAYS
        excess = r - daily_rf

        std = r.std()
        report.sharpe = (excess.mean() / (std + 1e-10)) * np.sqrt(TRADING_DAYS)

        downside = excess[excess < 0]
        down_std = np.sqrt((downside ** 2).mean()) if len(downside) > 0 else 1e-10
        report.sortino = (excess.mean() / down_std) * np.sqrt(TRADING_DAYS)

        # Drawdown
        cum_series = pd.Series(cum)
        rolling_max = cum_series.cummax()
        dd_series = (cum_series - rolling_max) / rolling_max
        report.max_drawdown = float(abs(dd_series.min()))
        report.avg_drawdown = float(abs(dd_series[dd_series < 0].mean())) if (dd_series < 0).any() else 0.0
        report.calmar = report.cagr / (report.max_drawdown + 1e-10)

        # Max drawdown duration
        in_dd = dd_series < 0
        max_dur, cur_dur = 0, 0
        for v in in_dd:
            if v:
                cur_dur += 1
                max_dur = max(max_dur, cur_dur)
            else:
                cur_dur = 0
        report.max_dd_duration = max_dur

        # VaR / CVaR
        report.var_95  = float(-np.percentile(r, 5))
        cvar_mask = r <= -report.var_95
        report.cvar_95 = float(-r[cvar_mask].mean()) if cvar_mask.any() else report.var_95

        # Omega ratio
        threshold = daily_rf
        gains = r[r > threshold] - threshold
        losses = threshold - r[r <= threshold]
        report.omega = gains.sum() / (losses.sum() + 1e-10)

        # Trade stats
        if trades:
            report.n_trades = len(trades)
            pnls = [t.pnl for t in trades]
            winners = [p for p in pnls if p > 0]
            losers  = [p for p in pnls if p <= 0]
            report.win_rate = len(winners) / len(pnls)
            report.avg_pnl_per_trade = float(np.mean(pnls))
            report.profit_factor = (
                sum(winners) / (abs(sum(losers)) + 1e-10)
            )
            report.avg_holding_days = float(
                np.mean([t.holding_days for t in trades])
            )
        report.total_costs = total_costs

        return report
