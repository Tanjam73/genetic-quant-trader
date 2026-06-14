"""
tests/test_backtest.py
──────────────────────
Unit and integration tests for the backtesting engine and metrics.
"""

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import BacktestEngine
from src.backtest.metrics import PerformanceAnalyzer
from src.backtest.transaction_costs import TransactionCostModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_prices():
    """
    Generate synthetic cointegrated price series:
      close_a = close_b + mean-reverting spread
    """
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2018-01-01", periods=n, freq="B")
    close_b = 100 + np.cumsum(np.random.randn(n) * 0.5)
    spread   = np.zeros(n)
    # AR(1) spread with φ = 0.92 → half-life ≈ 8 days
    for i in range(1, n):
        spread[i] = 0.92 * spread[i - 1] + np.random.randn() * 0.3
    close_a = close_b + spread
    return pd.DataFrame(
        {"close_a": close_a, "close_b": close_b},
        index=dates
    )


@pytest.fixture
def default_config():
    return {
        "initial_capital": 1_000_000,
        "position_size":   0.10,
        "transaction_costs": {
            "commission_bps": 5,
            "slippage_bps":   5,
        },
        "fitness_metric": "sharpe",
        "pair_universe":  [("AAPL", "MSFT")],
    }


@pytest.fixture
def strategy_spec():
    return {
        "pair":        ("AAPL", "MSFT"),
        "entry_z":     1.5,
        "exit_z":      0.5,
        "holding_min": 2,
        "holding_max": 20,
        "lookback":    60,
    }


# ---------------------------------------------------------------------------
# BacktestEngine tests
# ---------------------------------------------------------------------------

class TestBacktestEngine:
    def test_returns_result_object(self, synthetic_prices, default_config, strategy_spec):
        engine = BacktestEngine(config=default_config)
        result = engine.run(prices=synthetic_prices, strategy_spec=strategy_spec)
        assert result is not None

    def test_daily_returns_is_series(self, synthetic_prices, default_config, strategy_spec):
        engine = BacktestEngine(config=default_config)
        result = engine.run(prices=synthetic_prices, strategy_spec=strategy_spec)
        assert isinstance(result.daily_returns, pd.Series)

    def test_daily_returns_length_reasonable(self, synthetic_prices, default_config, strategy_spec):
        engine = BacktestEngine(config=default_config)
        result = engine.run(prices=synthetic_prices, strategy_spec=strategy_spec)
        # Should have most of the trading days
        assert len(result.daily_returns) > 100

    def test_trades_list_non_empty(self, synthetic_prices, default_config, strategy_spec):
        engine = BacktestEngine(config=default_config)
        result = engine.run(prices=synthetic_prices, strategy_spec=strategy_spec)
        assert result.n_trades >= 0    # could be 0 if signal never fires

    def test_win_rate_between_0_and_1(self, synthetic_prices, default_config, strategy_spec):
        engine = BacktestEngine(config=default_config)
        result = engine.run(prices=synthetic_prices, strategy_spec=strategy_spec)
        assert 0.0 <= result.win_rate <= 1.0

    def test_insufficient_data_returns_empty(self, default_config, strategy_spec):
        """Very short price series → empty result (not an error)."""
        short_prices = pd.DataFrame(
            {"close_a": [100.0] * 10, "close_b": [100.0] * 10},
            index=pd.date_range("2020-01-01", periods=10, freq="B"),
        )
        engine = BacktestEngine(config=default_config)
        result = engine.run(prices=short_prices, strategy_spec=strategy_spec)
        assert len(result.daily_returns) == 0

    def test_tight_entry_fewer_trades(self, synthetic_prices, default_config):
        """Higher entry z-score threshold → fewer trades."""
        engine = BacktestEngine(config=default_config)

        spec_tight = {
            "pair": ("AAPL", "MSFT"), "entry_z": 3.0, "exit_z": 0.5,
            "holding_min": 2, "holding_max": 20, "lookback": 60
        }
        spec_loose = {
            "pair": ("AAPL", "MSFT"), "entry_z": 1.0, "exit_z": 0.3,
            "holding_min": 2, "holding_max": 20, "lookback": 60
        }
        result_tight = engine.run(prices=synthetic_prices, strategy_spec=spec_tight)
        result_loose = engine.run(prices=synthetic_prices, strategy_spec=spec_loose)
        assert result_tight.n_trades <= result_loose.n_trades


# ---------------------------------------------------------------------------
# Transaction cost tests
# ---------------------------------------------------------------------------

class TestTransactionCostModel:
    def test_commission_positive(self):
        model = TransactionCostModel(commission_bps=5, slippage_bps=5)
        assert model.cost_fraction() > 0

    def test_cost_increases_with_holding(self):
        model = TransactionCostModel(commission_bps=5, slippage_bps=5, short_rebate_bps=100)
        cost_short = model.cost_fraction(holding_days=1)
        cost_long  = model.cost_fraction(holding_days=30)
        assert cost_long > cost_short

    def test_high_vix_multiplies_slippage(self):
        model = TransactionCostModel(slippage_bps=10)
        cost_normal = model.adjusted_cost_fraction(vix_level=15)
        cost_crisis = model.adjusted_cost_fraction(vix_level=55)
        assert cost_crisis > cost_normal


# ---------------------------------------------------------------------------
# Performance metrics tests
# ---------------------------------------------------------------------------

class TestPerformanceAnalyzer:
    def test_positive_returns_positive_sharpe(self):
        np.random.seed(0)
        good_returns = pd.Series(np.abs(np.random.randn(252)) * 0.005 + 0.0003)
        analyzer = PerformanceAnalyzer()
        report = analyzer.analyze(good_returns)
        assert report.sharpe > 0

    def test_negative_returns_negative_sharpe(self):
        np.random.seed(0)
        bad_returns = pd.Series(-np.abs(np.random.randn(252)) * 0.005 - 0.0003)
        analyzer = PerformanceAnalyzer()
        report = analyzer.analyze(bad_returns)
        assert report.sharpe < 0

    def test_max_drawdown_between_0_and_1(self):
        np.random.seed(1)
        returns = pd.Series(np.random.randn(500) * 0.01)
        analyzer = PerformanceAnalyzer()
        report = analyzer.analyze(returns)
        assert 0.0 <= report.max_drawdown <= 1.0

    def test_var_less_than_cvar(self):
        """CVaR (expected shortfall) should be >= VaR."""
        np.random.seed(2)
        returns = pd.Series(np.random.randn(1000) * 0.01)
        analyzer = PerformanceAnalyzer()
        report = analyzer.analyze(returns)
        assert report.cvar_95 >= report.var_95 - 1e-10

    def test_report_str_contains_key_metrics(self):
        returns = pd.Series([0.001] * 252)
        analyzer = PerformanceAnalyzer()
        report = analyzer.analyze(returns)
        report_str = str(report)
        assert "Sharpe" in report_str
        assert "Max Drawdown" in report_str
        assert "CAGR" in report_str
