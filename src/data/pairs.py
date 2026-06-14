"""
pairs.py
────────
Statistical pair selection using cointegration analysis.

Pipeline:
  1. Compute all pairwise correlations (fast pre-filter)
  2. Run Engle-Granger cointegration test on correlated candidates
  3. Optionally run Johansen test for multi-variate cointegration
  4. Rank pairs by p-value and half-life; return top N

Half-life
─────────
The half-life of mean reversion measures how quickly the spread
reverts to its mean (in days). Estimated via AR(1) on the spread:

    spread(t) - spread(t-1) = β * spread(t-1) + ε(t)
    half_life = -log(2) / log(1 + β)

Shorter half-life → faster reversion → more trading opportunities.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PairStats:
    """Statistics for a cointegrated pair."""
    pair:           Tuple[str, str]
    correlation:    float
    coint_pvalue:   float           # Engle-Granger p-value
    half_life:      float           # mean-reversion half-life (days)
    hedge_ratio:    float           # OLS beta
    spread_std:     float
    score:          float = 0.0     # composite ranking score

    def is_tradeable(
        self,
        max_pvalue: float = 0.05,
        max_half_life: float = 30.0,
        min_half_life: float = 2.0,
    ) -> bool:
        """Return True if the pair meets trading criteria."""
        return (
            self.coint_pvalue < max_pvalue
            and min_half_life < self.half_life < max_half_life
        )


class PairSelector:
    """
    Screens a universe of stocks for cointegrated pairs suitable for
    mean-reversion trading.

    Parameters
    ----------
    price_data    : dict[str, pd.DataFrame]  ticker → OHLCV DataFrame
    n_pairs       : int  maximum number of pairs to return
    lookback_days : int  history to use for cointegration tests
    min_corr      : float  minimum absolute correlation for pre-filtering
    max_pvalue    : float  cointegration test significance threshold
    """

    def __init__(
        self,
        price_data: Dict[str, pd.DataFrame],
        n_pairs: int = 50,
        lookback_days: int = 252,
        min_corr: float = 0.70,
        max_pvalue: float = 0.05,
    ):
        self.price_data    = price_data
        self.n_pairs       = n_pairs
        self.lookback_days = lookback_days
        self.min_corr      = min_corr
        self.max_pvalue    = max_pvalue

    def select(self) -> List[PairStats]:
        """
        Run the full pair selection pipeline.

        Returns
        -------
        list of PairStats, sorted by composite score (best first),
        length <= n_pairs.
        """
        logger.info("Starting pair selection from %d tickers…", len(self.price_data))

        # Step 1: Build a log-price matrix (aligned dates)
        prices = self._build_price_matrix()
        if prices.empty:
            return []

        # Step 2: Correlation pre-filter
        candidates = self._correlation_filter(prices)
        logger.info("%d candidate pairs after correlation filter.", len(candidates))

        # Step 3: Cointegration test
        pairs = self._cointegration_screen(prices, candidates)
        logger.info("%d pairs passed cointegration test (p < %.2f).", len(pairs), self.max_pvalue)

        # Step 4: Rank and return top N
        ranked = sorted(pairs, key=lambda p: p.score, reverse=True)
        selected = ranked[: self.n_pairs]
        logger.info("Selected %d final pairs.", len(selected))
        return selected

    # ------------------------------------------------------------------ #
    #  Pipeline steps                                                      #
    # ------------------------------------------------------------------ #

    def _build_price_matrix(self) -> pd.DataFrame:
        """Align all tickers on common dates; use log prices."""
        series = {}
        for ticker, df in self.price_data.items():
            if "close" in df.columns:
                series[ticker] = np.log(df["close"].dropna())
        if not series:
            return pd.DataFrame()
        matrix = pd.DataFrame(series).dropna()
        return matrix.iloc[-self.lookback_days:] if len(matrix) > self.lookback_days else matrix

    def _correlation_filter(
        self, prices: pd.DataFrame
    ) -> List[Tuple[str, str]]:
        """Return all ticker pairs with |correlation| >= min_corr."""
        tickers = list(prices.columns)
        corr_matrix = prices.corr()
        candidates = []
        for i in range(len(tickers)):
            for j in range(i + 1, len(tickers)):
                a, b = tickers[i], tickers[j]
                corr = corr_matrix.loc[a, b]
                if abs(corr) >= self.min_corr:
                    candidates.append((a, b))
        return candidates

    def _cointegration_screen(
        self,
        prices: pd.DataFrame,
        candidates: List[Tuple[str, str]],
    ) -> List[PairStats]:
        """Run Engle-Granger test and compute half-life for each candidate."""
        try:
            from statsmodels.tsa.stattools import coint
        except ImportError:
            raise ImportError("statsmodels required: pip install statsmodels")

        results = []
        for a, b in candidates:
            try:
                series_a = prices[a].values
                series_b = prices[b].values

                # Engle-Granger cointegration test
                _, pvalue, _ = coint(series_a, series_b)

                if pvalue >= self.max_pvalue:
                    continue

                # OLS hedge ratio
                beta = np.cov(series_a, series_b)[0, 1] / (np.var(series_b) + 1e-10)

                # Spread
                spread = series_a - beta * series_b
                spread_std = spread.std()

                # Half-life via AR(1)
                half_life = self._compute_half_life(spread)
                if half_life <= 0 or half_life > 252:
                    continue

                # Composite ranking score (lower pvalue + shorter half-life = better)
                corr = np.corrcoef(series_a, series_b)[0, 1]
                score = (
                    (1 - pvalue) * 0.4
                    + (1 / (half_life + 1)) * 0.4
                    + abs(corr) * 0.2
                )

                results.append(
                    PairStats(
                        pair=(a, b),
                        correlation=float(corr),
                        coint_pvalue=float(pvalue),
                        half_life=float(half_life),
                        hedge_ratio=float(beta),
                        spread_std=float(spread_std),
                        score=float(score),
                    )
                )
            except Exception as e:
                logger.debug("Pair (%s, %s) failed: %s", a, b, e)

        return results

    @staticmethod
    def _compute_half_life(spread: np.ndarray) -> float:
        """
        Estimate mean-reversion half-life via OLS AR(1) regression on spread differences.

        half_life = -log(2) / log(1 + beta_hat)
        """
        spread = np.asarray(spread, dtype=float)
        delta = np.diff(spread)
        lag   = spread[:-1]

        # Remove means for regression
        lag   = lag - lag.mean()
        delta = delta - delta.mean()

        if lag.std() < 1e-10:
            return np.inf

        # OLS slope
        beta = np.dot(lag, delta) / (np.dot(lag, lag) + 1e-10)

        if beta >= 0:
            return np.inf    # not mean-reverting

        return float(-np.log(2) / np.log(1 + beta))
