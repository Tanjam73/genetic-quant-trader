"""
loader.py
─────────
Historical equity data pipeline.

Downloads and caches OHLCV data from Yahoo Finance (via yfinance).
Supports:
  - Single ticker download
  - Batch download for S&P 500 or custom universe
  - Local caching to Parquet to minimize API calls
  - 9-year default date range (2015–2024)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_START = "2015-01-01"
DEFAULT_END   = "2024-01-01"
CACHE_DIR     = Path("data/cache")


class DataLoader:
    """
    Downloads and manages historical price data.

    Parameters
    ----------
    cache_dir : Path
        Directory where Parquet files are cached. Created if it doesn't exist.
    start_date : str  ISO format (YYYY-MM-DD)
    end_date   : str  ISO format (YYYY-MM-DD)
    """

    def __init__(
        self,
        cache_dir: Path = CACHE_DIR,
        start_date: str = DEFAULT_START,
        end_date:   str = DEFAULT_END,
    ):
        self.cache_dir  = Path(cache_dir)
        self.start_date = start_date
        self.end_date   = end_date
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  Single-ticker loader                                                #
    # ------------------------------------------------------------------ #

    def load_ticker(self, ticker: str, use_cache: bool = True) -> pd.DataFrame:
        """
        Load OHLCV data for a single ticker.

        Returns
        -------
        pd.DataFrame with columns [open, high, low, close, volume]
        and a DatetimeIndex.
        """
        cache_path = self.cache_dir / f"{ticker}_{self.start_date}_{self.end_date}.parquet"

        if use_cache and cache_path.exists():
            logger.debug("Cache hit for %s", ticker)
            return pd.read_parquet(cache_path)

        try:
            import yfinance as yf
            logger.info("Downloading %s (%s → %s)…", ticker, self.start_date, self.end_date)
            df = yf.download(
                ticker,
                start=self.start_date,
                end=self.end_date,
                auto_adjust=True,
                progress=False,
            )
            if df.empty:
                raise ValueError(f"No data returned for {ticker}")

            df.columns = [c.lower() for c in df.columns]
            df.index.name = "date"
            df.to_parquet(cache_path)
            return df

        except ImportError:
            raise ImportError(
                "yfinance is not installed. Run: pip install yfinance"
            )

    # ------------------------------------------------------------------ #
    #  Batch loader for a universe                                         #
    # ------------------------------------------------------------------ #

    def load_universe(
        self, tickers: List[str], min_days: int = 1000
    ) -> Dict[str, pd.DataFrame]:
        """
        Load data for all tickers; drop tickers with insufficient history.

        Returns
        -------
        dict mapping ticker → DataFrame
        """
        data = {}
        for ticker in tickers:
            try:
                df = self.load_ticker(ticker)
                if len(df) >= min_days:
                    data[ticker] = df
                else:
                    logger.warning(
                        "Skipping %s: only %d days of data (min=%d)",
                        ticker, len(df), min_days
                    )
            except Exception as e:
                logger.warning("Failed to load %s: %s", ticker, e)

        logger.info("Loaded %d / %d tickers successfully.", len(data), len(tickers))
        return data

    # ------------------------------------------------------------------ #
    #  Pair price panel builder                                            #
    # ------------------------------------------------------------------ #

    def build_pair_panel(
        self,
        ticker_a: str,
        ticker_b: str,
        price_type: str = "close",
    ) -> pd.DataFrame:
        """
        Build a two-column price DataFrame for a pair.

        Returns
        -------
        pd.DataFrame with columns ['close_a', 'close_b'] on aligned dates.
        """
        df_a = self.load_ticker(ticker_a)[[price_type]].rename(columns={price_type: "close_a"})
        df_b = self.load_ticker(ticker_b)[[price_type]].rename(columns={price_type: "close_b"})
        panel = df_a.join(df_b, how="inner").dropna()
        if len(panel) < 100:
            raise ValueError(
                f"Insufficient overlapping data for pair ({ticker_a}, {ticker_b}): "
                f"{len(panel)} days"
            )
        return panel

    def build_all_pair_panels(
        self,
        pairs: List[Tuple[str, str]],
        price_type: str = "close",
    ) -> Dict[Tuple[str, str], pd.DataFrame]:
        """Build pair panels for all pairs."""
        panels = {}
        for pair in pairs:
            try:
                panels[pair] = self.build_pair_panel(*pair, price_type=price_type)
                logger.debug("Built panel for %s/%s: %d rows", *pair, len(panels[pair]))
            except Exception as e:
                logger.warning("Skipping pair %s: %s", pair, e)
        return panels

    # ------------------------------------------------------------------ #
    #  S&P 500 helper                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def get_sp500_tickers() -> List[str]:
        """
        Fetch the current S&P 500 constituent list from Wikipedia.

        Returns
        -------
        list of ticker strings
        """
        try:
            tables = pd.read_html(
                "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            )
            sp500 = tables[0]["Symbol"].tolist()
            # Clean up ticker symbols (remove dots, etc.)
            sp500 = [t.replace(".", "-") for t in sp500]
            logger.info("Fetched %d S&P 500 tickers from Wikipedia.", len(sp500))
            return sp500
        except Exception as e:
            logger.error("Failed to fetch S&P 500 tickers: %s", e)
            # Return a curated subset as fallback
            return [
                "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
                "JPM", "JNJ", "V", "UNH", "HD", "PG", "MA", "DIS",
                "BAC", "XOM", "PFE", "KO", "PEP", "CSCO", "ABBV", "WMT",
                "CVX", "MRK", "LLY", "TMO", "AVGO", "NKE", "ORCL",
                "ACN", "MCD", "DHR", "BMY", "COST", "ABT", "CRM", "TXN",
            ]
