"""
download_data.py
────────────────
CLI script to pre-download and cache historical price data.

Usage:
    python scripts/download_data.py --tickers sp500 --start 2015-01-01 --end 2024-01-01
    python scripts/download_data.py --tickers AAPL MSFT JPM BAC XOM CVX
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import DataLoader


def main():
    parser = argparse.ArgumentParser(
        description="Download and cache historical equity data."
    )
    parser.add_argument(
        "--tickers", nargs="+", default=["sp500"],
        help="Ticker symbols or 'sp500' for full S&P 500 universe."
    )
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end",   default="2024-01-01")
    parser.add_argument("--cache-dir", default="data/cache")
    args = parser.parse_args()

    loader = DataLoader(
        cache_dir=Path(args.cache_dir),
        start_date=args.start,
        end_date=args.end,
    )

    if args.tickers == ["sp500"]:
        tickers = DataLoader.get_sp500_tickers()
        print(f"Downloading S&P 500: {len(tickers)} tickers…")
    else:
        tickers = args.tickers
        print(f"Downloading {len(tickers)} tickers: {tickers}")

    data = loader.load_universe(tickers)
    print(f"\nDone. Successfully cached {len(data)}/{len(tickers)} tickers.")
    print(f"Cache location: {loader.cache_dir.resolve()}")


if __name__ == "__main__":
    main()
