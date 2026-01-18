"""
Simple helper to download time series from Yahoo Finance.
Depends on: yfinance, pandas
Install: pip install yfinance pandas
"""
from typing import Optional
import argparse

import pandas as pd

try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None


def download_timeseries(
    ticker: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    interval: str = "1d",
    auto_adjust: bool = True,
    prepost: bool = False,
    threads: bool = True,
) -> pd.DataFrame:
    """Download historical price data for a ticker from Yahoo Finance.

    start and end accept strings like '2020-01-01'.
    Returns a pandas DataFrame indexed by Date with OHLCV (and other) columns.
    """
    if yf is None:
        raise ImportError("yfinance is required. Install with: pip install yfinance")

    ticker_obj = yf.Ticker(ticker)
    df = ticker_obj.history(
        start=start, end=end, interval=interval, auto_adjust=auto_adjust, prepost=prepost
    )

    if isinstance(df.index, pd.DatetimeIndex):
        # make timezone naive for consistency
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df.index.name = "Date"

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download time series from Yahoo Finance")
    parser.add_argument("ticker", help="Ticker symbol, e.g. AAPL")
    parser.add_argument("--start", help="Start date (YYYY-MM-DD)", default=None)
    parser.add_argument("--end", help="End date (YYYY-MM-DD)", default=None)
    parser.add_argument("--interval", help="Data interval (e.g. 1d, 1h)", default="1d")
    parser.add_argument("--out", help="Output CSV file (if omitted prints head)", default=None)
    args = parser.parse_args()

    df = download_timeseries(args.ticker, start=args.start, end=args.end, interval=args.interval)
    if args.out:
        df.to_csv(args.out)
        print(f"Saved {len(df)} rows to {args.out}")
    else:
        print(df.head())

