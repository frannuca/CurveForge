"""Download historical daily time series for major FX pairs via Yahoo Finance."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# The seven "major" FX pairs, all quoted against USD.
MAJOR_FX_PAIRS: dict[str, str] = {
    "EURUSD": "EURUSD=X",
    "USDJPY": "USDJPY=X",
    "GBPUSD": "GBPUSD=X",
    "USDCHF": "USDCHF=X",
    "AUDUSD": "AUDUSD=X",
    "USDCAD": "USDCAD=X",
    "NZDUSD": "NZDUSD=X",
}

SINGLE_FX_PAIRS: dict[str, str] = {
    "EURUSD": "EURUSD=X"
}

@dataclass
class FXDownloader:
    """Downloads and caches daily OHLC time series for FX pairs from Yahoo Finance."""

    output_dir: Path = field(default_factory=lambda: Path("data/raw"))
    years: int = 20

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _date_range(self) -> tuple[date, date]:
        end = date.today()
        start = end - timedelta(days=365 * self.years)
        return start, end

    def download_pair(self, pair: str, ticker: str | None = None) -> pd.DataFrame:
        """Download the full daily history for a single FX pair."""
        ticker = ticker or MAJOR_FX_PAIRS[pair]
        start, end = self._date_range()
        logger.info("Downloading %s (%s) from %s to %s", pair, ticker, start, end)

        df = yf.download(
            ticker,
            start=start.isoformat(),
            end=end.isoformat(),
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
        if df.empty:
            raise ValueError(f"No data returned for {pair} ({ticker})")

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.index.name = "date"
        df.columns.name = None
        df = df.rename(columns=str.lower)
        return df[["open", "high", "low", "close", "volume"]]

    def download_all(
        self, pairs: dict[str, str] | None = None
    ) -> dict[str, pd.DataFrame]:
        """Download daily history for every requested FX pair."""
        pairs = pairs or MAJOR_FX_PAIRS
        results: dict[str, pd.DataFrame] = {}
        for pair, ticker in pairs.items():
            try:
                results[pair] = self.download_pair(pair, ticker)
            except Exception:
                logger.exception("Failed to download %s (%s)", pair, ticker)
        return results

    def save(self, pair: str, df: pd.DataFrame) -> Path:
        path = self.output_dir / f"{pair}.csv"
        df.to_csv(path)
        logger.info("Saved %s rows to %s", len(df), path)
        return path

    def load(self, pair: str) -> pd.DataFrame:
        path = self.output_dir / f"{pair}.csv"
        return pd.read_csv(path, index_col="date", parse_dates=True)

    def download_and_save_all(
        self, pairs: dict[str, str] | None = None
    ) -> dict[str, pd.DataFrame]:
        data = self.download_all(pairs)
        for pair, df in data.items():
            self.save(pair, df)
        return data


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Download major FX pair time series.")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/raw"), help="Directory to store CSV files."
    )
    parser.add_argument("--years", type=int, default=20, help="Years of history to fetch.")
    parser.add_argument(
        "--pairs",
        nargs="+",
        default=list(MAJOR_FX_PAIRS.keys()),
        help="Subset of pairs to download (default: all majors).",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upsert downloaded close prices into Postgres (requires DATABASE_URL).",
    )
    args = parser.parse_args()

    selected = {p: MAJOR_FX_PAIRS[p] for p in args.pairs}
    downloader = FXDownloader(output_dir=args.output_dir, years=args.years)
    data = downloader.download_and_save_all(selected)

    if args.upload:
        from fx_forecasting.data.db import upsert_pairs

        upsert_pairs(data)


if __name__ == "__main__":
    main()
