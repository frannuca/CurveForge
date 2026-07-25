"""Download short-term interest rate series per currency from FRED, for the FX carry feature.

Mirrors fx_downloader.py's shape: a downloader that fetches from an external
source and, on request, upserts into Postgres via the same (symbol, source,
field) schema `db.py` already uses for prices — here `symbol` is a currency
code (e.g. "EUR") rather than a pair, `source="fred"`, `field="rate"`.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pandas_datareader.data as web

logger = logging.getLogger(__name__)

# OECD "3-Month or 100-Day Rates and Yields: Interbank Rates" series on FRED —
# a consistently-named short-term policy-adjacent rate available for each of
# the 8 currencies behind MAJOR_FX_PAIRS. Monthly frequency.
RATE_SERIES: dict[str, str] = {
    "USD": "IR3TIB01USM156N",
    "EUR": "IR3TIB01EZM156N",
    "JPY": "IR3TIB01JPM156N",
    "GBP": "IR3TIB01GBM156N",
    "CHF": "IR3TIB01CHM156N",
    "AUD": "IR3TIB01AUM156N",
    "CAD": "IR3TIB01CAM156N",
    "NZD": "IR3TIB01NZM156N",
}


@dataclass
class RatesDownloader:
    """Downloads and caches each currency's short-term interest rate from FRED."""

    output_dir: Path = field(default_factory=lambda: Path("data/raw_rates"))
    years: int = 20

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _date_range(self) -> tuple[date, date]:
        end = date.today()
        start = end - timedelta(days=365 * self.years)
        return start, end

    def download_currency(self, currency: str, series_id: str | None = None) -> pd.Series:
        """Download the full monthly rate history for a single currency."""
        series_id = series_id or RATE_SERIES[currency]
        start, end = self._date_range()
        logger.info("Downloading %s rate (%s) from %s to %s", currency, series_id, start, end)

        df = web.DataReader(series_id, "fred", start, end)
        if df.empty:
            raise ValueError(f"No data returned for {currency} ({series_id})")

        series = df.iloc[:, 0].rename(currency)
        series.index.name = "date"
        return series

    def download_all(self, currencies: dict[str, str] | None = None) -> dict[str, pd.Series]:
        """Download rate history for every requested currency."""
        currencies = currencies or RATE_SERIES
        results: dict[str, pd.Series] = {}
        for currency, series_id in currencies.items():
            try:
                results[currency] = self.download_currency(currency, series_id)
            except Exception:
                logger.exception("Failed to download %s (%s)", currency, series_id)
        return results

    def save(self, currency: str, series: pd.Series) -> Path:
        path = self.output_dir / f"{currency}.csv"
        series.to_csv(path)
        logger.info("Saved %d rows to %s", len(series), path)
        return path

    def download_and_save_all(self, currencies: dict[str, str] | None = None) -> dict[str, pd.Series]:
        data = self.download_all(currencies)
        for currency, series in data.items():
            self.save(currency, series)
        return data


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Download short-term interest rates for major FX currencies.")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/raw_rates"), help="Directory to store CSV files."
    )
    parser.add_argument("--years", type=int, default=20, help="Years of history to fetch.")
    parser.add_argument(
        "--currencies",
        nargs="+",
        default=list(RATE_SERIES.keys()),
        help="Subset of currencies to download (default: all).",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upsert downloaded rates into Postgres (requires POSTGRES_* env vars).",
    )
    args = parser.parse_args()

    selected = {c: RATE_SERIES[c] for c in args.currencies}
    downloader = RatesDownloader(output_dir=args.output_dir, years=args.years)
    data = downloader.download_and_save_all(selected)

    if args.upload:
        from fx_forecasting.data.db import get_connection, upsert_series

        with get_connection() as conn:
            for currency, series in data.items():
                upsert_series(conn, currency, series, source="fred", field="rate")


if __name__ == "__main__":
    main()
