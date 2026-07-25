"""Postgres persistence for FX close prices into the quote_keys / market_data tables."""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from datetime import date, timezone

import pandas as pd
import psycopg
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_SOURCE = "yahoo"
DEFAULT_FIELD = "close"
DEFAULT_METRICS = ("open", "high", "low", "volume")


def get_connection() -> psycopg.Connection:
    return psycopg.connect(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=os.environ.get("POSTGRES_PORT", "5432"),
        dbname=os.environ["POSTGRES_DB"],
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
    )


def get_or_create_quote_id(conn: psycopg.Connection, symbol: str, source: str, field: str) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO quant.quote_keys (symbol, source, field)
            VALUES (%s, %s, %s)
            ON CONFLICT (symbol, source, field) DO NOTHING
            RETURNING quote_id
            """,
            (symbol, source, field),
        )
        row = cur.fetchone()
        if row is None:
            cur.execute(
                "SELECT quote_id FROM quant.quote_keys WHERE symbol = %s AND source = %s AND field = %s",
                (symbol, source, field),
            )
            row = cur.fetchone()
        return row[0]


def upsert_series(
    conn: psycopg.Connection,
    symbol: str,
    series: pd.Series,
    source: str = DEFAULT_SOURCE,
    field: str = DEFAULT_FIELD,
) -> int:
    """Delete-then-insert a date-indexed value series for one symbol/field into market_data."""
    series = series.dropna()
    if series.empty:
        return 0

    quote_id = get_or_create_quote_id(conn, symbol, source, field)

    rows = [
        (quote_id, idx.to_pydatetime().replace(tzinfo=timezone.utc), True, False, float(value))
        for idx, value in series.items()
    ]
    start = min(r[1] for r in rows)
    end = max(r[1] for r in rows)

    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM quant.market_data WHERE quote_id = %s AND as_of BETWEEN %s AND %s",
            (quote_id, start, end),
        )
        cur.executemany(
            """
            INSERT INTO quant.market_data (quote_id, as_of, is_closed, is_settlement, value)
            VALUES (%s, %s, %s, %s, %s)
            """,
            rows,
        )
    conn.commit()
    logger.info("Upserted %d rows for %s|%s|%s", len(rows), symbol, source, field)
    return len(rows)


def upsert_metric_series(
    conn: psycopg.Connection, quote_id: int, metric_name: str, series: pd.Series
) -> int:
    """Delete-then-insert a date-indexed metric series (e.g. high/low/volume) into quant.metrics."""
    series = series.dropna()
    if series.empty:
        return 0

    rows = [
        (quote_id, idx.to_pydatetime().replace(tzinfo=timezone.utc), metric_name, float(value))
        for idx, value in series.items()
    ]
    start = min(r[1] for r in rows)
    end = max(r[1] for r in rows)

    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM quant.metrics WHERE quote_id = %s AND metric_name = %s AND as_of BETWEEN %s AND %s",
            (quote_id, metric_name, start, end),
        )
        cur.executemany(
            """
            INSERT INTO quant.metrics (quote_id, as_of, metric_name, value)
            VALUES (%s, %s, %s, %s)
            """,
            rows,
        )
    conn.commit()
    logger.info("Upserted %d %s rows for quote_id=%s", len(rows), metric_name, quote_id)
    return len(rows)


def upsert_metrics(
    conn: psycopg.Connection,
    symbol: str,
    df: pd.DataFrame,
    source: str = DEFAULT_SOURCE,
    field: str = DEFAULT_FIELD,
    metrics: Sequence[str] = DEFAULT_METRICS,
) -> None:
    """Upsert OHLCV metrics (open/high/low/volume) for `symbol` into quant.metrics.

    Shares the same quote_id as the canonical `field` (close) series in
    quote_keys; individual metrics are distinguished by `metric_name`.
    """
    quote_id = get_or_create_quote_id(conn, symbol, source, field)
    for metric_name in metrics:
        if metric_name not in df.columns:
            continue
        upsert_metric_series(conn, quote_id, metric_name, df[metric_name])


def upsert_pairs(
    data: dict[str, pd.DataFrame],
    source: str = DEFAULT_SOURCE,
    field: str = DEFAULT_FIELD,
    metrics: Sequence[str] = DEFAULT_METRICS,
) -> None:
    """Upsert the close-price series (market_data) and OHLCV metrics for each downloaded pair."""
    with get_connection() as conn:
        for symbol, df in data.items():
            upsert_series(conn, symbol, df[field], source=source, field=field)
            upsert_metrics(conn, symbol, df, source=source, field=field, metrics=metrics)


def _pivot_long_to_wide(rows: list[tuple], symbols: list[str]) -> pd.DataFrame:
    """Reshape (symbol, date, value) rows into a wide DataFrame: date index, one column per symbol."""
    long_df = pd.DataFrame(rows, columns=["symbol", "date", "value"])
    if long_df.empty:
        return pd.DataFrame(columns=symbols)

    wide = long_df.pivot(index="date", columns="symbol", values="value")
    wide.index = pd.to_datetime(wide.index)
    wide.columns.name = None
    return wide.reindex(columns=symbols)


def get_time_series(
    symbols: Sequence[str],
    start: date | str,
    end: date | str,
    source: str = DEFAULT_SOURCE,
    field: str = DEFAULT_FIELD,
) -> pd.DataFrame:
    """Fetch stored values as a wide DataFrame: date index, one column per symbol."""
    symbols = list(symbols)
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT qk.symbol, md.as_of_date, md.value
            FROM quant.market_data md
            JOIN quant.quote_keys qk ON qk.quote_id = md.quote_id
            WHERE qk.symbol = ANY(%s)
              AND qk.source = %s
              AND qk.field = %s
              AND md.as_of_date BETWEEN %s AND %s
            ORDER BY md.as_of_date
            """,
            (symbols, source, field, start, end),
        )
        rows = cur.fetchall()

    return _pivot_long_to_wide(rows, symbols)


def get_metric_series(
    symbols: Sequence[str],
    metric_name: str,
    start: date | str,
    end: date | str,
    source: str = DEFAULT_SOURCE,
    field: str = DEFAULT_FIELD,
) -> pd.DataFrame:
    """Fetch a stored OHLCV metric (e.g. 'high', 'low', 'volume') as a wide DataFrame.

    `field` selects the quote_keys row (and therefore quote_id) each symbol's
    metrics are attached to; this must match what `upsert_metrics` used (the
    canonical `close` field by default).
    """
    symbols = list(symbols)
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT qk.symbol, m.as_of_date, m.value
            FROM quant.metrics m
            JOIN quant.quote_keys qk ON qk.quote_id = m.quote_id
            WHERE qk.symbol = ANY(%s)
              AND qk.source = %s
              AND qk.field = %s
              AND m.metric_name = %s
              AND m.as_of_date BETWEEN %s AND %s
            ORDER BY m.as_of_date
            """,
            (symbols, source, field, metric_name, start, end),
        )
        rows = cur.fetchall()

    return _pivot_long_to_wide(rows, symbols)
