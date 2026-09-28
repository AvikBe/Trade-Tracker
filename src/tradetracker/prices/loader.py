"""Backfill and update daily prices within Tiingo's free-tier quota."""

from datetime import date, timedelta

import httpx
import psycopg

from .. import store
from .tiingo import DAILY_LIMIT, HOURLY_LIMIT, TickerNotFound, TiingoClient

BENCHMARKS = [
    "SPY",
    # SPDR sector ETFs, for sector-relative excess returns.
    "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY",
]
HISTORY_START = date(2014, 1, 1)  # a year of lead-in before the 2015 backtest window


def remaining_quota(conn: psycopg.Connection) -> int:
    hourly = HOURLY_LIMIT - store.api_calls_since(conn, "tiingo", "1 hour")
    daily = DAILY_LIMIT - store.api_calls_since(conn, "tiingo", "1 day")
    return max(0, min(hourly, daily))


def tickers_to_load(conn: psycopg.Connection) -> list[tuple[str, date]]:
    """Tickers needing a call, with the date to start from. Backfills come first."""
    rows = conn.execute(
        """
        WITH wanted AS (
            SELECT DISTINCT ticker FROM trades WHERE ticker IS NOT NULL
            UNION SELECT unnest(%s::text[])
        ), latest AS (
            SELECT ticker, max(date) AS last_date FROM daily_prices GROUP BY ticker
        )
        SELECT w.ticker, l.last_date
        FROM wanted w
        LEFT JOIN latest l USING (ticker)
        LEFT JOIN price_status s USING (ticker)
        WHERE (s.status IS DISTINCT FROM 'not_found' OR s.last_attempt < now() - interval '30 days')
          AND (l.last_date IS NULL OR l.last_date < current_date - 1)
        ORDER BY (l.last_date IS NOT NULL), w.ticker
        """,
        (BENCHMARKS,),
    ).fetchall()
    return [(t, (last + timedelta(days=1)) if last else HISTORY_START) for t, last in rows]


def _status(conn, ticker: str, status: str, detail: str | None = None) -> None:
    conn.execute(
        """
        INSERT INTO price_status (ticker, status, detail) VALUES (%s, %s, %s)
        ON CONFLICT (ticker) DO UPDATE SET status = EXCLUDED.status,
          detail = EXCLUDED.detail, last_attempt = now()
        """,
        (ticker, status, detail),
    )


def run(conn: psycopg.Connection, client: TiingoClient, limit: int | None = None) -> dict:
    budget = remaining_quota(conn)
    if limit is not None:
        budget = min(budget, limit)
    todo = tickers_to_load(conn)
    stats = {"requested": 0, "bars": 0, "not_found": 0, "errors": 0, "pending": len(todo)}
    for ticker, start in todo[:budget]:
        store.log_api_call(conn, "tiingo", ticker)
        stats["requested"] += 1
        try:
            bars = client.daily(ticker, start)
        except TickerNotFound:
            _status(conn, ticker, "not_found")
            stats["not_found"] += 1
        except httpx.HTTPError as e:
            _status(conn, ticker, "error", str(e)[:500])
            stats["errors"] += 1
        else:
            stats["bars"] += store.save_prices(conn, bars)
            _status(conn, ticker, "ok")
        conn.commit()
    stats["pending"] -= stats["requested"]
    return stats
