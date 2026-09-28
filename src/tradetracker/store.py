"""Idempotent writes of parsed filings and price bars.

Filings are keyed on (source, source_filing_id) and trades on (filing_id, line_no),
so re-running any loader is safe. Existing rows are never overwritten: first_seen_at
and filed_at stay as first recorded, which keeps backtests point in time.
"""

from collections.abc import Iterable

import psycopg
from psycopg.types.json import Jsonb

from .models import ParsedFiling
from .prices.tiingo import PriceBar


def _filer_id(conn: psycopg.Connection, f: ParsedFiling) -> int:
    row = conn.execute(
        """
        INSERT INTO filers (source, source_key, name, kind)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (source, source_key) DO UPDATE SET name = filers.name
        RETURNING filer_id
        """,
        (f.source, f.filer_key, f.filer_name, f.filer_kind),
    ).fetchone()
    return row[0]


def _original_filing(conn: psycopg.Connection, filer_id: int, f: ParsedFiling) -> int | None:
    """An amendment points at the earliest filing by the same owner, issuer and period."""
    if not f.document_type.endswith("/A"):
        return None
    row = conn.execute(
        """
        SELECT filing_id FROM filings
        WHERE filer_id = %s AND issuer_cik IS NOT DISTINCT FROM %s
          AND period_of_report IS NOT DISTINCT FROM %s
          AND document_type NOT LIKE '%%/A' AND filed_at <= %s
        ORDER BY filed_at, filing_id LIMIT 1
        """,
        (filer_id, f.issuer_cik, f.period_of_report, f.filed_at),
    ).fetchone()
    return row[0] if row else None


def save_filing(conn: psycopg.Connection, f: ParsedFiling) -> tuple[int, bool]:
    """Insert a filing and its trades. Returns (filing_id, created)."""
    filer_id = _filer_id(conn, f)
    row = conn.execute(
        """
        INSERT INTO filings (filer_id, source, source_filing_id, source_url, document_type,
                             issuer_cik, issuer_name, issuer_ticker, period_of_report,
                             filed_at, accepted_at, amends_filing_id, raw)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (source, source_filing_id) DO NOTHING
        RETURNING filing_id
        """,
        (
            filer_id, f.source, f.source_filing_id, f.source_url, f.document_type,
            f.issuer_cik, f.issuer_name, f.issuer_ticker, f.period_of_report,
            f.filed_at, f.accepted_at, _original_filing(conn, filer_id, f),
            Jsonb(f.raw) if f.raw is not None else None,
        ),
    ).fetchone()
    if row is None:
        # Already ingested. Fill accepted_at if the live feed now supplies it.
        existing = conn.execute(
            """
            UPDATE filings SET accepted_at = COALESCE(accepted_at, %s)
            WHERE source = %s AND source_filing_id = %s RETURNING filing_id
            """,
            (f.accepted_at, f.source, f.source_filing_id),
        ).fetchone()
        return existing[0], False

    filing_id = row[0]
    for t in f.trades:
        conn.execute(
            """
            INSERT INTO trades (filing_id, line_no, ticker, asset_type, side, txn_code,
                                is_10b5_1, trade_date, amount_low, amount_high,
                                shares, price, owner, role)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (filing_id, line_no) DO NOTHING
            """,
            (
                filing_id, t.line_no, f.issuer_ticker, t.asset_type, t.side, t.txn_code,
                t.is_10b5_1, t.trade_date, t.amount_low, t.amount_high,
                t.shares, t.price, t.owner, f.role,
            ),
        )
    if f.issuer_cik and f.issuer_ticker:
        record_ticker(conn, f.issuer_cik, f.issuer_ticker, f.filed_at.date(), "filing")
    return filing_id, True


def link_amendments(conn: psycopg.Connection) -> int:
    """Link amendments whose original arrived after them. Returns how many were linked.

    save_filing links at insert time, but within a quarter's data set an amendment
    can come before its original, and quarters can be loaded in any order.
    Amendments of filings older than the loaded history stay unlinked.
    """
    cur = conn.execute(
        """
        WITH originals AS (
            SELECT a.filing_id, (
                SELECT o.filing_id FROM filings o
                WHERE o.filer_id = a.filer_id
                  AND o.issuer_cik IS NOT DISTINCT FROM a.issuer_cik
                  AND o.period_of_report IS NOT DISTINCT FROM a.period_of_report
                  AND o.document_type NOT LIKE '%/A' AND o.filed_at <= a.filed_at
                ORDER BY o.filed_at, o.filing_id LIMIT 1
            ) AS original_id
            FROM filings a
            WHERE a.document_type LIKE '%/A' AND a.amends_filing_id IS NULL
        )
        UPDATE filings a SET amends_filing_id = originals.original_id
        FROM originals
        WHERE a.filing_id = originals.filing_id AND originals.original_id IS NOT NULL
        """
    )
    return cur.rowcount


def record_ticker(conn: psycopg.Connection, cik: str, ticker: str, seen, source: str) -> None:
    conn.execute(
        """
        INSERT INTO ticker_history (cik, ticker, first_seen, last_seen, source)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (cik, ticker, source) DO UPDATE
          SET first_seen = LEAST(ticker_history.first_seen, EXCLUDED.first_seen),
              last_seen  = GREATEST(ticker_history.last_seen, EXCLUDED.last_seen)
        """,
        (cik, ticker, seen, seen, source),
    )


def fill_missing_tickers(conn: psycopg.Connection, current: dict[str, str]) -> int:
    """Map trades whose filing had no symbol using today's SEC mapping.

    Only used as a fallback: a current mapping can differ from the one in force on
    the filing date, so these trades are marked via ticker_history source.
    """
    rows = conn.execute(
        """
        SELECT DISTINCT f.issuer_cik FROM trades t JOIN filings f USING (filing_id)
        WHERE t.ticker IS NULL AND f.issuer_cik IS NOT NULL
        """
    ).fetchall()
    n = 0
    for (cik,) in rows:
        ticker = current.get(cik)
        if not ticker:
            continue
        cur = conn.execute(
            """
            UPDATE trades t SET ticker = %s FROM filings f
            WHERE t.filing_id = f.filing_id AND f.issuer_cik = %s AND t.ticker IS NULL
            """,
            (ticker, cik),
        )
        n += cur.rowcount
    return n


def save_prices(conn: psycopg.Connection, bars: Iterable[PriceBar], source: str = "tiingo") -> int:
    rows = [
        (b.ticker, b.date, b.open, b.high, b.low, b.close, b.adj_open, b.adj_close, b.volume, source)
        for b in bars
    ]
    if not rows:
        return 0
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO daily_prices (ticker, date, open, high, low, close,
                                      adj_open, adj_close, volume, source)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (ticker, date) DO UPDATE SET
              open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low,
              close = EXCLUDED.close, adj_open = EXCLUDED.adj_open,
              adj_close = EXCLUDED.adj_close, volume = EXCLUDED.volume
            """,
            rows,
        )
    return len(rows)


def log_api_call(conn: psycopg.Connection, vendor: str, endpoint: str) -> None:
    conn.execute("INSERT INTO api_calls (vendor, endpoint) VALUES (%s, %s)", (vendor, endpoint))


def api_calls_since(conn: psycopg.Connection, vendor: str, interval: str) -> int:
    return conn.execute(
        "SELECT count(*) FROM api_calls WHERE vendor = %s AND called_at > now() - %s::interval",
        (vendor, interval),
    ).fetchone()[0]
