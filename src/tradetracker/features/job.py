"""Load trades and prices from Postgres, compute features, write trade_features."""

from collections import Counter
from datetime import date, datetime

import psycopg

from .compute import FEATURE_VERSION, EASTERN, FeatureRow, PriceSeries, TradeInput, compute

TRADES_SQL = """
    SELECT t.trade_id, t.filing_id, f.filer_id, f.source, coalesce(f.document_type, ''),
           f.amends_filing_id, f.filed_at, t.ticker, t.side, t.trade_date,
           t.shares, t.price, t.amount_low, t.amount_high, t.role, t.is_10b5_1
    FROM trades t JOIN filings f USING (filing_id)
"""


def load_trades(conn: psycopg.Connection) -> list[TradeInput]:
    return [TradeInput(*row) for row in conn.execute(TRADES_SQL)]


def price_lookup(conn: psycopg.Connection):
    def lookup(ticker: str) -> PriceSeries | None:
        rows = conn.execute(
            "SELECT date, coalesce(adj_close, close) FROM daily_prices WHERE ticker = %s",
            (ticker,),
        ).fetchall()
        return PriceSeries(rows) if rows else None

    return lookup


def _missing(conn: psycopg.Connection) -> set[int]:
    return {
        r[0]
        for r in conn.execute(
            """
            SELECT t.trade_id FROM trades t
            WHERE NOT EXISTS (SELECT 1 FROM trade_features x
                              WHERE x.trade_id = t.trade_id AND x.feature_version = %s)
            """,
            (FEATURE_VERSION,),
        )
    }


def save(conn: psycopg.Connection, rows: list[FeatureRow], computed_at: datetime) -> int:
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO trade_features (
                trade_id, computed_at, feature_version, filing_date, lag_days, lag_ratio,
                drift_pct, days_since_filing, filer_lag_zscore, cluster_count, role_weight,
                size_score, size_basis, trade_value, committee_match,
                duplicate_of_trade_id, flags)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (trade_id, computed_at) DO NOTHING
            """,
            [
                (
                    r.trade_id, computed_at, FEATURE_VERSION, r.filing_date, r.lag_days,
                    r.lag_ratio, r.drift_pct, r.days_since_filing, r.filer_lag_zscore,
                    r.cluster_count, r.role_weight, r.size_score, r.size_basis,
                    r.trade_value, r.committee_match, r.duplicate_of_trade_id, r.flags,
                )
                for r in rows
            ],
        )
    return len(rows)


def run(conn: psycopg.Connection, *, recompute: bool = False, as_of: date | None = None) -> dict:
    """Compute features for trades that lack them (or all trades with recompute).

    History features need every earlier trade, so all trades are loaded either way;
    only the write is limited.
    """
    computed_at = datetime.now(EASTERN)
    as_of = as_of or computed_at.date()
    trades = load_trades(conn)
    targets = None if recompute else _missing(conn)
    rows = compute(trades, price_lookup(conn), as_of)
    if targets is not None:
        rows = [r for r in rows if r.trade_id in targets]
    written = save(conn, rows, computed_at)
    conn.commit()
    flags = Counter(f for r in rows for f in r.flags)
    return {"trades": len(trades), "written": written, **dict(sorted(flags.items()))}
