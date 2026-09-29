"""Recompute trade_features in plain SQL and compare with what `tt features` wrote.

An independent second implementation: set-based SQL instead of the Python sweep, and
holiday tables loaded from the reference files in tests/fixtures/calendars (generated
from exchange_calendars and python-holidays) instead of our calendar rules.

    DATABASE_URL=... python scripts/crosscheck_features.py [--as-of 2024-04-30]

Prints, per feature, how many trades were compared and how many disagree, with
examples. Exits non-zero on any disagreement.
"""

import argparse
import os
import sys
from datetime import date
from pathlib import Path

import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tradetracker.features.calendar import FEDERAL_EXTRA  # noqa: E402

REF = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "calendars"

SETUP = """
CREATE TEMP TABLE sec_closed (d date PRIMARY KEY);
CREATE TEMP TABLE nyse_closed (d date PRIMARY KEY);
CREATE TEMP TABLE f AS
SELECT x.trade_id, x.filing_date, x.lag_days, x.drift_pct, x.days_since_filing,
       x.filer_lag_zscore, x.cluster_count, x.size_score, x.size_basis, x.trade_value,
       x.duplicate_of_trade_id, t.ticker, t.side, t.trade_date, fl.filer_id, fl.filing_id,
       coalesce(fl.document_type, '') LIKE '%/A' AS is_amendment, true AS sampled
FROM latest_trade_features x JOIN trades t USING (trade_id) JOIN filings fl USING (filing_id);
CREATE INDEX ON f (ticker, side, trade_date);
CREATE INDEX ON f (filer_id, side, filing_date);
CREATE INDEX ON f (side, filing_date);
"""

CHECKS = {
    # Business days in (trade_date, filing_date], from the reference federal list.
    "lag_days": """
        SELECT trade_id, lag_days::numeric, (
            SELECT count(*) FROM generate_series(trade_date + 1, filing_date, '1 day') g(d)
            WHERE extract(isodow FROM g.d) < 6 AND g.d::date NOT IN (SELECT d FROM sec_closed)
        )::numeric
        FROM f WHERE lag_days IS NOT NULL AND sampled
    """,
    "days_since_filing": """
        SELECT trade_id, days_since_filing::numeric, greatest(0, (
            SELECT count(*) FROM generate_series(filing_date + 1, %(as_of)s::date, '1 day') g(d)
            WHERE extract(isodow FROM g.d) < 6 AND g.d::date NOT IN (SELECT d FROM nyse_closed)
        ))::numeric
        FROM f WHERE sampled
    """,
    "drift_pct": """
        SELECT f.trade_id, round(f.drift_pct::numeric, 6),
               round((CASE WHEN f.side = 'sell' THEN -1 ELSE 1 END)
                     * (p1.c / p0.c - 1) * 100, 6)
        FROM f
        CROSS JOIN LATERAL (
            SELECT coalesce(adj_close, close) c, date FROM daily_prices
            WHERE ticker = f.ticker AND date <= f.trade_date AND coalesce(adj_close, close) > 0
            ORDER BY date DESC LIMIT 1) p0
        CROSS JOIN LATERAL (
            SELECT coalesce(adj_close, close) c, date FROM daily_prices
            WHERE ticker = f.ticker AND date <= f.filing_date AND coalesce(adj_close, close) > 0
            ORDER BY date DESC LIMIT 1) p1
        WHERE f.lag_days IS NOT NULL AND f.sampled
          AND f.trade_date - p0.date <= 5 AND f.filing_date - p1.date <= 5
    """,
    # Distinct insiders, same ticker and side, trade dates within 14 days, already public.
    "cluster_count": """
        SELECT f.trade_id, f.cluster_count::numeric, (
            SELECT count(DISTINCT o.filer_id) FROM f o
            WHERE o.ticker = f.ticker AND o.side = f.side AND o.duplicate_of_trade_id IS NULL
              AND abs(o.trade_date - f.trade_date) <= 14 AND o.filing_date <= f.filing_date
        )::numeric
        FROM f WHERE f.ticker IS NOT NULL AND f.duplicate_of_trade_id IS NULL AND f.sampled
    """,
    # Mid-rank percentile among the filer's earlier same-side trades.
    "size_score (filer)": """
        SELECT f.trade_id, round(f.size_score::numeric, 9), round(h.score, 9)
        FROM f CROSS JOIN LATERAL (
            SELECT (count(*) FILTER (WHERE o.trade_value < f.trade_value)
                    + 0.5 * count(*) FILTER (WHERE o.trade_value = f.trade_value))
                   / count(*) AS score
            FROM f o
            WHERE o.filer_id = f.filer_id AND o.side = f.side AND o.trade_value IS NOT NULL
              AND o.duplicate_of_trade_id IS NULL AND o.filing_date < f.filing_date
        ) h
        WHERE f.size_basis = 'filer' AND f.sampled
    """,
    # Otherwise: mid-rank percentile among all same-side trades filed in the past year.
    "size_score (market)": """
        SELECT f.trade_id, round(f.size_score::numeric, 9), round(h.score, 9)
        FROM f CROSS JOIN LATERAL (
            SELECT (count(*) FILTER (WHERE o.trade_value < f.trade_value)
                    + 0.5 * count(*) FILTER (WHERE o.trade_value = f.trade_value))
                   / count(*) AS score
            FROM f o
            WHERE o.side = f.side AND o.trade_value IS NOT NULL
              AND o.duplicate_of_trade_id IS NULL AND o.filing_date < f.filing_date
              AND f.filing_date - o.filing_date <= 365
        ) h
        WHERE f.size_basis = 'market' AND f.sampled
    """,
    # Filer z-score: one lag per earlier non-amendment filing, population std floored at 0.5.
    "filer_lag_zscore": """
        WITH per_filing AS (
            SELECT filer_id, filing_id, filing_date, max(lag_days) AS lag FROM f
            WHERE NOT is_amendment AND duplicate_of_trade_id IS NULL AND lag_days IS NOT NULL
            GROUP BY 1, 2, 3
        )
        SELECT f.trade_id, round(f.filer_lag_zscore::numeric, 6),
               round((f.lag_days - h.mean) / greatest(h.sd, 0.5), 6)
        FROM f CROSS JOIN LATERAL (
            SELECT avg(lag) AS mean, stddev_pop(lag) AS sd, count(*) AS n FROM per_filing p
            WHERE p.filer_id = f.filer_id AND p.filing_date < f.filing_date
        ) h
        WHERE f.lag_days IS NOT NULL AND h.n >= 3 AND f.sampled
    """,
}

# The converse: the SQL says a value exists but Python left it NULL.
MISSING = {
    "filer_lag_zscore": """
        SELECT count(*) FROM f WHERE f.lag_days IS NOT NULL AND f.filer_lag_zscore IS NULL
          AND f.sampled
          AND (SELECT count(DISTINCT p.filing_id) FROM f p
               WHERE p.filer_id = f.filer_id AND p.filing_date < f.filing_date
                 AND NOT p.is_amendment AND p.duplicate_of_trade_id IS NULL
                 AND p.lag_days IS NOT NULL) >= 3
    """,
}


def check(conn: psycopg.Connection, as_of: str, limit: int | None = None) -> dict[str, list]:
    """Run every check; returns {check name: [(trade_id, stored, recomputed), ...]}."""
    conn.execute(SETUP)
    if limit:
        # Only the outer trades are sampled; history lookups still see every trade.
        conn.execute(
            "UPDATE f SET sampled = trade_id IN "
            "(SELECT trade_id FROM f ORDER BY md5(trade_id::text) LIMIT %s)",
            (limit,),
        )
    with conn.cursor() as cur:
        for name, table in (("us_federal_holidays.txt", "sec_closed"),
                            ("xnys_closures.txt", "nyse_closed")):
            days = {l.split("\t")[0] for l in (REF / name).read_text().splitlines()
                    if l and not l.startswith("#")}
            if table == "sec_closed":
                days |= {d.isoformat() for d in FEDERAL_EXTRA}
            cur.executemany(f"INSERT INTO {table} VALUES (%s)", [(d,) for d in sorted(days)])
    results = {}
    for name, sql in CHECKS.items():
        rows = conn.execute(sql, {"as_of": as_of}).fetchall()
        results[name] = (len(rows), [r for r in rows if r[1] != r[2]])
    for name, sql in MISSING.items():
        results[name + " missing"] = (None, [None] * conn.execute(sql).fetchone()[0])
    conn.rollback()  # drop the temp tables
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of", default=date.today().isoformat())
    ap.add_argument("--limit", type=int, help="check a random sample of this many trades")
    args = ap.parse_args()
    conn = psycopg.connect(os.environ["DATABASE_URL"])
    bad = 0
    for name, (n, wrong) in check(conn, args.as_of, args.limit).items():
        bad += len(wrong)
        compared = "" if n is None else f"compared {n:>8}  "
        print(f"{name:<28} {compared}mismatches {len(wrong):>6}  {wrong[:3]}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
