"""Milestone 1 exit check: coverage per source and year, plus a lag histogram.

Mapped share under ~95% for Form 4 means ticker mapping needs work before
milestone 2 (features) starts.
"""

from datetime import date, timedelta

import psycopg

LAG_BUCKETS = [(0, 0), (1, 1), (2, 2), (3, 5), (6, 10), (11, 30), (31, 45), (46, 10_000)]


def business_days(start: date, end: date) -> int:
    """Weekdays from start (exclusive) to end (inclusive). Holidays ignored for now."""
    if end <= start:
        return 0
    days = (end - start).days
    weeks, rem = divmod(days, 7)
    count = weeks * 5
    for i in range(1, rem + 1):
        if (start + timedelta(days=weeks * 7 + i)).weekday() < 5:
            count += 1
    return count


def coverage(conn: psycopg.Connection) -> list[tuple]:
    return conn.execute(
        """
        SELECT f.source,
               extract(year FROM f.filed_at)::int AS year,
               count(*) AS trades,
               round(100.0 * count(t.ticker) / count(*), 1) AS mapped_pct,
               round(100.0 * count(*) FILTER (WHERE EXISTS (
                   SELECT 1 FROM daily_prices p WHERE p.ticker = t.ticker
                   AND p.date BETWEEN t.trade_date AND t.trade_date + 7)) / count(*), 1) AS priced_pct,
               count(*) FILTER (WHERE t.side = 'buy') AS buys,
               count(*) FILTER (WHERE t.is_10b5_1) AS plan_trades
        FROM trades t JOIN filings f USING (filing_id)
        GROUP BY 1, 2 ORDER BY 1, 2
        """
    ).fetchall()


def _label(lo: int, hi: int) -> str:
    if lo == hi:
        return str(lo)
    return f"{lo}+" if hi >= 10_000 else f"{lo}-{hi}"


def lag_histogram(conn: psycopg.Connection, source: str = "edgar") -> dict[str, int]:
    rows = conn.execute(
        """
        SELECT t.trade_date, (f.filed_at AT TIME ZONE 'America/New_York')::date
        FROM trades t JOIN filings f USING (filing_id)
        WHERE f.source = %s AND f.document_type NOT LIKE '%%/A'  -- amendments repeat old trades
        """,
        (source,),
    ).fetchall()
    hist = {_label(lo, hi): 0 for lo, hi in LAG_BUCKETS}
    for trade_date, filed in rows:
        lag = business_days(trade_date, filed)
        for lo, hi in LAG_BUCKETS:
            if lo <= lag <= hi:
                hist[_label(lo, hi)] += 1
                break
    return hist


def render(conn: psycopg.Connection) -> str:
    lines = ["source  year  trades  mapped%  priced%  buys  10b5-1"]
    for src, year, n, mapped, priced, buys, plans in coverage(conn):
        lines.append(f"{src:<7} {year:<5} {n:>6}  {mapped:>7}  {priced:>7}  {buys:>4}  {plans:>6}")
    lines.append("")
    lines.append("Form 4 lag (business days, trade to filing)")
    hist = lag_histogram(conn)
    total = sum(hist.values()) or 1
    for bucket, n in hist.items():
        lines.append(f"  {bucket:>7}: {n:>8}  {'#' * round(40 * n / total)}")
    return "\n".join(lines)
