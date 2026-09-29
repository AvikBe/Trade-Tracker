"""The spec's validation rules. Failures go to the rejects table for review."""

import psycopg

RULES = {
    "trade_after_filing": """
        SELECT t.trade_id, 'trade ' || t.trade_date || ' after filing ' || f.filed_at::date
        FROM trades t JOIN filings f USING (filing_id)
        WHERE t.trade_date > (f.filed_at AT TIME ZONE 'America/New_York')::date
    """,
    "unknown_ticker": """
        SELECT t.trade_id, 'no ticker for issuer CIK ' || coalesce(f.issuer_cik, '?')
        FROM trades t JOIN filings f USING (filing_id)
        WHERE t.ticker IS NULL OR t.ticker IN ('NONE', 'N/A', 'NA')
    """,
    "price_out_of_range": """
        SELECT t.trade_id, 'price ' || t.price || ' vs range ' || p.low || '-' || p.high
        FROM trades t JOIN daily_prices p ON p.ticker = t.ticker AND p.date = t.trade_date
        WHERE t.price > 0 AND (t.price < p.low * 0.8 OR t.price > p.high * 1.2)
    """,
}


def run(conn: psycopg.Connection) -> dict[str, int]:
    counts = {}
    for rule, query in RULES.items():
        cur = conn.execute(
            f"""
            INSERT INTO rejects (trade_id, rule, detail)
            SELECT q.trade_id, %s, q.detail FROM ({query}) AS q(trade_id, detail)
            ON CONFLICT (trade_id, rule) DO NOTHING
            """,
            (rule,),
        )
        counts[rule] = cur.rowcount
    conn.commit()
    return counts
