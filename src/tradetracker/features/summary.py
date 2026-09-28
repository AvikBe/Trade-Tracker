"""`tt feature-report`: what the feature job produced, for eyeballing real data."""

import psycopg

LAG_RATIO_BUCKETS = """
    CASE WHEN x.lag_ratio IS NULL THEN 'n/a'
         WHEN x.lag_ratio <= 1 THEN 'on time (<=1x)'
         WHEN x.lag_ratio <= 2 THEN 'late (1-2x)'
         ELSE 'very late (>2x)' END
"""

BASE = """
    FROM latest_trade_features x JOIN trades t USING (trade_id)
    WHERE x.duplicate_of_trade_id IS NULL
"""


def _table(title: str, header: list[str], rows) -> list[str]:
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) if rows else len(str(h))
              for i, h in enumerate(header)]
    out = [title, "  " + "  ".join(str(h).rjust(w) for h, w in zip(header, widths))]
    out += ["  " + "  ".join(str(v).rjust(w) for v, w in zip(r, widths)) for r in rows]
    return out + [""]


def render(conn: psycopg.Connection) -> str:
    lines: list[str] = []
    lines += _table(
        "Coverage (share of trades with a value, amendment repeats excluded)",
        ["side", "trades", "lag%", "drift%", "zscore%", "size%", "cluster%", "role%"],
        conn.execute(f"""
            SELECT t.side, count(*),
                   round(100.0 * count(x.lag_days) / count(*), 1),
                   round(100.0 * count(x.drift_pct) / count(*), 1),
                   round(100.0 * count(x.filer_lag_zscore) / count(*), 1),
                   round(100.0 * count(x.size_score) / count(*), 1),
                   round(100.0 * count(x.cluster_count) / count(*), 1),
                   round(100.0 * count(x.role_weight) / count(*), 1)
            {BASE} GROUP BY 1 ORDER BY 1""").fetchall(),
    )
    lines += _table(
        "Lag ratio buckets (Form 4 deadline = 2 SEC business days)",
        ["bucket", "buys", "sells", "median drift buys", "median drift sells"],
        conn.execute(f"""
            SELECT {LAG_RATIO_BUCKETS} AS b,
                   count(*) FILTER (WHERE t.side = 'buy'),
                   count(*) FILTER (WHERE t.side = 'sell'),
                   round(percentile_cont(0.5) WITHIN GROUP (ORDER BY x.drift_pct)
                         FILTER (WHERE t.side = 'buy')::numeric, 2),
                   round(percentile_cont(0.5) WITHIN GROUP (ORDER BY x.drift_pct)
                         FILTER (WHERE t.side = 'sell')::numeric, 2)
            {BASE} GROUP BY 1 ORDER BY 1""").fetchall(),
    )
    lines += _table(
        "Drift quintiles (percent, positive = moved the insider's way before filing)",
        ["side", "p5", "p20", "p40", "p60", "p80", "p95", "over 20%"],
        conn.execute(f"""
            SELECT t.side,
                   round(percentile_cont(0.05) WITHIN GROUP (ORDER BY x.drift_pct)::numeric, 2),
                   round(percentile_cont(0.2) WITHIN GROUP (ORDER BY x.drift_pct)::numeric, 2),
                   round(percentile_cont(0.4) WITHIN GROUP (ORDER BY x.drift_pct)::numeric, 2),
                   round(percentile_cont(0.6) WITHIN GROUP (ORDER BY x.drift_pct)::numeric, 2),
                   round(percentile_cont(0.8) WITHIN GROUP (ORDER BY x.drift_pct)::numeric, 2),
                   round(percentile_cont(0.95) WITHIN GROUP (ORDER BY x.drift_pct)::numeric, 2),
                   count(*) FILTER (WHERE x.drift_pct > 20)
            {BASE} AND x.drift_pct IS NOT NULL GROUP BY 1 ORDER BY 1""").fetchall(),
    )
    lines += _table(
        "Role mix",
        ["role", "weight", "buys", "sells"],
        conn.execute(f"""
            SELECT coalesce(t.role, 'unknown'), x.role_weight,
                   count(*) FILTER (WHERE t.side = 'buy'), count(*) FILTER (WHERE t.side = 'sell')
            {BASE} GROUP BY 1, 2 ORDER BY 2 DESC NULLS LAST""").fetchall(),
    )
    lines += _table(
        "Cluster size (distinct insiders, same side and ticker, within 14 days)",
        ["insiders", "buys", "sells"],
        conn.execute(f"""
            SELECT CASE WHEN x.cluster_count >= 5 THEN '5+' ELSE x.cluster_count::text END,
                   count(*) FILTER (WHERE t.side = 'buy'), count(*) FILTER (WHERE t.side = 'sell')
            {BASE} AND x.cluster_count IS NOT NULL GROUP BY 1 ORDER BY 1""").fetchall(),
    )
    lines += _table(
        "Filer lag z-score (filers with 3+ earlier filings)",
        ["bucket", "trades"],
        conn.execute(f"""
            SELECT CASE WHEN x.filer_lag_zscore < -1 THEN 'a: faster than usual (<-1)'
                        WHEN x.filer_lag_zscore <= 1 THEN 'b: usual (-1..1)'
                        WHEN x.filer_lag_zscore <= 3 THEN 'c: later (1..3)'
                        ELSE 'd: much later (>3)' END,
                   count(*)
            {BASE} AND x.filer_lag_zscore IS NOT NULL GROUP BY 1 ORDER BY 1""").fetchall(),
    )
    lines += _table(
        "Size score basis",
        ["basis", "trades", "median score"],
        conn.execute(f"""
            SELECT coalesce(x.size_basis, 'none'), count(*),
                   round(percentile_cont(0.5) WITHIN GROUP (ORDER BY x.size_score)::numeric, 3)
            {BASE} GROUP BY 1 ORDER BY 1""").fetchall(),
    )
    lines += _table(
        "Flags",
        ["flag", "trades"],
        conn.execute("""
            SELECT f, count(*) FROM latest_trade_features, unnest(flags) f
            GROUP BY 1 ORDER BY 2 DESC""").fetchall(),
    )
    return "\n".join(lines)
