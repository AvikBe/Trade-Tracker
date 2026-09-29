"""Recompute a backtest run independently and report mismatches.

    python scripts/crosscheck_backtest.py --run results/ --prices DIR [--db URL] [--sample N]

Independent of `tradetracker.backtest` on purpose: trading sessions come from
`exchange_calendars` (XNYS), bars are read with pandas straight from the CSV files,
events are regrouped in SQL, and statistics are recomputed with numpy/scipy.
Needs `pip install pandas exchange_calendars scipy psycopg`.

Checks:
1. Events: the SQL regrouping of trades matches events.csv.gz (count, lag, lines, value).
2. Entry and exit days match XNYS sessions.
3. Stock, SPY, sector and IWM returns and drift match the raw bars.
4. Bucket means and t-stats (plain and month-clustered) match the report's formulas.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from pathlib import Path

import exchange_calendars as xcals
import pandas as pd
from scipy import stats as sps

HORIZONS = (5, 20, 60)
TOL = 1e-6


def bars(root: Path, ticker: str, cache: dict) -> pd.DataFrame | None:
    if ticker not in cache:
        d = root / "bars" if (root / "bars").is_dir() else root
        path = d / (re.sub(r"[^A-Za-z0-9.\-]", "_", ticker) + ".csv.gz")
        if not path.exists():
            cache[ticker] = None
        else:
            df = pd.read_csv(path, parse_dates=["date"])
            df["date"] = df["date"].dt.date
            df = df[df["adjClose"] > 0].drop_duplicates("date", keep="last").set_index("date").sort_index()
            if df["adjOpen"].isna().all():
                df["adjOpen"] = df["open"] * df["adjClose"] / df["close"]
            cache[ticker] = df
    return cache[ticker]


def check_events_sql(run: pd.DataFrame, url: str) -> list[str]:
    import psycopg

    sql = """
        SELECT f.filer_id, t.ticker, t.side, x.filing_date, max(x.lag_days) AS lag_days,
               count(*) AS n_lines,
               coalesce(sum(t.shares * t.price) FILTER (WHERE t.shares > 0 AND t.price > 0), 0) AS value
        FROM trades t JOIN filings f USING (filing_id)
        JOIN (SELECT DISTINCT ON (trade_id) * FROM trade_features
              ORDER BY trade_id, computed_at DESC) x USING (trade_id)
        WHERE f.source = 'edgar' AND x.duplicate_of_trade_id IS NULL
          AND coalesce(f.document_type, '') NOT LIKE '%%/A'
          AND x.lag_days IS NOT NULL AND t.ticker IS NOT NULL AND t.ticker <> ''
        GROUP BY 1, 2, 3, 4
    """
    with psycopg.connect(url) as conn:
        db = pd.DataFrame(conn.execute(sql).fetchall(),
                          columns=["filer_id", "ticker", "side", "filing_date", "lag_days", "n_lines", "value"])
    db["filing_date"] = db["filing_date"].astype(str)
    db["value"] = db["value"].astype(float)
    key = ["filer_id", "ticker", "side", "filing_date"]
    m = run.merge(db, on=key, how="outer", suffixes=("", "_sql"), indicator=True)
    out = [f"events: run {len(run)}, SQL {len(db)}, only in run {(m._merge == 'left_only').sum()}, "
           f"only in SQL {(m._merge == 'right_only').sum()}"]
    both = m[m._merge == "both"]
    for col in ("lag_days", "n_lines"):
        bad = (both[col] != both[col + "_sql"]).sum()
        out.append(f"  {col}: {bad} mismatches")
    bad = (abs(both["value"] - both["value_sql"]) > 0.01 + 1e-6 * both["value_sql"].abs()).sum()
    out.append(f"  value: {bad} mismatches")
    return out


def check_returns(run: pd.DataFrame, root: Path, sample: int, seed: int) -> list[str]:
    cal = xcals.get_calendar("XNYS", start="2013-01-01")
    sessions = [d.date() for d in cal.sessions]
    pos = {d: i for i, d in enumerate(sessions)}
    import bisect

    def next_session(d):
        return sessions[bisect.bisect_right(sessions, d)]

    def add(d, n):
        return sessions[pos[d] + n]

    cache: dict = {}
    spy = bars(root, "SPY", cache)
    last = spy.index[-1]
    ok = run[run.status == "ok"]
    rows = ok.sample(min(sample, len(ok)), random_state=seed) if sample else ok
    counts = {k: 0 for k in ("checked", "entry_day", "drift", "stock", "spy", "sector", "iwm", "exit_day")}
    worst = {}

    def cmp(name, a, b, ident):
        if (a is None or (isinstance(a, float) and math.isnan(a))) and (
                b is None or (isinstance(b, float) and math.isnan(b))):
            return
        if a is None or b is None or (isinstance(a, float) and math.isnan(a)) or (
                isinstance(b, float) and math.isnan(b)) or abs(a - b) > TOL * max(1, abs(b)):
            counts[name] += 1
            worst.setdefault(name, ident)

    # Every event: the planned entry day.
    planned_bad = 0
    for fd, acc in zip(run.filing_date, run.accepted_at):
        if isinstance(acc, str) and acc:
            continue  # live filings: checked in unit tests
        if next_session(pd.Timestamp(fd).date()) != _planned(fd, sessions):
            planned_bad += 1

    for r in rows.itertuples():
        counts["checked"] += 1
        ident = f"{r.ticker} {r.filing_date}"
        df = bars(root, r.ticker, cache)
        planned = next_session(pd.Timestamp(r.filing_date).date())
        entry = pd.Timestamp(r.entry).date()
        if entry != planned and not (isinstance(r.flags, str) and "late_entry" in r.flags):
            counts["entry_day"] += 1
            worst.setdefault("entry_day", ident)
        e_open = df.loc[entry, "adjOpen"]
        pre = df[df.index < entry]
        t0 = df[df.index <= pd.Timestamp(r.first_trade_date).date()]
        if len(t0) and (pd.Timestamp(r.first_trade_date).date() - t0.index[-1]).days <= 5 and len(pre) \
                and pre.index[-1] >= t0.index[-1]:
            move = pre["adjClose"].iloc[-1] / t0["adjClose"].iloc[-1] - 1
            cmp("drift", 100 * (-move if r.side == "sell" else move), r.drift_pct, ident)
        sector = bars(root, r.sector, cache) if r.sector and r.sector != "SPY" else spy
        iwm = bars(root, "IWM", cache)
        for h in HORIZONS:
            target = add(entry, h - 1)
            if target > last:
                continue
            got_exit = getattr(r, f"exit_{h}")
            if not isinstance(got_exit, str):
                continue
            upto = df[df.index <= target]
            ex = upto.index[-1]
            if (target - ex).days <= 10 and ex != pd.Timestamp(got_exit).date():
                counts["exit_day"] += 1
                worst.setdefault("exit_day", ident)
            ex = pd.Timestamp(got_exit).date()
            cmp("stock", df.loc[ex, "adjClose"] / e_open - 1, getattr(r, f"stock_{h}"), ident)
            for name, b in (("spy", spy), ("sector", sector), ("iwm", iwm)):
                if b is None or entry not in b.index:
                    continue
                bx = b[b.index <= ex]
                val = bx["adjClose"].iloc[-1] / b.loc[entry, "adjOpen"] - 1
                cmp(name, val, getattr(r, f"{name}_{h}"), ident)
    out = [f"planned entry vs XNYS next session: {planned_bad} mismatches over {len(run)} events",
           f"returns: {counts['checked']} events checked"]
    out += [f"  {k}: {v} mismatches" + (f" (first: {worst[k]})" if k in worst else "")
            for k, v in counts.items() if k != "checked"]
    return out


def _planned(fd, sessions):
    import bisect

    return sessions[bisect.bisect_right(sessions, pd.Timestamp(fd).date())]


def check_stats(run: pd.DataFrame, h: int = 20) -> list[str]:
    ok = run[(run.status == "ok") & (run.side == "buy") & (run.is_10b5_1 == 0)].copy()
    ok = ok[ok[f"spy_{h}"].notna() & ok[f"stock_{h}"].notna()]
    ok["x"] = ok[f"stock_{h}"] - ok[f"spy_{h}"]
    ok["bucket"] = pd.cut(ok.lag_days, [-1, 1, 2, 4, 10**9], labels=["fast", "on time", "late", "very late"])
    ok["month"] = pd.to_datetime(ok.entry).dt.to_period("M")
    out = [f"buy buckets, {h}-day excess vs SPY (numpy/scipy):",
           "  bucket n mean t t_clustered"]
    for b, g in ok.groupby("bucket", observed=True):
        x = g.x.to_numpy()
        t = sps.ttest_1samp(x, 0).statistic
        dev = (g.x - x.mean()).groupby(g.month).sum().to_numpy()
        G, n = len(dev), len(x)
        tc = x.mean() / math.sqrt((dev ** 2).sum() / n ** 2 * G / (G - 1))
        out.append(f"  {b} {n} {100 * x.mean():.2f}% {t:.2f} {tc:.2f}")
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--prices", required=True)
    p.add_argument("--db", default=os.environ.get("DATABASE_URL"))
    p.add_argument("--sample", type=int, default=5000, help="0 checks every event")
    p.add_argument("--seed", type=int, default=1)
    a = p.parse_args()
    run = pd.read_csv(Path(a.run) / "events.csv.gz", keep_default_na=True,
                      dtype={"issuer_cik": str, "accepted_at": str, "flags": str, "sector": str})
    lines = []
    if a.db:
        lines += check_events_sql(run, a.db)
    lines += check_returns(run, Path(a.prices), a.sample, a.seed)
    lines += check_stats(run)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
