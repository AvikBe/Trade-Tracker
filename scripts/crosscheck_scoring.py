"""Recompute a phase 4 score-backtest run independently and report mismatches.

    python scripts/crosscheck_scoring.py --run results/ --prices DIR [--db URL] [--sample N]

Independent of `tradetracker.scoring` on purpose: signals are rebuilt from SQL and
pandas, bucket rules and the score formula are re-implemented here from the spec in the
model docstring, fold tables are refit with pandas groupby, weights with scipy's
bounded least squares, and the top-20 portfolio is re-simulated with pandas.
Needs `pip install pandas numpy scipy psycopg`.

Checks:
1. Stake change per event from SQL (min of owned-after minus shares, per account).
2. Repeat buys (same insider and stock, previous 730 days) from the events file.
3. 52-week drawdown from the raw bars.
4. Track record (mean 20-day excess of earlier matured buys) from the events file.
5. Every scored event's score and above-cutoff flag from its fold's model JSON.
6. Each fold's bucket tables, counts and weights refit from the shown pool.
7. Per-year and pooled top-decile means, and the portfolio's IR, from scratch, with the
   run's setup (summary.json "config"): horizon, benchmark (a beta-hedged run shorts
   clip(beta, 0, 3) x SPY per position) and volatility-scaled sizing.
8. With --shares: market cap (latest share count dated 100+ days before entry x the last
   raw close before entry, split-adjusted), and beta and volatility, from the raw bars.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

TOL = 1e-6
NEW_STAKE = 10.0
SHRINK_N = 500
BETA_GRID = (0.0, 1.0, 2.0, 3.0, 5.0, 8.0)


# ---------------------------------------------------------------- inputs

def load(run: Path):
    kw = {"parse_dates": ["filing_date", "entry", "exit"], "dtype": {"issuer_cik": str}}
    scored = pd.read_csv(run / "scored.csv.gz", **kw)
    pool = pd.read_csv(run / "pool.csv.gz", **kw)
    ev = pd.read_csv(run / "events.csv.gz",
                     parse_dates=["filing_date", "entry", "exit_20", "first_trade_date"])
    models = {int(p.stem): json.loads(p.read_text()) for p in (run / "models").glob("*.json")}
    summary = json.loads((run / "summary.json").read_text())
    return scored, pool, ev, models, summary


def bars(root: Path, ticker: str, cache: dict):
    if ticker not in cache:
        d = root / "bars" if (root / "bars").is_dir() else root
        path = d / (re.sub(r"[^A-Za-z0-9.\-]", "_", ticker) + ".csv.gz")
        if not path.exists():
            cache[ticker] = None
        else:
            df = pd.read_csv(path, parse_dates=["date"])
            df = df[df["adjClose"] > 0].drop_duplicates("date", keep="last")
            cache[ticker] = df.set_index("date").sort_index()
    return cache[ticker]


def report(name: str, bad: list[str], n: int) -> list[str]:
    print(f"{name}: {n} checked, {len(bad)} mismatches")
    for b in bad[:10]:
        print("   ", b)
    return bad


# ---------------------------------------------------------------- 1-4: signals

STAKE_SQL = """
    SELECT f.filer_id, t.ticker, x.filing_date, coalesce(t.owner, 'self') AS acct,
           t.shares, t.price, t.shares_owned_after
    FROM trades t JOIN filings f USING (filing_id)
    JOIN (SELECT DISTINCT ON (trade_id) * FROM trade_features
          ORDER BY trade_id, computed_at DESC) x USING (trade_id)
    WHERE f.source = 'edgar' AND t.side = 'buy' AND x.duplicate_of_trade_id IS NULL
      AND coalesce(f.document_type, '') NOT LIKE '%%/A' AND x.lag_days IS NOT NULL
      AND t.ticker IS NOT NULL
"""


def check_stake(scored: pd.DataFrame, url: str) -> list[str]:
    import psycopg

    with psycopg.connect(url) as c:
        lines = pd.DataFrame(c.execute(STAKE_SQL).fetchall(),
                             columns=["filer_id", "ticker", "filing_date", "acct", "shares",
                                      "price", "owned_after"])
    for col in ("shares", "price", "owned_after"):
        lines[col] = lines[col].astype(float)
    lines["filing_date"] = pd.to_datetime(lines["filing_date"])
    lines["before"] = lines.owned_after - lines.shares
    lines["missing"] = lines.owned_after.isna() | lines.shares.isna()
    # Shares count only from lines with a dollar value (price > 0), as in the events.
    lines["sh"] = lines.shares.where((lines.shares > 0) & (lines.price > 0), 0.0)
    key = ["filer_id", "ticker", "filing_date"]
    acct = lines.groupby(key + ["acct"]).agg(before=("before", "min"),
                                              missing=("missing", "any")).reset_index()
    per = acct.groupby(key).agg(before=("before", "sum"), missing=("missing", "any"))
    per["shares"] = lines.groupby(key)["sh"].sum()
    m = scored.merge(per.reset_index(), on=key, how="left")
    bad = []
    for r in m.itertuples():
        if r.missing or pd.isna(r.before) or not r.shares > 0:
            want = None
        elif r.before <= 0:
            want = NEW_STAKE
        else:
            want = min(r.shares / r.before, NEW_STAKE)
        got = None if pd.isna(r.stake_change) else r.stake_change
        if (want is None) != (got is None) or (want is not None and abs(want - got) > 1e-4):
            bad.append(f"{r.ticker} {r.filing_date.date()} filer {r.filer_id}: {got} vs {want}")
    return report("stake change (SQL)", bad, len(m))


def check_repeat_and_track(scored: pd.DataFrame, ev: pd.DataFrame) -> list[str]:
    buys = ev[ev.side == "buy"].copy()
    by_key = {k: np.sort(g.filing_date.values) for k, g in buys.groupby(["filer_id", "ticker"])}
    bad, n = [], 0
    for r in scored.itertuples():
        ds = by_key.get((r.filer_id, r.ticker), np.array([], dtype="datetime64[ns]"))
        lo = np.datetime64(r.filing_date - pd.Timedelta(days=730))
        want = int(((ds >= lo) & (ds < np.datetime64(r.filing_date))).sum())
        n += 1
        if want != r.repeat_buys:
            bad.append(f"repeat {r.ticker} {r.filing_date.date()}: {r.repeat_buys} vs {want}")
    matured = buys[(buys.status == "ok") & (buys.is_10b5_1 == 0) & buys.stock_20.notna()
                   & buys.spy_20.notna()].copy()
    matured["r"] = matured.stock_20 - matured.spy_20
    by_filer = {k: g.sort_values("exit_20") for k, g in matured.groupby("filer_id")}
    for r in scored.itertuples():
        g = by_filer.get(r.filer_id)
        prior = g[g.exit_20 < r.entry]["r"] if g is not None else pd.Series(dtype=float)
        want = prior.mean() if len(prior) >= 2 else None
        got = None if pd.isna(r.track_record) else r.track_record
        if len(prior) != r.track_n or (want is None) != (got is None) or \
                (want is not None and abs(want - got) > 1e-5):
            bad.append(f"track {r.ticker} {r.filing_date.date()} filer {r.filer_id}: "
                       f"{got} (n={r.track_n}) vs {want} (n={len(prior)})")
    return report("repeat buys and track record (pandas)", bad, 2 * n)


def check_drawdown(scored: pd.DataFrame, root: Path, sample: int, seed: int) -> list[str]:
    rows = scored if not sample or sample >= len(scored) else scored.sample(sample, random_state=seed)
    cache: dict = {}
    bad = []
    for r in rows.itertuples():
        b = bars(root, r.ticker, cache)
        want = None
        if b is not None:
            before = b[b.index < r.entry]
            if len(before) >= 60:
                w = before["adjClose"].iloc[-250:]
                want = w.iloc[-1] / w.max() - 1
        got = None if pd.isna(r.drawdown) else r.drawdown
        if (want is None) != (got is None) or (want is not None and abs(want - got) > 1e-5):
            bad.append(f"{r.ticker} {r.entry.date()}: {got} vs {want}")
    return report("52-week drawdown (raw bars)", bad, len(rows))


def check_cap_and_risk(pool: pd.DataFrame, shares_csv: Path, root: Path, sample: int,
                       seed: int) -> list[str]:
    sh = pd.read_csv(shares_csv, dtype={"cik": str}, parse_dates=["end"])
    sh["cik"] = sh.cik.str.lstrip("0")
    sh = sh.drop_duplicates(["cik", "end"], keep="last").sort_values("end")
    by_cik = {k: g for k, g in sh.groupby("cik")}
    rows = pool if not sample or sample >= len(pool) else pool.sample(sample, random_state=seed)
    cache: dict = {}
    spy = bars(root, "SPY", cache)
    bad = []
    for r in rows.itertuples():
        b = bars(root, r.ticker, cache)
        want = None
        g = by_cik.get(str(r.issuer_cik).lstrip("0")) if isinstance(r.issuer_cik, str) else None
        if g is not None and b is not None:
            known = g[g.end <= r.entry - pd.Timedelta(days=100)]
            before = b[b.index < r.entry]
            if len(known) and len(before):
                end, n = known.end.iloc[-1], known.shares.iloc[-1]
                then = b[b.index <= end]
                last = before.iloc[-1]
                t0 = then.iloc[-1] if len(then) else b.iloc[0]
                split = (t0.close / t0.adjClose) / (last.close / last.adjClose)
                want = n * split * last.close
        got = None if pd.isna(r.market_cap) else r.market_cap
        if (want is None) != (got is None) or (want is not None and abs(want / got - 1) > 1e-6):
            bad.append(f"cap {r.ticker} {r.entry.date()}: {got} vs {want}")
        # beta and volatility: up to 250 daily returns before entry, on days SPY traded.
        wb = wv = None
        if b is not None:
            before = b[b.index < r.entry]
            if len(before) > 60:
                w = before.iloc[-251:]
                ret = w.adjClose.pct_change().iloc[1:]
                s = spy.adjClose.reindex(w.index)
                sret = (s / s.shift(1) - 1).iloc[1:]
                ok = sret.notna()
                if ok.sum() >= 59:
                    x, y = sret[ok], ret[ok]
                    wv = y.std(ddof=1) * math.sqrt(252)
                    wb = np.cov(x, y, ddof=1)[0, 1] / x.var(ddof=1) if x.var() > 0 else None
        for name, got, want in (("beta", r.beta, wb), ("vol", r.vol, wv)):
            got = None if pd.isna(got) else got
            if (want is None) != (got is None) or (want is not None and abs(want - got) > 1e-6):
                bad.append(f"{name} {r.ticker} {r.entry.date()}: {got} vs {want}")
    return report("market cap, beta and volatility (raw bars, share counts)", bad, 3 * len(rows))


# ---------------------------------------------------------------- 5-6: model

def edges(x, cuts, labels, missing):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return missing
    for c, lab in zip(cuts, labels):
        if x < c:
            return lab
    return labels[-1]


def buckets(r) -> dict:
    nan = lambda v: v is None or (isinstance(v, float) and math.isnan(v))  # noqa: E731
    sc = None if nan(r["stake_change"]) else r["stake_change"]
    return {
        "role": r["role"] if isinstance(r["role"], str) and r["role"] else "none",
        "value": edges(r["value"], [10e3, 50e3, 250e3, 1e6],
                       ["<$10k", "$10k-50k", "$50k-250k", "$250k-1M", ">=$1M"], "<$10k"),
        "stake": "new stake" if sc is not None and sc >= NEW_STAKE else
        edges(sc, [0.05, 0.2, 1.0], ["<5%", "5-20%", "20-100%", ">=100%"], "unknown"),
        "cluster": "1" if r["cluster_count"] <= 1 else "2" if r["cluster_count"] == 2 else "3+",
        "repeat": edges(r["repeat_buys"], [1, 2, 4], ["0", "1", "2-3", "4+"], "0"),
        "drawdown": edges(None if nan(r["drawdown"]) else r["drawdown"], [-0.45, -0.25, -0.10],
                          ["<=-45%", "-45 to -25%", "-25 to -10%", "within 10%"], "unknown"),
        "track": edges(None if nan(r["track_record"]) else r["track_record"], [-0.05, 0, 0.05],
                       ["< -5%", "-5 to 0%", "0 to 5%", ">= 5%"], "none yet"),
        "liquidity": edges(None if nan(r["dollar_volume"]) else r["dollar_volume"],
                           [1e6, 5e6, 50e6],
                           ["<$1M/day", "$1-5M/day", "$5-50M/day", ">=$50M/day"], "<$1M/day"),
    }


def score_of(m: dict, b: dict, drift) -> tuple[float, float]:
    w, t = m["weights"], m["tables"]
    lo = sum(w[c] * min(t[c].values()) for c in w)
    hi = sum(w[c] * max(t[c].values()) for c in w)
    raw = sum(w[c] * t[c].get(b[c], 0.0) for c in w)
    q = (raw - lo) / (hi - lo) if hi > lo else 0.5
    f = 1.0 if drift is None or math.isnan(drift) else max(0.0, 1 - m["beta"] * max(drift, -0.2))
    return q * f, q


def check_scores(scored: pd.DataFrame, models: dict) -> list[str]:
    bad = []
    for r in scored.to_dict("records"):
        m = models[r["year"]]
        s, q = score_of(m, buckets(r), r["drift"])
        above = s >= m["threshold"] - 1e-12
        if abs(s - r["score"]) > 2e-6 or abs(q - r["quality"]) > 2e-6 or \
                (above != bool(r["above"]) and abs(s - m["threshold"]) > 2e-6):
            bad.append(f"{r['ticker']} {r['entry'].date()}: {r['score']} vs {s}")
    return report("scores from fold models", bad, len(scored))


def check_refit(pool: pd.DataFrame, models: dict) -> list[str]:
    """Refit every fold from the shown events of every year (pool.csv.gz)."""
    bad = []
    rows = pool.to_dict("records")
    bk = [buckets(r) for r in rows]
    for year, m in sorted(models.items()):
        start = pd.Timestamp(year, 1, 1)
        idx = [i for i, r in enumerate(rows) if not pd.isna(r["target"])
               and r["exit"] < start]
        y = np.array([rows[i]["target"] for i in idx])
        s = np.sort(y)
        lo, hi = s[int(0.01 * (len(s) - 1))], s[int(0.99 * (len(s) - 1))]
        y = np.clip(y, lo, hi)
        overall = y.mean()
        if len(idx) != m["n_train"]:
            bad.append(f"{year}: n_train {m['n_train']} vs {len(idx)}")
            continue
        X = []
        for c in m["tables"]:
            lab = pd.Series([bk[i][c] for i in idx])
            g = pd.DataFrame({"b": lab, "y": y}).groupby("b")["y"].agg(["sum", "count"])
            want = {b: n / (n + SHRINK_N) * (sm / n - overall) for b, (sm, n) in g.iterrows()}
            for b, v in want.items():
                if abs(m["tables"][c].get(b, 99) - v) > 1e-9 or m["counts"][c].get(b) != g.loc[b, "count"]:
                    bad.append(f"{year} {c}/{b}: {m['tables'][c].get(b)} vs {v}")
            X.append([want[b] for b in lab])
        X = np.array(X).T
        w = lsq_linear(X, y - overall, bounds=(0, 1), lsmr_tol="auto", tol=1e-12).x
        for c, wc in zip(m["tables"], w):
            if abs(m["weights"][c] - wc) > 1e-3:
                bad.append(f"{year} weight {c}: {m['weights'][c]} vs {wc}")
    return report("fold refits (tables, counts, weights)", bad, len(models))


# ---------------------------------------------------------------- 7: metrics

def check_metrics(scored: pd.DataFrame, summary: dict, root: Path) -> list[str]:
    bad = []
    cfg = summary.get("config", {})
    hedged, vt = cfg.get("bench") == "beta", cfg.get("vol_target")
    top = scored[scored.above == 1]
    for y, g in top.groupby("year"):
        want = g.target.dropna().mean()
        got = summary["years"][str(y)]["net_mean"]
        if got is None or abs(want - got) > 1e-9:
            bad.append(f"{y} top net mean: {got} vs {want}")
    pooled = top.target.dropna().mean()
    if abs(pooled - summary["pooled"]["net_mean"]) > 1e-9:
        bad.append(f"pooled: {summary['pooled']['net_mean']} vs {pooled}")
    hit = (top.gross.dropna() > 0).mean()
    if abs(hit - summary["pooled"]["hit"]) > 1e-9:
        bad.append(f"hit: {summary['pooled']['hit']} vs {hit}")

    # Portfolio, re-simulated with pandas.
    cache: dict = {}
    spy = bars(root, "SPY", cache)
    spy_ret = spy["adjClose"].pct_change()
    spy_intraday = spy["adjClose"] / spy["adjOpen"] - 1
    cand = top.sort_values(["entry", "score"], ascending=[True, False])
    # Exit days are the phase 3 legs' (cross-checked against XNYS by crosscheck_backtest).
    exits = {(r.ticker, r.entry): r.exit for r in cand.itertuples()
             if not pd.isna(r.exit) and bars(root, r.ticker, cache) is not None}
    days = spy.index[(spy.index >= cand.entry.min())]
    held, excess, trades = [], [], 0
    by_day = {d: g for d, g in cand.groupby("entry")}
    end = max(exits.values())
    for d in days[days <= end]:
        tick = {h[0] for h in held}
        for r in by_day.get(d, pd.DataFrame()).itertuples():
            if len(held) >= 20 or r.ticker in tick or (r.ticker, r.entry) not in exits:
                continue
            b = 1.0 if pd.isna(r.beta) else min(max(r.beta, 0.0), 3.0)
            wt = 1.0 if not vt or pd.isna(r.vol) or not r.vol else min(2.0, vt / r.vol)
            held.append((r.ticker, r.entry, exits[(r.ticker, r.entry)], r.cost,
                         b if hedged else 1.0, wt))
            tick.add(r.ticker)
            trades += 1
        tot = 0.0
        for t, entry, ex, cost, beta, wt in held:
            b = bars(root, t, cache)
            first = d == entry
            if d in b.index:
                if first:
                    rr = b.at[d, "adjClose"] / b.at[d, "adjOpen"] - 1
                    s = spy_intraday.get(d, 0.0)
                else:
                    k = b.index.get_loc(d)
                    rr = b["adjClose"].iloc[k] / b["adjClose"].iloc[k - 1] - 1 if k > 0 else None
                    s = spy_ret.get(d, 0.0)
                e = 0.0 if rr is None or pd.isna(rr) else rr - beta * (0.0 if pd.isna(s) else s)
            else:
                e = 0.0
            if first:
                e -= cost / 2
            if d == ex:
                e -= cost / 2
            tot += e * wt
        excess.append((d, tot / 20, len(held)))
        held = [h for h in held if h[2] > d]
    while excess and excess[-1][2] == 0:
        excess.pop()
    ex = np.array([e for _, e, _ in excess])
    ir = ex.mean() / ex.std(ddof=1) * math.sqrt(252)
    p = summary["portfolio"]
    if trades != p["trades"] or len(ex) != p["days"] or abs(ir - p["ir"]) > 1e-6:
        bad.append(f"portfolio: trades {p['trades']} vs {trades}, days {p['days']} vs "
                   f"{len(ex)}, IR {p['ir']} vs {ir}")
    return report("top-decile means, hit rate and portfolio", bad, len(summary["years"]) + 3)


def run_checks(run: Path, root: Path, db: str | None, sample: int = 0,
               seed: int = 1, shares: Path | None = None) -> list[str]:
    """Signals are checked on the whole shown pool (every year); the model on test years."""
    scored, pool, ev, models, summary = load(run)
    bad = []
    if db:
        bad += check_stake(pool, db)
    bad += check_repeat_and_track(pool, ev)
    bad += check_drawdown(pool, root, sample, seed)
    if shares:
        bad += check_cap_and_risk(pool, shares, root, sample, seed)
    if models:
        bad += check_scores(scored, models)
        bad += check_refit(pool, models)
        bad += check_metrics(scored, summary, root)
    return bad


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--prices", required=True)
    p.add_argument("--db", default=os.environ.get("DATABASE_URL"))
    p.add_argument("--sample", type=int, default=5000, help="drawdown sample; 0 checks all")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--shares", help="share counts CSV, to check market caps")
    a = p.parse_args()
    bad = run_checks(Path(a.run), Path(a.prices), a.db, a.sample, a.seed,
                     Path(a.shares) if a.shares else None)
    print(f"total mismatches: {len(bad)}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
