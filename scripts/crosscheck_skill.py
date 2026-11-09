"""Recompute a `tt skill-study` run's insider records independently, with pandas.

    python scripts/crosscheck_skill.py --run results/skill --prices DIR \
        --insiders history.csv.gz [--sample 4000]

Independent of `tradetracker.skill` on purpose: everything is rebuilt from the run's
event list (events.csv.gz), its filer -> owner map, the insider-history CSV and the raw
Yahoo bar files. Checks:

1. Episodes: one per owner, issuer and filing month (earliest filing), non-plan priced
   buys from the event list, plus 2014 history buys priced from the bars (open after the
   filing day to the h-th close, minus SPY), clipped to +-100%. Compared with
   episodes.csv.gz, row for row.
2. Shrinkage k: method of moments on episodes known before 2017.
3. Records: for a sample of shown buys, n, raw mean, hit rate, prior and the shrunk
   record from episodes whose exit is strictly before the filing date, against
   skill-pool.csv.gz.
4. Point in time: for the same sample, the record from a history truncated at the filing
   date (every later episode deleted) equals the full-history record.
5. The proven-insider gate: the scored CSV's flagged buys all have a positive 60-day
   record, and their count and net mean match summary.json.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TOL = 1e-9
MIN_EPISODES = 3
MIN_PRIOR = 100
DB_START = pd.Timestamp("2015-01-01")
K_BEFORE = pd.Timestamp("2017-01-01")
MAX_GAP = 7


def load_bars(prices: Path, ticker: str, cache: dict):
    if ticker not in cache:
        p = prices / "bars" / (re.sub(r"[^A-Za-z0-9.\-]", "_", ticker) + ".csv.gz")
        if not p.exists():
            cache[ticker] = None
        else:
            df = pd.read_csv(p, parse_dates=["date"])
            if "adjOpen" not in df or df["adjOpen"].isna().all():
                df["adjOpen"] = df["open"] * df["adjClose"] / df["close"]
            df = df[df.adjClose > 0].sort_values("date").reset_index(drop=True)
            cache[ticker] = df
    return cache[ticker]


def history_rows(run_events: pd.DataFrame, history: Path, prices: Path, h: int) -> pd.DataFrame:
    hist = pd.read_csv(history, dtype=str)
    hist = hist[hist.side == "buy"].copy()
    hist["filed"] = pd.to_datetime(hist.filed)
    hist = hist[hist.filed < DB_START]
    hist["owner"] = hist.owner_cik.str.lstrip("0")
    hist["issuer"] = hist.issuer_cik.str.lstrip("0")
    hist = hist[(hist.owner != "") & (hist.issuer != "")]
    hist = hist.drop_duplicates(["owner", "issuer", "filed"])
    ev = run_events.dropna(subset=["issuer"]).sort_values("filing_date")
    tick = ev.drop_duplicates("issuer").set_index("issuer").ticker.to_dict()
    cache: dict = {}
    spy = load_bars(prices, "SPY", cache).set_index("date")
    out = []
    for r in hist.itertuples():
        t = tick.get(r.issuer)
        df = load_bars(prices, t, cache) if t else None
        if df is None:
            continue
        after = df.index[df.date > r.filed]
        if len(after) == 0:
            continue
        i = after[0]
        if i + h - 1 >= len(df):
            continue
        a, b = df.iloc[i], df.iloc[i + h - 1]
        if (a.date - r.filed).days > MAX_GAP or not (a.adjOpen > 0):
            continue
        if a.date not in spy.index or b.date not in spy.index:
            continue
        sa, sb = spy.loc[a.date], spy.loc[b.date]
        if not (sa.adjOpen > 0):
            continue
        x = (b.adjClose / a.adjOpen - 1) - (sb.adjClose / sa.adjOpen - 1)
        out.append((r.owner, r.issuer, r.filed, b.date, x))
    return pd.DataFrame(out, columns=["owner", "issuer", "filed", "known", "excess"])


def episodes(events: pd.DataFrame, hist: pd.DataFrame, h: int) -> pd.DataFrame:
    e = events[(events.side == "buy") & (events.is_10b5_1 == 0) & (events.status == "ok")]
    e = e.dropna(subset=[f"stock_{h}", f"spy_{h}", f"exit_{h}", "owner"])
    rows = pd.DataFrame({"owner": e.owner, "issuer": e.issuer.fillna(e.ticker),
                         "filed": e.filing_date, "known": pd.to_datetime(e[f"exit_{h}"]),
                         "excess": e[f"stock_{h}"] - e[f"spy_{h}"]})
    rows = pd.concat([rows, hist], ignore_index=True)
    rows["ym"] = rows.filed.dt.year * 100 + rows.filed.dt.month
    rows = rows.sort_values(["filed", "known"]).drop_duplicates(["owner", "issuer", "ym"])
    rows["excess"] = rows.excess.clip(-1, 1)
    return rows.drop(columns="ym").sort_values("known").reset_index(drop=True)


def estimate_k(eps: pd.DataFrame) -> float:
    g = eps[eps.known < K_BEFORE].groupby("owner").excess
    n = g.size()
    keep = n[n >= 2].index
    sub = eps[(eps.known < K_BEFORE) & eps.owner.isin(keep)]
    if len(keep) < 30:
        return 20.0
    gg = sub.groupby("owner").excess
    m, cnt = gg.mean(), gg.size()
    s2 = ((sub.excess - sub.owner.map(m)) ** 2).sum() / (cnt - 1).sum()
    tau2 = m.var(ddof=1) - (s2 / cnt).mean()
    return 100.0 if tau2 <= 0 else float(min(max(s2 / tau2, 1.0), 100.0))


def record(eps: pd.DataFrame, owner: str, d: pd.Timestamp, k: float) -> dict:
    known = eps[eps.known < d]
    prior = known.excess.mean() if len(known) >= MIN_PRIOR else 0.0
    mine = known[known.owner == owner].excess
    n = len(mine)
    if n == 0:
        return {"n": 0, "raw": math.nan, "hit": math.nan, "shrunk": math.nan}
    raw = mine.mean()
    shrunk = prior + n / (n + k) * (raw - prior) if n >= MIN_EPISODES else math.nan
    return {"n": n, "raw": raw, "hit": (mine > 0).mean(), "shrunk": shrunk}


def same(a, b, tol=TOL) -> bool:
    an, bn = a is None or (isinstance(a, float) and math.isnan(a)), b is None or (
        isinstance(b, float) and math.isnan(b))
    if an or bn:
        return an and bn
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--prices", required=True)
    ap.add_argument("--insiders", required=True)
    ap.add_argument("--sample", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    run, prices = Path(a.run), Path(a.prices)
    bad = 0

    events = pd.read_csv(run / "events.csv.gz", dtype={"issuer_cik": str},
                         parse_dates=["filing_date"])
    keys = pd.read_csv(run / "owner-keys.csv.gz", dtype=str)
    keys["owner"] = keys.owner_cik.str.lstrip("0")
    events["owner"] = events.filer_id.astype(str).map(keys.set_index("filer_id").owner)
    events["issuer"] = events.issuer_cik.str.lstrip("0").replace("", np.nan)
    mine = pd.read_csv(run / "episodes.csv.gz", dtype={"owner": str, "issuer": str},
                       parse_dates=["filed", "known"])
    summary = json.loads((run / "summary.json").read_text())
    pool = pd.read_csv(run / "skill-pool.csv.gz", dtype={"owner": str},
                       parse_dates=["filing_date"])
    sample = pool.sample(min(a.sample, len(pool)), random_state=a.seed)

    for h in (20, 60):
        hist = history_rows(events, Path(a.insiders), prices, h)
        eps = episodes(events, hist, h)
        theirs = mine[mine.horizon == h].reset_index(drop=True)
        key = ["owner", "issuer", "filed"]
        m = eps.merge(theirs, on=key, how="outer", suffixes=("", "_run"), indicator=True)
        only = (m._merge != "both").sum()
        diff = m[(m._merge == "both") & (((m.excess - m.excess_run).abs() > 1e-5)
                                         | (m.known != m.known_run))]
        print(f"[{h}d] episodes: mine {len(eps)} (history {len(hist)}), run {len(theirs)}, "
              f"unmatched {only}, outcome or exit differences {len(diff)}")
        bad += only + len(diff)
        if only:
            print(m[m._merge != "both"].head(10).to_string())

        k = estimate_k(eps)
        k_run = summary["coverage"]["k"][str(h)]
        print(f"[{h}d] k: mine {k:.6f}, run {k_run:.6f}")
        bad += not same(k, k_run, 1e-4)

        # Records on the sample, with the run's own episodes (so this checks the record
        # logic alone; the episode check above covers their inputs) and point in time.
        mism = pit = 0
        for r in sample.itertuples():
            owner = r.owner if isinstance(r.owner, str) else f"filer:{r.filer_id}"
            got = record(theirs, owner, r.filing_date, k_run)
            cut = record(theirs[theirs.known < r.filing_date], owner, r.filing_date, k_run)
            run_vals = {"n": getattr(r, f"skill_n{h}"), "raw": getattr(r, f"skill_raw{h}"),
                        "hit": getattr(r, f"skill_hit{h}"), "shrunk": getattr(r, f"skill_{h}")}
            ok = all(same(float(got[f]), float(run_vals[f]), 1e-6) for f in got)
            mism += not ok
            pit += any(not same(float(got[f]), float(cut[f])) for f in got)
            if not ok and mism <= 5:
                print(f"  record mismatch {r.ticker} {r.filing_date.date()} {owner}: "
                      f"mine {got} run {run_vals}")
        print(f"[{h}d] records: {len(sample)} sampled buys, {mism} mismatches, "
              f"{pit} point-in-time changes")
        bad += mism + pit

    sc = pd.read_csv(run / "scored-proven-60-days.csv.gz")
    flagged = sc[sc.above == 1]
    merged = flagged.merge(pool[["filer_id", "ticker", "filing_date", "skill_60"]].assign(
        filing_date=pool.filing_date.dt.strftime("%Y-%m-%d")),
        on=["filer_id", "ticker", "filing_date"], how="left")
    neg = (~(merged.skill_60 > 0)).sum()
    name = next(n for n in summary["variants"] if n.startswith("proven insiders, 60"))
    pooled = summary["variants"][name]["pooled"]
    print(f"proven gate: {len(flagged)} flagged, {neg} without a positive record; "
          f"summary {pooled['above']} flagged, mean {pooled['net_mean']:.6f} vs "
          f"recomputed {flagged.target.mean():.6f}")
    bad += neg + (len(flagged) != pooled["above"]) + (not same(flagged.target.mean(),
                                                                pooled["net_mean"], 1e-9))
    print("OK" if bad == 0 else f"{bad} problems")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
