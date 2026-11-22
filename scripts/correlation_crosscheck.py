"""Independent recomputation of the correlation study's numbers.

    python scripts/correlation_crosscheck.py --study /mnt/project-files/correlation \
        --cache /tmp/correlation-cache --events ... --bars ... [--sample 3000]

Different code paths from tradetracker.correlation on purpose:

1. Sample and baseline returns: filters and means redone in SQL (sqlite) straight from
   the raw event CSV, compared with tables/baseline.csv.
2. Price features: 52-week drawdown and prior 20-day return redone with pandas
   rolling windows and merge_asof for every event; drawdown and market cap also
   compared with the phase 4e feature file (a separate, earlier pipeline).
3. Point in time: for a sample, features recomputed from bars cut off at the last
   close before entry must equal the ones from the full history.
4. Returns: for a sample, the 20 and 60-day stock returns recomputed from the bars
   (adjusted open at entry to adjusted close h sessions later).
5. Correlations: monthly rank correlations redone with scipy.stats.spearmanr and the
   Newey-West t with an explicit autocovariance loop; bucket counts and means redone
   in SQL from the bucket edges.
Each check prints PASS/FAIL with the worst gap; the exit status is 1 on any FAIL.
"""
from __future__ import annotations

import argparse
import math
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from tradetracker import correlation as C

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str):
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
    if not ok:
        FAILS.append(name)


def sql_baseline(events_csv: str, study: Path):
    raw = pd.read_csv(events_csv, low_memory=False)
    con = sqlite3.connect(":memory:")
    cols = ["side", "status", "is_10b5_1", "entry_open", "dollar_volume",
            *[f"stock_{h}" for h in C.HORIZONS], *[f"spy_{h}" for h in C.HORIZONS]]
    raw[cols].to_sql("ev", con)
    base = pd.read_csv(study / "tables" / "baseline.csv")
    worst = 0.0
    for h in C.HORIZONS:
        q = f"""SELECT side, COUNT(*) n, AVG(stock_{h} - spy_{h}) adj, AVG(stock_{h}) raw
                FROM ev WHERE status = 'ok' AND COALESCE(is_10b5_1, 0) = 0
                  AND entry_open >= 2 AND dollar_volume >= 100000
                  AND stock_{h} IS NOT NULL AND spy_{h} IS NOT NULL
                GROUP BY side"""
        for side, n, adj, rawm in con.execute(q):
            b = base[(base["side"] == side) & (base["h"] == h)].iloc[0]
            check(f"baseline {side} {h}d count", int(b["n"]) == n, f"{b['n']} vs SQL {n}")
            worst = max(worst, abs(b["adj_mean"] - adj), abs(b["raw_mean"] - rawm))
    check("baseline means (SQL)", worst < 1e-9, f"worst gap {worst:.2e}")


def rolling_features(e: pd.DataFrame, bars_dir: Path) -> pd.DataFrame:
    out = []
    for t, g in e.groupby("ticker"):
        p = bars_dir / f"{t}.csv.gz"
        if not p.exists():
            continue
        b = pd.read_csv(p, usecols=["date", "adjClose"]).dropna()
        b = b[b["adjClose"] > 0]
        b["date"] = pd.to_datetime(b["date"])
        b = b.sort_values("date").drop_duplicates("date")
        b["hi"] = b["adjClose"].rolling(250, min_periods=60).max()
        b["r20"] = b["adjClose"] / b["adjClose"].shift(20) - 1
        b["dd"] = b["adjClose"] / b["hi"] - 1
        # the last bar strictly before entry
        k = g[["entry"]].reset_index().sort_values("entry")
        m = pd.merge_asof(k, b[["date", "dd", "r20"]], left_on="entry", right_on="date",
                          allow_exact_matches=False)
        out.append(m.set_index("index")[["dd", "r20"]])
    return pd.concat(out)


def features(e: pd.DataFrame, bars_dir: Path):
    f = rolling_features(e, bars_dir)
    j = e[["drawdown", "pre_ret_20"]].join(f, how="inner")
    for a, b in (("drawdown", "dd"), ("pre_ret_20", "r20")):
        both = j[[a, b]].dropna()
        gap = (both[a] - both[b]).abs()
        miss = int(j[a].notna().sum() != j[b].notna().sum())
        check(f"{a} vs rolling recompute", gap.max() < 1e-9 and not miss,
              f"{len(both)} events, worst gap {gap.max():.2e}, "
              f"non-null {j[a].notna().sum()} vs {j[b].notna().sum()}")
    buys = e[e["side"] == "buy"]
    both = buys[["drawdown", "gbm_drawdown"]].dropna()
    gap = (both["drawdown"] - both["gbm_drawdown"]).abs()
    check("drawdown vs phase 4e features", (gap < 1e-3).mean() > 0.99,
          f"{len(both)} buys, {100 * (gap < 1e-3).mean():.2f}% within 0.1 pp, "
          f"rank corr {spear(both['drawdown'], both['gbm_drawdown']):.4f}")
    both = buys[["log_market_cap", "gbm_log_market_cap"]].dropna()
    gap = (both["log_market_cap"] - both["gbm_log_market_cap"]).abs()
    check("market cap vs phase 4e features", (gap < 0.01).mean() > 0.99,
          f"{len(both)} buys, {100 * (gap < 0.01).mean():.2f}% within 1%, "
          f"rank corr {spear(both['log_market_cap'], both['gbm_log_market_cap']):.4f}")


def spear(a, b) -> float:
    return float(stats.spearmanr(a, b, nan_policy="omit").statistic)


def point_in_time(e: pd.DataFrame, bars_dir: Path, shares_path: str, n: int, seed=7):
    s = e.sample(min(n, len(e)), random_state=seed)
    spy = C.read_bars(bars_dir / "SPY.csv.gz")
    shares = C.load_shares(shares_path)
    worst, count = 0.0, 0
    for t, g in s.groupby("ticker"):
        bars = C.read_bars(bars_dir / f"{t}.csv.gz")
        if bars is None:
            continue
        for idx, row in g.iterrows():
            sh = shares.get(str(row["issuer_cik"]).lstrip("0"))
            cut = bars[bars["date"] < row["entry"]].reset_index(drop=True)
            if cut.empty:
                continue
            a, _ = C.price_features(g.loc[[idx]], cut, spy, sh)
            for c in ("drawdown", "pre_ret_20", "market_cap"):
                x, y = a[c].iloc[0], e.at[idx, c]
                if (pd.isna(x)) != (pd.isna(y)):
                    worst = math.inf
                elif not pd.isna(x):
                    worst = max(worst, abs(x - y) / max(1.0, abs(y)))
            count += 1
    check("point in time (bars cut before entry)", worst < 1e-12,
          f"{count} events, worst relative gap {worst:.2e}")


def returns(e: pd.DataFrame, bars_dir: Path, n: int, seed=11):
    s = e.sample(min(n, len(e)), random_state=seed)
    gaps = {h: [] for h in (20, 60)}
    for t, g in s.groupby("ticker"):
        p = bars_dir / f"{t}.csv.gz"
        b = pd.read_csv(p, usecols=["date", "adjOpen", "adjClose"])
        b["date"] = pd.to_datetime(b["date"])
        b = b.dropna(subset=["adjClose"]).sort_values("date").reset_index(drop=True)
        for _, row in g.iterrows():
            i = b.index[b["date"] == row["entry"]]
            if not len(i):
                continue
            i = i[0]
            for h in (20, 60):
                if pd.isna(row[f"stock_{h}"]) or i + h - 1 >= len(b):
                    continue
                exit_date = pd.to_datetime(row[f"exit_{h}"])
                k = b.index[b["date"] == exit_date]
                if not len(k):
                    continue
                r = b.at[k[0], "adjClose"] / b.at[i, "adjOpen"] - 1
                gaps[h].append(abs(r - row[f"stock_{h}"]))
    for h, g in gaps.items():
        g = np.array(g)
        check(f"{h}-day stock return from bars", len(g) > 0 and np.quantile(g, 0.99) < 1e-3,
              f"{len(g)} events, 99th pct gap {np.quantile(g, 0.99):.2e}, "
              f"{100 * (g < 1e-4).mean():.2f}% within 0.01 pp")


def nw(x: np.ndarray, L: int) -> tuple[float, float]:
    n, m = len(x), x.mean()
    gamma = [np.sum((x[k:] - m) * (x[:n - k] - m)) / n for k in range(L + 1)]
    var = gamma[0] + 2 * sum((1 - k / (L + 1)) * gamma[k] for k in range(1, L + 1))
    return m, m / math.sqrt(var / n)


def correlations(e: pd.DataFrame, study: Path):
    ic = pd.read_csv(study / "tables" / "ic.csv")
    worst_ic, worst_t, worst_p = 0.0, 0.0, 0.0
    for _, r in ic.iterrows():
        s = e[e["side"] == r["side"]]
        col, ret = r["column"], f"{r['return']}_{r['h']}"
        d = s[[col, ret, "month"]].dropna()
        vals = []
        for _, g in d.groupby("month"):
            if len(g) >= C.MIN_MONTH_N:
                v = stats.spearmanr(g[col], g[ret]).statistic
                if np.isfinite(v):
                    vals.append(v)
        m, t = nw(np.array(vals), max(1, math.ceil(r["h"] / 21)))
        worst_ic = max(worst_ic, abs(m - r["monthly_ic"]))
        worst_t = max(worst_t, abs(t - r["t"]))
        worst_p = max(worst_p, abs(spear(d[col], d[ret]) - r["pooled_ic"]))
    check("monthly rank correlations (scipy)", worst_ic < 1e-9,
          f"{len(ic)} rows, worst gap {worst_ic:.2e}")
    check("Newey-West t-stats", worst_t < 1e-6, f"worst gap {worst_t:.2e}")
    check("pooled rank correlations (scipy)", worst_p < 1e-9, f"worst gap {worst_p:.2e}")


def buckets_sql(e: pd.DataFrame, study: Path):
    b = pd.read_csv(study / "tables" / "buckets.csv")
    con = sqlite3.connect(":memory:")
    worst_n, worst_m, rows = 0, 0.0, 0
    for side in ("buy", "sell"):
        s = e[e["side"] == side]
        for name, col, kind in C.FEATURES:
            if kind != "q" or s[col].notna().sum() < C.MIN_FEATURE_N:
                continue
            edges = list(s[col].dropna().quantile([0.2, 0.4, 0.6, 0.8]))
            t = s[[col, "adj_60", "adj_60_w"]].rename(columns={col: "x"})
            t.to_sql("t", con, if_exists="replace")
            case = ("CASE WHEN x <= ? THEN 'Q1' WHEN x <= ? THEN 'Q2' WHEN x <= ? "
                    "THEN 'Q3' WHEN x <= ? THEN 'Q4' ELSE 'Q5' END")
            q = (f"SELECT {case} q, COUNT(*), AVG(adj_60_w) FROM t "
                 "WHERE x IS NOT NULL AND adj_60 IS NOT NULL GROUP BY q")
            for q_, n, m in con.execute(q, edges):
                r = b[(b["side"] == side) & (b["feature"] == name) & (b["h"] == 60)
                      & (b["return"] == "adj") & (b["bucket"] == q_)].iloc[0]
                worst_n = max(worst_n, abs(int(r["n"]) - n))
                worst_m = max(worst_m, abs(r["wmean"] - m))
                rows += 1
    check("quintile buckets, 60 days (SQL)", worst_n == 0 and worst_m < 1e-9,
          f"{rows} buckets, worst count gap {worst_n}, worst mean gap {worst_m:.2e}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--events", required=True)
    ap.add_argument("--bars", required=True)
    ap.add_argument("--shares", required=True)
    ap.add_argument("--sample", type=int, default=3000)
    a = ap.parse_args(argv)
    study, bars = Path(a.study), Path(a.bars)
    e = pd.read_pickle(Path(a.cache) / "events.pkl")
    sys.path.insert(0, str(Path(__file__).parent))
    from correlation_study import add_winsorized
    e = add_winsorized(e)
    sql_baseline(a.events, study)
    correlations(e, study)
    buckets_sql(e, study)
    features(e, bars)
    point_in_time(e, bars, a.shares, a.sample // 3)
    returns(e, bars, a.sample)
    print(f"\n{len(FAILS)} failed" + (f": {', '.join(FAILS)}" if FAILS else ""))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
