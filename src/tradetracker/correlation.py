"""Correlation study: insider trade characteristics against later stock returns.

Reads the phase 3 event file (one row per insider, issuer and filing, with returns
already measured from the filing) and the cached Yahoo bars, adds the point-in-time
price features each event needs, and measures how each characteristic lines up with
the stock's return 5, 20 and 60 trading days after entry.

Everything an event's features use is known before the entry open: bars up to the
last close before entry, share counts dated at least 100 days earlier (as in
`fundamentals.py`), and cluster counts that only see filings public by that day.

Statistics, all on the stock's return (not signed by the insider's side, so a sell
that "worked" shows a negative return):

* rank correlation (Spearman) per filing month, averaged, with a Newey-West t-stat
  (the lag covers the overlap of 60-day windows across months);
* the same correlation per calendar year, for stability;
* bucket means of the winsorized return with 95% intervals clustered by filing
  month (events in the same month share the market's moves);
* a Fama-MacBeth regression of the return on every characteristic at once (each
  ranked within its month), to see which ones survive the others.

pandas and numpy only, so the core runs in CI without the data.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

HORIZONS = (5, 20, 60)
MIN_PRICE = 2.0                 # same liquid filter as the phase 3 backtest
MIN_DOLLAR_VOLUME = 100_000
SHARES_LAG_DAYS = 100           # fundamentals.LAG_DAYS
DRAWDOWN_SESSIONS = 250          # as scoring/signals.py: the previous 250 sessions
MIN_DRAWDOWN_SESSIONS = 60
RUNUP_SESSIONS = 20
PATH_SESSIONS = 60              # event-study window either side of entry
WINSOR = (0.01, 0.99)
MIN_MONTH_N = 30                # events a month needs to get a cross-sectional IC
MIN_FEATURE_N = 1000           # a characteristic needs this many events on a side

LAG_BUCKETS = [(-np.inf, 1, "0-1 days"), (1, 2, "2 days"), (2, 4, "3-4 days"),
               (4, np.inf, "5+ days")]
CLUSTER_BUCKETS = [(0, 1, "alone"), (1, 2, "2 insiders"), (2, 4, "3-4 insiders"),
                   (4, np.inf, "5+ insiders")]
ROLE_ORDER = ["CEO", "CFO", "officer", "director", "10% owner", "unknown"]

# plain-language name, column, kind ("q" = quintiles, "lag"/"cluster" = fixed
# buckets, "cat" = role)
FEATURES = [
    ("trade size ($)", "log_value", "q"),
    ("trade size vs market cap", "value_to_cap", "q"),
    ("buy size vs holdings (buys only)", "stake_change", "q"),
    ("insider role", "role", "cat"),
    ("insiders trading together", "cluster_count", "cluster"),
    ("% below 52-week high", "below_high", "q"),
    ("run-up, trade to filing", "drift", "q"),
    ("prior 20-day return", "pre_ret_20", "q"),
    ("filing lag", "lag_days", "lag"),
    ("company size (market cap)", "log_market_cap", "q"),
]
NUMERIC = [c for _, c, k in FEATURES if k != "cat"]


# ---------------------------------------------------------------- loading


def load_events(path: Path | str) -> pd.DataFrame:
    """The phase 3 events with returns: status ok, no 10b5-1 plan, liquid."""
    e = pd.read_csv(path, low_memory=False,
                    dtype={"issuer_cik": str, "ticker": str, "role": str})
    return prepare_events(e)


def prepare_events(e: pd.DataFrame) -> pd.DataFrame:
    e = e[(e["status"] == "ok") & (e["is_10b5_1"].fillna(0).astype(int) == 0)].copy()
    e = e[(e["entry_open"] >= MIN_PRICE) & (e["dollar_volume"] >= MIN_DOLLAR_VOLUME)]
    e["filing_date"] = pd.to_datetime(e["filing_date"])
    e["entry"] = pd.to_datetime(e["entry"])
    e["month"] = e["filing_date"].dt.to_period("M")
    e["year"] = e["filing_date"].dt.year
    e["role"] = e["role"].fillna("unknown")
    e["log_value"] = np.log(e["value"].where(e["value"] > 0))
    e["drift"] = e["drift_pct"] / 100.0
    for h in HORIZONS:
        e[f"raw_{h}"] = e[f"stock_{h}"]
        e[f"adj_{h}"] = e[f"stock_{h}"] - e[f"spy_{h}"]
        e[f"iwm_adj_{h}"] = e[f"stock_{h}"] - e[f"iwm_{h}"]
    return e.reset_index(drop=True)


def read_bars(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    b = pd.read_csv(path, usecols=["date", "close", "adjClose"])
    b = b.dropna(subset=["adjClose"])
    b = b[b["adjClose"] > 0]
    b["date"] = pd.to_datetime(b["date"])
    return b.sort_values("date").drop_duplicates("date").reset_index(drop=True)


def load_shares(path: Path | str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    s = pd.read_csv(path, dtype={"cik": str})
    s["cik"] = s["cik"].str.lstrip("0")
    s["end"] = pd.to_datetime(s["end"])
    s = s[s["shares"] > 0].drop_duplicates(["cik", "end"]).sort_values(["cik", "end"])
    return {k: (g["end"].to_numpy(), g["shares"].to_numpy()) for k, g in s.groupby("cik")}


# ---------------------------------------------------------------- price features


def price_features(events: pd.DataFrame, bars: pd.DataFrame, spy: pd.DataFrame,
                   shares: tuple[np.ndarray, np.ndarray] | None = None,
                   paths: bool = False):
    """Point-in-time features for one ticker's events, from its bars.

    Returns a frame indexed like `events` with drawdown, pre_ret_20 and market_cap,
    and, with paths=True, an (n, 2*PATH_SESSIONS+1) array of cumulative returns
    minus SPY's, from PATH_SESSIONS before entry to PATH_SESSIONS after, all relative
    to the last close before entry (NaN where the bars don't reach).
    """
    n = len(events)
    out = pd.DataFrame(index=events.index,
                       data={"drawdown": np.nan, "pre_ret_20": np.nan,
                             "market_cap": np.nan})
    dates = bars["date"].to_numpy()
    adj = bars["adjClose"].to_numpy(float)
    close = bars["close"].to_numpy(float)
    # index of the entry bar; the last close before entry is one before it
    ie = np.searchsorted(dates, events["entry"].to_numpy(), side="left")
    pre = ie - 1
    valid = pre >= 0
    dd = np.full(n, np.nan)
    r20 = np.full(n, np.nan)
    cap = np.full(n, np.nan)
    for k in np.flatnonzero(valid):
        i = pre[k]
        lo = max(0, i - DRAWDOWN_SESSIONS + 1)
        if i - lo + 1 >= MIN_DRAWDOWN_SESSIONS:
            dd[k] = adj[i] / adj[lo:i + 1].max() - 1.0
        if i - RUNUP_SESSIONS >= 0:
            r20[k] = adj[i] / adj[i - RUNUP_SESSIONS] - 1.0
        if shares is not None:
            ends, counts = shares
            cutoff = events["entry"].iloc[k] - pd.Timedelta(days=SHARES_LAG_DAYS)
            j = np.searchsorted(ends, cutoff.to_datetime64(), side="right") - 1
            if j >= 0:
                jb = np.searchsorted(dates, ends[j], side="right") - 1
                jb = max(jb, 0)
                if close[i] > 0 and close[jb] > 0:
                    split = (close[jb] / adj[jb]) / (close[i] / adj[i])
                    cap[k] = counts[j] * split * close[i]
    out["drawdown"], out["pre_ret_20"], out["market_cap"] = dd, r20, cap
    if not paths:
        return out, None
    w = PATH_SESSIONS
    sp = spy.set_index("date")["adjClose"]
    spy_on = sp.reindex(bars["date"]).to_numpy(float)
    P = np.full((n, 2 * w + 1), np.nan)
    for k in np.flatnonzero(valid):
        i = pre[k]
        a, b = max(0, i - w), min(len(adj) - 1, i + w)
        seg = adj[a:b + 1] / adj[i] - spy_on[a:b + 1] / spy_on[i]
        P[k, a - i + w:b - i + w + 1] = seg
    return out, P


def add_price_features(events: pd.DataFrame, bars_dir: Path | str,
                       shares_path: Path | str | None = None, paths: bool = False):
    bars_dir = Path(bars_dir)
    spy = read_bars(bars_dir / "SPY.csv.gz")
    if spy is None:
        raise RuntimeError("no SPY bars")
    shares = load_shares(shares_path) if shares_path else {}
    feats = []
    P = np.full((len(events), 2 * PATH_SESSIONS + 1), np.nan) if paths else None
    keys = ["ticker", "issuer_cik"] if "issuer_cik" in events else ["ticker"]
    cache: dict[str, pd.DataFrame | None] = {}
    # by ticker and issuer: a reused ticker belongs to different companies (and share
    # counts) over the years
    for key, g in events.groupby(keys):
        ticker = key[0] if isinstance(key, tuple) else key
        if ticker not in cache:
            cache = {ticker: read_bars(bars_dir / f"{ticker}.csv.gz")}
        bars = cache[ticker]
        if bars is None:
            continue
        cik = str(key[1]).lstrip("0") if len(keys) == 2 else None
        f, p = price_features(g, bars, spy, shares.get(cik), paths=paths)
        feats.append(f)
        if paths:
            P[g.index.to_numpy()] = p
    f = pd.concat(feats) if feats else pd.DataFrame(
        columns=["drawdown", "pre_ret_20", "market_cap"])
    e = events.drop(columns=[c for c in f.columns if c in events], errors="ignore")
    e = e.join(f)
    e["below_high"] = -e["drawdown"]          # 0.3 = 30% below the 52-week high
    e["log_market_cap"] = np.log(e["market_cap"].where(e["market_cap"] > 0))
    e["value_to_cap"] = e["value"] / e["market_cap"]
    return e, P


# ---------------------------------------------------------------- statistics


def spearman(x, y) -> float:
    x, y = pd.Series(x, dtype=float), pd.Series(y, dtype=float)
    ok = x.notna().to_numpy() & y.notna().to_numpy()
    if ok.sum() < 3:
        return float("nan")
    rx = x[ok].rank().to_numpy()
    ry = y[ok].rank().to_numpy()
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def newey_west_t(x, lags: int) -> tuple[float, float]:
    """Mean of a series and its t-stat with Newey-West (Bartlett) standard errors."""
    x = np.asarray(pd.Series(x, dtype=float).dropna(), float)
    n = len(x)
    if n < 3:
        return float("nan"), float("nan")
    m = x.mean()
    u = x - m
    s = (u @ u) / n
    for L in range(1, min(lags, n - 1) + 1):
        s += 2 * (1 - L / (lags + 1)) * (u[L:] @ u[:-L]) / n
    se = math.sqrt(s / n) if s > 0 else float("nan")
    return float(m), float(m / se) if se and se == se else float("nan")


def nw_lags(h: int) -> int:
    return max(1, math.ceil(h / 21))


def winsorize(s: pd.Series, lo: float = WINSOR[0], hi: float = WINSOR[1]) -> pd.Series:
    a, b = s.quantile(lo), s.quantile(hi)
    return s.clip(a, b)


@dataclass
class IC:
    pooled: float
    monthly_mean: float
    t: float
    months: int
    share_positive: float
    n: int


def monthly_ic(df: pd.DataFrame, feat: str, ret: str, h: int) -> IC:
    d = df[[feat, ret, "month"]].dropna()
    ics = []
    for _, g in d.groupby("month"):
        if len(g) >= MIN_MONTH_N:
            ics.append(spearman(g[feat], g[ret]))
    ics = pd.Series(ics, dtype=float).dropna()
    m, t = newey_west_t(ics, nw_lags(h))
    return IC(pooled=spearman(d[feat], d[ret]), monthly_mean=m, t=t, months=len(ics),
              share_positive=float((ics > 0).mean()) if len(ics) else float("nan"),
              n=len(d))


def yearly_ic(df: pd.DataFrame, feat: str, ret: str) -> pd.Series:
    d = df[[feat, ret, "year"]].dropna()
    return d.groupby("year").apply(lambda g: spearman(g[feat], g[ret]),
                                   include_groups=False)


def clustered_mean(y: pd.Series, clusters: pd.Series) -> tuple[float, float]:
    """Mean and its standard error clustered by `clusters`."""
    ok = y.notna()
    y, c = y[ok].astype(float), clusters[ok]
    n = len(y)
    if n < 2:
        return (float(y.mean()) if n else float("nan")), float("nan")
    m = y.mean()
    sums = (y - m).groupby(c.to_numpy()).sum()
    G = len(sums)
    if G < 2:
        return float(m), float("nan")
    var = (sums ** 2).sum() / n ** 2 * G / (G - 1)
    return float(m), float(math.sqrt(var))


def bucket_labels(df: pd.DataFrame, feat: str, kind: str,
                  edges: list[float] | None = None) -> tuple[pd.Series, list[str]]:
    x = df[feat]
    if kind == "cat":
        order = [r for r in ROLE_ORDER if r in set(x.dropna())]
        return x.where(x.isin(order)), order
    if kind in ("lag", "cluster"):
        spec = LAG_BUCKETS if kind == "lag" else CLUSTER_BUCKETS
        lab = pd.Series(pd.NA, index=x.index, dtype=object)
        for lo, hi, name in spec:
            lab[(x > lo) & (x <= hi)] = name
        return lab, [s[2] for s in spec]
    if edges is None:
        edges = list(x.dropna().quantile([0.2, 0.4, 0.6, 0.8]))
    bins = [-np.inf, *edges, np.inf]
    names = [f"Q{i + 1}" for i in range(5)]
    lab = pd.cut(x, bins=bins, labels=names, duplicates="drop").astype(object)
    return lab.where(x.notna()), names


def bucket_table(df: pd.DataFrame, feat: str, kind: str, ret: str,
                 net: bool = False) -> pd.DataFrame:
    """One row per bucket: n, mean, winsorized mean with clustered 95% CI, median,
    hit rate. `ret` should already be winsorized in a column `ret + '_w'`."""
    lab, order = bucket_labels(df, feat, kind)
    rows = []
    for name in order:
        g = df[lab == name]
        y = g[ret].dropna()
        if net:
            y = y - g.loc[y.index, "cost"]
        yw = g.loc[y.index, ret + "_w"] - (g.loc[y.index, "cost"] if net else 0)
        m, se = clustered_mean(yw, g.loc[y.index, "month"].astype(str))
        x = g.loc[y.index, feat]
        rows.append({
            "bucket": name, "n": len(y),
            "range": (f"{x.min():.3g} to {x.max():.3g}" if kind == "q" and len(x) else ""),
            "mean": y.mean() if len(y) else np.nan,
            "wmean": m, "lo": m - 1.96 * se, "hi": m + 1.96 * se,
            "median": y.median() if len(y) else np.nan,
            "hit": (y > 0).mean() if len(y) else np.nan,
        })
    return pd.DataFrame(rows)


def fama_macbeth(df: pd.DataFrame, feats: list[str], ret: str, h: int,
                 dummies: list[str] = ()) -> pd.DataFrame:
    """Monthly cross-sectional OLS of `ret` on each numeric feature ranked to [0, 1]
    within the month (missing set to 0.5) plus 0/1 dummies; mean slopes, NW t.

    A slope is the return gap between the lowest- and highest-ranked event, other
    characteristics held equal."""
    cols = list(feats) + list(dummies)
    slopes = []
    for _, g in df.dropna(subset=[ret]).groupby("month"):
        if len(g) < max(MIN_MONTH_N, 3 * len(cols)):
            continue
        X = []
        for f in feats:
            r = g[f].rank(pct=True)
            r = (r - r.min()) / (r.max() - r.min()) if r.max() > r.min() else r * 0
            X.append(r.fillna(0.5).to_numpy())
        for d in dummies:
            X.append(g[d].to_numpy(float))
        X = np.column_stack([np.ones(len(g)), *X])
        beta, *_ = np.linalg.lstsq(X, g[ret].to_numpy(float), rcond=None)
        slopes.append(beta[1:])
    S = pd.DataFrame(slopes, columns=cols)
    out = []
    for c in cols:
        m, t = newey_west_t(S[c], nw_lags(h))
        out.append({"feature": c, "slope": m, "t": t, "months": int(S[c].notna().sum())})
    return pd.DataFrame(out)
