"""Correlation study: insider trade characteristics against later returns.

    python scripts/correlation_study.py \
        --events /mnt/project-files/m3-backtest/events-yahoo-2015-2026.csv.gz \
        --bars /mnt/project-files/prices-yahoo/bars \
        --shares /mnt/project-files/fundamentals/shares-outstanding.csv \
        --features /mnt/project-files/m4e-gbm/features-all.csv.gz \
        --out /mnt/project-files/correlation

Writes tables (CSV), summary.json, report.md and charts (PNG, needs matplotlib).
The enriched events are cached in --cache so a rerun skips the bar reads.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tradetracker import correlation as C

SIDES = ("buy", "sell")


def enrich(args) -> tuple[pd.DataFrame, np.ndarray]:
    cache = Path(args.cache)
    if (cache / "events.pkl").exists() and (cache / "paths.npy").exists():
        return pd.read_pickle(cache / "events.pkl"), np.load(cache / "paths.npy")
    e = C.load_events(args.events)
    e, paths = C.add_price_features(e, args.bars, args.shares, paths=True)
    f = pd.read_csv(args.features, dtype={"issuer_cik": str},
                    usecols=["filer_id", "issuer_cik", "filing_date", "stake_change",
                             "drawdown", "log_market_cap"])
    f["filing_date"] = pd.to_datetime(f["filing_date"])
    f = f.drop_duplicates(["filer_id", "issuer_cik", "filing_date"]).rename(
        columns={"drawdown": "gbm_drawdown", "log_market_cap": "gbm_log_market_cap"})
    e["issuer_cik"] = e["issuer_cik"].astype(str)
    e = e.merge(f, on=["filer_id", "issuer_cik", "filing_date"], how="left")
    e.loc[e["side"] != "buy", "stake_change"] = np.nan  # holdings change: buys only
    cache.mkdir(parents=True, exist_ok=True)
    e.to_pickle(cache / "events.pkl")
    np.save(cache / "paths.npy", paths)
    return e, paths


def add_winsorized(e: pd.DataFrame) -> pd.DataFrame:
    for side in SIDES:
        m = e["side"] == side
        for h in C.HORIZONS:
            for kind in ("raw", "adj"):
                col = f"{kind}_{h}"
                e.loc[m, col + "_w"] = C.winsorize(e.loc[m, col])
    return e


def baseline(e: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for side in SIDES:
        s = e[e["side"] == side]
        for h in C.HORIZONS:
            y = s[f"adj_{h}"].dropna()
            idx = y.index
            m, se = C.clustered_mean(s.loc[idx, f"adj_{h}_w"], s.loc[idx, "month"].astype(str))
            rows.append({
                "side": side, "h": h, "n": len(y),
                "raw_mean": s.loc[idx, f"raw_{h}"].mean(),
                "adj_mean": y.mean(), "adj_wmean": m,
                "lo": m - 1.96 * se, "hi": m + 1.96 * se,
                "adj_median": y.median(), "hit": (y > 0).mean(),
                "iwm_adj_wmean": C.winsorize(s.loc[idx, f"iwm_adj_{h}"]).mean(),
                "net_wmean": (s.loc[idx, f"adj_{h}_w"] - s.loc[idx, "cost"]).mean(),
            })
    return pd.DataFrame(rows)


def ic_table(e: pd.DataFrame, dedup: bool = False) -> pd.DataFrame:
    rows = []
    for side in SIDES:
        s = e[e["side"] == side]
        if dedup:  # one event per stock and entry day
            s = s.drop_duplicates(["ticker", "entry"])
        for name, col, kind in C.FEATURES:
            if kind == "cat" or s[col].notna().sum() < C.MIN_FEATURE_N:
                continue
            for h in C.HORIZONS:
                for r in ("raw", "adj"):
                    ic = C.monthly_ic(s, col, f"{r}_{h}", h)
                    rows.append({"side": side, "feature": name, "column": col, "h": h,
                                 "return": r, "pooled_ic": ic.pooled,
                                 "monthly_ic": ic.monthly_mean, "t": ic.t,
                                 "months": ic.months, "share_positive": ic.share_positive,
                                 "n": ic.n})
    return pd.DataFrame(rows)


def buckets(e: pd.DataFrame) -> pd.DataFrame:
    out = []
    for side in SIDES:
        s = e[e["side"] == side]
        for name, col, kind in C.FEATURES:
            if s[col].notna().sum() < C.MIN_FEATURE_N:
                continue
            for h in C.HORIZONS:
                for r in ("raw", "adj"):
                    t = C.bucket_table(s, col, kind, f"{r}_{h}")
                    t.insert(0, "return", r)
                    t.insert(0, "h", h)
                    t.insert(0, "feature", name)
                    t.insert(0, "side", side)
                    if side == "buy" and r == "adj":
                        n = C.bucket_table(s, col, kind, f"{r}_{h}", net=True)
                        t["net_wmean"] = n["wmean"].to_numpy()
                    out.append(t)
    return pd.concat(out, ignore_index=True)


def yearly(e: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for side in SIDES:
        s = e[e["side"] == side]
        for name, col, kind in C.FEATURES:
            if kind == "cat" or s[col].notna().sum() < C.MIN_FEATURE_N:
                continue
            for h in C.HORIZONS:
                y = C.yearly_ic(s, col, f"adj_{h}")
                for year, v in y.items():
                    rows.append({"side": side, "feature": name, "h": h, "year": year,
                                 "ic": v})
    return pd.DataFrame(rows)


def joint(e: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for side in SIDES:
        s = e[e["side"] == side].copy()
        feats = [c for _, c, k in C.FEATURES
                 if k != "cat" and c != "value_to_cap"
                 and s[c].notna().sum() >= C.MIN_FEATURE_N]
        dummies = []
        for role in ("CEO", "CFO", "director", "10% owner"):  # officer is the base
            col = "role_" + role.replace(" ", "_").replace("%", "pct")
            s[col] = (s["role"] == role).astype(float)
            dummies.append(col)
        for h in C.HORIZONS:
            t = C.fama_macbeth(s, feats, f"adj_{h}_w", h, dummies)
            t.insert(0, "h", h)
            t.insert(0, "side", side)
            rows.append(t)
    return pd.concat(rows, ignore_index=True)


def event_paths(e: pd.DataFrame, P: np.ndarray) -> pd.DataFrame:
    days = np.arange(-C.PATH_SESSIONS, C.PATH_SESSIONS + 1)
    out = {"session": days}
    for side in SIDES:
        idx = e.index[e["side"] == side].to_numpy()
        M = P[idx]
        lo, hi = np.nanquantile(M, 0.01, axis=0), np.nanquantile(M, 0.99, axis=0)
        M = np.clip(M, lo, hi)
        out[f"{side}_mean"] = np.nanmean(M, axis=0)
        out[f"{side}_median"] = np.nanmedian(M, axis=0)
        out[f"{side}_n"] = np.sum(~np.isnan(M), axis=0)
    return pd.DataFrame(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True)
    ap.add_argument("--bars", required=True)
    ap.add_argument("--shares", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default="/tmp/correlation-cache")
    ap.add_argument("--no-charts", action="store_true")
    args = ap.parse_args(argv)
    out = Path(args.out)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    e, P = enrich(args)
    e = add_winsorized(e)
    res = {
        "baseline": baseline(e), "ic": ic_table(e), "ic_dedup": ic_table(e, dedup=True),
        "buckets": buckets(e), "yearly_ic": yearly(e), "joint": joint(e),
        "event_paths": event_paths(e, P),
    }
    for k, v in res.items():
        v.to_csv(out / "tables" / f"{k}.csv", index=False, float_format="%.12g")
    coverage = {side: {c: float(e.loc[e["side"] == side, c].notna().mean())
                       for _, c, _k in C.FEATURES}
                for side in SIDES}
    summary = {"events": {s: int((e["side"] == s).sum()) for s in SIDES},
               "years": [int(e["year"].min()), int(e["year"].max())],
               "coverage": coverage}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    if not args.no_charts:
        from correlation_charts import draw_all  # noqa: E402  (scripts/)
        draw_all(res, out)
    return res


if __name__ == "__main__":
    main()
