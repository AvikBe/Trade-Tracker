"""Recompute an `tt ml-study` run's inputs and headline numbers independently.

    python scripts/crosscheck_ml.py --run results/ --prices DIR [--sample 3000]

Independent of `tradetracker.ml` on purpose: features are rebuilt with pandas from the
raw bar files and the run's event list, and the fold sizes and pooled means from the
run's CSVs. Checks:

1. The stock's price features (returns, momentum, distance from the low, sessions since
   the high, volatilities, largest daily gain, dollar-volume trend, price), on a sample.
2. The market features (SPY and IWM), on every distinct entry day of the sample.
3. Other insiders' trades (buyers, sellers, net purchase ratio, the insider's sales and
   earlier buys, days since a buy, market-wide buying), on a sample.
4. Filing features that can be derived from others (trade size against cap, month).
5. Purge and embargo: each fold's training-set size, recounted from the feature rows
   (exit at least 14 days before 1 January of the test year, with a return).
6. Each setup's above-cutoff count and net mean, from its scored CSV, against summary.json.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TOL = 1e-7
YEAR = 250


def bars(prices: Path, ticker: str, cache: dict) -> pd.DataFrame | None:
    if ticker not in cache:
        p = prices / "bars" / (ticker.replace("/", "_") + ".csv.gz")
        if not p.exists():
            cache[ticker] = None
        else:
            df = pd.read_csv(p, parse_dates=["date"])
            df = df[df.adjClose > 0].sort_values("date").reset_index(drop=True)
            cache[ticker] = df
    return cache[ticker]


def close(a, b) -> bool:
    if a is None or (isinstance(a, float) and math.isnan(a)):
        return b is None or (isinstance(b, float) and math.isnan(b))
    if b is None or (isinstance(b, float) and math.isnan(b)):
        return False
    return abs(a - b) <= TOL * max(1.0, abs(a), abs(b))


def price_feats(df: pd.DataFrame | None, entry) -> dict:
    keys = ["ret_5", "ret_20", "ret_60", "ret_120", "ret_250", "mom_12_1", "low_distance",
            "sessions_since_high", "vol_20", "vol_ratio", "max_ret_20",
            "dollar_volume_trend", "log_price"]
    out = {k: math.nan for k in keys}
    if df is None:
        return out
    h = df[df.date < entry]
    if h.empty:
        return out
    c = h.adjClose.to_numpy()
    n = len(c)
    out["log_price"] = math.log(h.close.iloc[-1]) if h.close.iloc[-1] > 0 else math.nan
    for k in (5, 20, 60, 120, 250):
        if n > k:
            out[f"ret_{k}"] = c[-1] / c[-1 - k] - 1
    if n > YEAR:
        out["mom_12_1"] = c[-21] / c[-1 - YEAR] - 1
    r = c[1:] / c[:-1] - 1
    if n >= 60:
        w = c[-YEAR:]
        out["low_distance"] = c[-1] / w.min() - 1
        last_peak = len(w) - 1 - np.argmax(w[::-1])
        out["sessions_since_high"] = len(w) - 1 - last_peak
        v20 = np.std(r[-20:], ddof=1) * math.sqrt(252)
        out["vol_20"] = v20
        k = min(n - 1, YEAR)
        vl = np.std(r[-k:], ddof=1) * math.sqrt(252)
        out["vol_ratio"] = v20 / vl if vl > 0 else math.nan
    if n > 20:
        out["max_ret_20"] = r[-20:].max()
    dv = (h.close.fillna(0) * h.volume.fillna(0)).to_numpy()[-120:]
    if len(dv) >= 60:
        a, b = np.median(dv[-20:]), np.median(dv)
        out["dollar_volume_trend"] = math.log(a / b) if a > 0 and b > 0 else math.nan
    return out


def market_feats(spy, iwm, entry) -> dict:
    s = spy[spy.date < entry].adjClose.to_numpy()
    out = {"spy_ret_20": s[-1] / s[-21] - 1, "spy_ret_250": s[-1] / s[-251] - 1,
           "spy_drawdown": s[-1] / s[-YEAR:].max() - 1,
           "spy_vol_20": np.std((s[1:] / s[:-1] - 1)[-20:], ddof=1) * math.sqrt(252)}
    i = iwm[iwm.date < entry].adjClose.to_numpy()
    out["iwm_minus_spy_60"] = i[-1] / i[-61] - 1 - (s[-1] / s[-61] - 1)
    return out


def activity_feats(ev: pd.DataFrame, by_firm: dict, row) -> dict:
    k = row.firm
    d = row.filing_date
    f = by_firm.get(k, ev.iloc[:0])
    before = f[f.filing_date < d]
    w90 = before[before.filing_date >= d - pd.Timedelta(days=90)]
    w180 = before[before.filing_date >= d - pd.Timedelta(days=180)]
    nb, ns = (w180.side == "buy").sum(), (w180.side == "sell").sum()
    mine = before[(before.filer_id == row.filer_id) & (before.side == "sell")
                  & (before.filing_date >= d - pd.Timedelta(days=365))]
    buys = before[before.side == "buy"]
    return {
        "firm_buyers_90": w90[w90.side == "buy"].filer_id.nunique(),
        "firm_sellers_90": w90[w90.side == "sell"].filer_id.nunique(),
        "firm_net_purchase_180": (nb - ns) / (nb + ns) if nb + ns else math.nan,
        "insider_sells_365": len(mine),
        "days_since_firm_buy": min((d - buys.filing_date.max()).days, 730) if len(buys) else 730,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--prices", required=True)
    ap.add_argument("--sample", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    run, prices = Path(a.run), Path(a.prices)
    bad = 0

    ev = pd.read_csv(run / "events-all.csv.gz", parse_dates=["filing_date"],
                     dtype={"issuer_cik": str})
    ev["firm"] = ev.issuer_cik.fillna("").str.lstrip("0")
    ev.loc[ev.firm == "", "firm"] = "ticker:" + ev.ticker
    by_firm = {k: g for k, g in ev.groupby("firm")}
    filer_buys = ev[ev.side == "buy"].groupby("filer_id").filing_date.apply(lambda s: np.sort(s.values))
    mkt = np.sort(ev[ev.side == "buy"].filing_date.values)

    for name in ("features-cap-floor.csv.gz", "features-all.csv.gz"):
        path = run / name
        if not path.exists():
            continue
        fr = pd.read_csv(path, parse_dates=["filing_date", "entry", "exit_20", "exit_60"],
                         float_precision="round_trip",
                         dtype={"issuer_cik": str, "role": str, "sector": str,
                                "insider_type": str})
        fr["firm"] = fr.issuer_cik.fillna("").str.lstrip("0")
        fr.loc[fr.firm == "", "firm"] = "ticker:" + fr.ticker
        smp = fr.sample(min(a.sample, len(fr)), random_state=a.seed)
        cache: dict = {}
        n_price = n_act = n_mkt = 0
        mism = {"price": 0, "market": 0, "activity": 0, "derived": 0}
        spy, iwm = bars(prices, "SPY", cache), bars(prices, "IWM", cache)
        mk = {}
        for row in smp.itertuples():
            pf = price_feats(bars(prices, row.ticker, cache), row.entry)
            for k, v in pf.items():
                if not close(getattr(row, k), v):
                    mism["price"] += 1
                    if mism["price"] <= 5:
                        print(f"  price {row.ticker} {row.entry.date()} {k}: run {getattr(row, k)} vs {v}")
            n_price += 1
            if row.entry not in mk:
                mk[row.entry] = market_feats(spy, iwm, row.entry)
                for k, v in mk[row.entry].items():
                    n_mkt += 1
                    if not close(getattr(row, k), v):
                        mism["market"] += 1
            af = activity_feats(ev, by_firm, row)
            fb = filer_buys.get(row.filer_id, np.array([], dtype="datetime64[ns]"))
            af["insider_prior_buys"] = int(np.searchsorted(fb, np.datetime64(row.filing_date), "left"))
            d = np.datetime64(row.filing_date)
            recent = np.searchsorted(mkt, d, "left") - np.searchsorted(mkt, d - np.timedelta64(30, "D"), "left")
            year = (np.searchsorted(mkt, d - np.timedelta64(30, "D"), "left")
                    - np.searchsorted(mkt, d - np.timedelta64(395, "D"), "left"))
            af["market_buying_30"] = recent / (year * 30 / 365) if year else math.nan
            for k, v in af.items():
                if not close(float(getattr(row, k)), float(v)):
                    mism["activity"] += 1
                    if mism["activity"] <= 5:
                        print(f"  activity {row.ticker} {row.filing_date.date()} {k}: run {getattr(row, k)} vs {v}")
            n_act += 1
            if not math.isnan(row.value_to_cap):
                if not close(row.value_to_cap, math.exp(row.log_value - row.log_market_cap)):
                    mism["derived"] += 1
            if row.month != row.filing_date.month:
                mism["derived"] += 1
        print(f"{name}: {n_price} rows (price, activity, derived), {n_mkt} market values; "
              f"mismatches {mism}")
        bad += sum(mism.values())

        # 5. purge and embargo
        summ = json.loads((run / "summary.json").read_text()) if (run / "summary.json").exists() else None
        if summ is None:
            continue
        cap = name == "features-cap-floor.csv.gz"
        for setup, h in summ["setups"].items():
            if ("cap floor" in setup) != cap:
                continue
            hz = 60 if "60 days" in setup else 20
            sc = run / ("scored-" + "".join(c if c.isalnum() else "-" for c in setup.lower())
                        .strip("-").replace("--", "-") + ".csv.gz")
            for y in sorted(int(k) for k in h.get("rounds", {})):
                start = pd.Timestamp(y, 1, 1) - pd.Timedelta(days=14)
                n = ((fr[f"exit_{hz}"] < start) & fr[f"net_{hz}"].notna()).sum()
                got = summ.get("folds", {}).get(setup, {}).get(str(y))
                if got is not None and got != n:
                    bad += 1
                    print(f"  purge {setup} {y}: run {got} vs {n}")
            if sc.exists():
                s = pd.read_csv(sc)
                top = s[s.above == 1].target.dropna()
                ok = (len(top) == h["above"] and close(top.mean(), h["top_mean"]))
                bad += not ok
                print(f"{setup}: above {len(top)} (run {h['above']}), net mean "
                      f"{100 * top.mean():.4f}% (run {100 * h['top_mean']:.4f}%) {'ok' if ok else 'MISMATCH'}")
    print(f"total mismatches: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
