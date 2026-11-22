"""Tests for the correlation study (tradetracker.correlation)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from tradetracker import correlation as C


def bars_frame(closes, start="2020-01-01", raw=None):
    dates = pd.bdate_range(start, periods=len(closes))
    return pd.DataFrame({"date": dates, "adjClose": np.asarray(closes, float),
                         "close": np.asarray(raw if raw is not None else closes, float)})


def spy_frame(n, start="2020-01-01"):
    return bars_frame(np.full(n, 100.0), start)


# ------------------------------------------------------------- statistics


def test_spearman_matches_known_values():
    assert C.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert C.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    # monotone transform doesn't change a rank correlation
    x = np.random.default_rng(0).normal(size=200)
    y = x + np.random.default_rng(1).normal(size=200)
    assert C.spearman(x, y) == pytest.approx(C.spearman(np.exp(x), y ** 3))
    scipy = pytest.importorskip("scipy.stats")
    assert C.spearman(x, y) == pytest.approx(scipy.spearmanr(x, y).statistic, abs=1e-12)


def test_spearman_ignores_missing_and_constant():
    assert C.spearman([1, 2, None, 4], [1, 2, 99, 4]) == pytest.approx(1.0)
    assert math.isnan(C.spearman([1, 1, 1], [1, 2, 3]))
    assert math.isnan(C.spearman([1, 2], [1, 2]))


def test_newey_west_without_autocorrelation_is_close_to_plain_t():
    x = np.random.default_rng(3).normal(0.1, 1, 5000)
    m, t = C.newey_west_t(x, 0)
    plain = x.mean() / (x.std(ddof=0) / math.sqrt(len(x)))
    assert m == pytest.approx(x.mean())
    assert t == pytest.approx(plain, rel=1e-9)


def test_newey_west_widens_for_overlapping_series():
    rng = np.random.default_rng(4)
    e = rng.normal(size=3000)
    x = 0.05 + np.convolve(e, np.ones(3), mode="valid")  # MA(2): positive autocorrelation
    _, t0 = C.newey_west_t(x, 0)
    _, t3 = C.newey_west_t(x, 3)
    assert abs(t3) < abs(t0)


def test_clustered_mean_singletons_and_one_big_cluster():
    y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    m, se = C.clustered_mean(y, pd.Series(range(6)))
    n = len(y)
    expected = math.sqrt(((y - y.mean()) ** 2).sum() / n ** 2 * n / (n - 1))
    assert m == pytest.approx(3.5)
    assert se == pytest.approx(expected)  # = the usual standard error
    # perfectly correlated within clusters: SE grows
    _, se2 = C.clustered_mean(y, pd.Series([0, 0, 0, 1, 1, 1]))
    assert se2 > se


def test_winsorize_clips_tails_only():
    s = pd.Series(list(range(1, 101)) + [10_000.0])
    w = C.winsorize(s)
    assert w.max() < 200 and w.iloc[50] == s.iloc[50]


def test_monthly_ic_needs_enough_events_per_month():
    rng = np.random.default_rng(5)
    rows = []
    for m in range(24):
        x = rng.normal(size=40)
        rows.append(pd.DataFrame({"x": x, "r": x + rng.normal(size=40),
                                  "month": pd.Period("2020-01", "M") + m}))
    df = pd.concat(rows)
    ic = C.monthly_ic(df, "x", "r", 20)
    assert ic.months == 24 and ic.monthly_mean > 0.5 and ic.t > 5
    assert ic.share_positive == 1.0
    small = df.groupby("month").head(C.MIN_MONTH_N - 1)
    assert C.monthly_ic(small, "x", "r", 20).months == 0


def test_bucket_labels_fixed_and_quintiles():
    df = pd.DataFrame({"lag_days": [0, 1, 2, 3, 4, 5, 40],
                       "cluster_count": [1, 2, 3, 4, 5, 9, 1]})
    lab, order = C.bucket_labels(df, "lag_days", "lag")
    assert list(lab) == ["0-1 days", "0-1 days", "2 days", "3-4 days", "3-4 days",
                         "5+ days", "5+ days"]
    lab, _ = C.bucket_labels(df, "cluster_count", "cluster")
    assert list(lab) == ["alone", "2 insiders", "3-4 insiders", "3-4 insiders",
                         "5+ insiders", "5+ insiders", "alone"]
    q = pd.DataFrame({"x": np.arange(100.0)})
    lab, names = C.bucket_labels(q, "x", "q")
    assert names == ["Q1", "Q2", "Q3", "Q4", "Q5"]
    assert lab.value_counts().to_dict() == {f"Q{i}": 20 for i in range(1, 6)}


def test_bucket_table_counts_and_net():
    df = pd.DataFrame({"x": np.arange(100.0), "r": np.arange(100.0) / 1000,
                       "cost": 0.01, "month": ["2020-01"] * 50 + ["2020-02"] * 50})
    df["r_w"] = df["r"]
    t = C.bucket_table(df, "x", "q", "r")
    assert list(t["n"]) == [20] * 5
    assert t["wmean"].is_monotonic_increasing
    tn = C.bucket_table(df, "x", "q", "r", net=True)
    assert np.allclose(t["wmean"] - tn["wmean"], 0.01)


def test_fama_macbeth_recovers_a_planted_slope():
    rng = np.random.default_rng(9)
    rows = []
    for m in range(36):
        n = 200
        a, b = rng.uniform(size=n), rng.uniform(size=n)
        r = 0.02 * a + 0.0 * b + rng.normal(0, 0.01, n)
        rows.append(pd.DataFrame({"a": a, "b": b, "r": r,
                                  "month": pd.Period("2018-01", "M") + m}))
    out = C.fama_macbeth(pd.concat(rows), ["a", "b"], "r", 20).set_index("feature")
    assert out.at["a", "slope"] == pytest.approx(0.02, abs=0.002)
    assert out.at["a", "t"] > 10
    assert abs(out.at["b", "t"]) < 3


# ------------------------------------------------------------- events


def raw_events():
    base = {"filer_id": 1, "issuer_cik": "320193", "ticker": "AAA", "side": "buy",
            "filing_date": "2021-03-01", "entry": "2021-03-02", "status": "ok",
            "is_10b5_1": 0, "entry_open": 10.0, "dollar_volume": 1e6, "role": "CEO",
            "value": 1e5, "drift_pct": 2.5, "cost": 0.004, "lag_days": 1,
            "cluster_count": 1}
    for h in C.HORIZONS:
        base.update({f"stock_{h}": 0.05, f"spy_{h}": 0.02, f"iwm_{h}": 0.03})
    rows = [dict(base),
            dict(base, status="no_prices"),
            dict(base, is_10b5_1=1),
            dict(base, entry_open=1.5),
            dict(base, dollar_volume=50_000),
            dict(base, role=None, side="sell")]
    return pd.DataFrame(rows)


def test_prepare_events_filters_like_the_backtest():
    e = C.prepare_events(raw_events())
    assert len(e) == 2
    assert list(e["side"]) == ["buy", "sell"]
    assert e["role"].iloc[1] == "unknown"
    assert e["adj_20"].iloc[0] == pytest.approx(0.03)
    assert e["iwm_adj_60"].iloc[0] == pytest.approx(0.02)
    assert e["drift"].iloc[0] == pytest.approx(0.025)
    assert e["month"].iloc[0] == pd.Period("2021-03", "M")


# ------------------------------------------------------------- price features


def one_event(entry):
    return pd.DataFrame({"entry": [pd.Timestamp(entry)]})


def test_drawdown_and_prior_return_use_bars_before_entry_only():
    closes = list(np.linspace(50, 100, 200)) + list(np.linspace(100, 60, 100))
    bars = bars_frame(closes)
    entry = bars["date"].iloc[250]
    f, _ = C.price_features(one_event(entry), bars, spy_frame(len(bars)))
    last = closes[249]
    assert f["drawdown"].iloc[0] == pytest.approx(last / 100 - 1)
    assert f["pre_ret_20"].iloc[0] == pytest.approx(last / closes[229] - 1)
    # a crash on and after the entry day doesn't move the features
    crashed = bars.copy()
    crashed.loc[250:, "adjClose"] = 1.0
    crashed.loc[250:, "close"] = 1.0
    g, _ = C.price_features(one_event(entry), crashed, spy_frame(len(bars)))
    pd.testing.assert_frame_equal(f, g)


def test_drawdown_window_is_250_sessions_and_needs_history():
    closes = [200.0] + [100.0] * 299         # the old peak drops out of the window
    bars = bars_frame(closes)
    f, _ = C.price_features(one_event(bars["date"].iloc[260]), bars, spy_frame(300))
    assert f["drawdown"].iloc[0] == pytest.approx(0.0)
    f, _ = C.price_features(one_event(bars["date"].iloc[200]), bars, spy_frame(300))
    assert f["drawdown"].iloc[0] == pytest.approx(-0.5)
    f, _ = C.price_features(one_event(bars["date"].iloc[30]), bars, spy_frame(300))
    assert math.isnan(f["drawdown"].iloc[0])     # under 60 sessions of history
    assert not math.isnan(f["pre_ret_20"].iloc[0])


def test_market_cap_point_in_time_and_split_adjusted():
    n = 400
    raw = np.full(n, 50.0)
    raw[300:] = 25.0                         # 2-for-1 split at bar 300
    adj = np.full(n, 25.0)                   # adjusted series is flat
    bars = bars_frame(adj, raw=raw)
    d = bars["date"]
    shares = (np.array([d.iloc[10].to_datetime64(), d.iloc[330].to_datetime64()]),
              np.array([1_000.0, 2_000.0]))
    entry = d.iloc[350]                      # the count dated at bar 330 isn't public yet
    f, _ = C.price_features(one_event(entry), bars, spy_frame(n), shares)
    # 1,000 pre-split shares = 2,000 now, at $25
    assert f["market_cap"].iloc[0] == pytest.approx(2_000 * 25.0)
    later = d.iloc[330] + pd.Timedelta(days=C.SHARES_LAG_DAYS + 3)
    f, _ = C.price_features(one_event(later), bars_frame(np.full(500, 25.0)),
                            spy_frame(500), shares)
    assert f["market_cap"].iloc[0] == pytest.approx(2_000 * 25.0)
    f, _ = C.price_features(one_event(d.iloc[50]), bars, spy_frame(n), shares)
    assert math.isnan(f["market_cap"].iloc[0])  # no count public 100 days before


def test_event_paths_are_excess_over_spy_from_day_zero():
    n = 300
    stock = 100 * 1.01 ** np.arange(n)
    spy = spy_frame(n)
    bars = bars_frame(stock)
    entry = bars["date"].iloc[150]
    _, P = C.price_features(one_event(entry), bars, spy, paths=True)
    w = C.PATH_SESSIONS
    assert P.shape == (1, 2 * w + 1)
    assert P[0, w] == pytest.approx(0.0)
    assert P[0, w + 1] == pytest.approx(0.01)
    assert P[0, 0] == pytest.approx(1.01 ** -w - 1)
    # near the end of the bars the path is NaN, not wrapped
    _, P = C.price_features(one_event(bars["date"].iloc[n - 5]), bars, spy, paths=True)
    assert np.isnan(P[0, -1]) and not np.isnan(P[0, w])


def test_add_price_features_uses_each_events_own_issuer(tmp_path):
    # one ticker reused by two companies: each event gets its own company's count
    n = 400
    bars = bars_frame(np.full(n, 10.0))
    bars["adjOpen"] = bars["adjClose"]
    bars.to_csv(tmp_path / "XYZ.csv.gz", index=False)
    spy_frame(n).to_csv(tmp_path / "SPY.csv.gz", index=False)
    d = bars["date"]
    pd.DataFrame({"cik": ["1", "2"], "end": [d.iloc[5].date(), d.iloc[5].date()],
                  "shares": [100.0, 300.0], "accn": "", "frame": ""}).to_csv(
        tmp_path / "shares.csv", index=False)
    ev = pd.DataFrame({"ticker": ["XYZ", "XYZ"], "issuer_cik": ["1", "2"],
                       "entry": [d.iloc[300], d.iloc[310]], "value": [10.0, 30.0]})
    e, _ = C.add_price_features(ev, tmp_path, tmp_path / "shares.csv")
    assert list(e["market_cap"]) == [1_000.0, 3_000.0]
    assert list(e["value_to_cap"]) == [0.01, 0.01]


def test_study_script_end_to_end(tmp_path, monkeypatch):
    """The study and the independent cross-check agree on synthetic data."""
    monkeypatch.setattr(C, "MIN_FEATURE_N", 100)
    import sys
    from pathlib import Path
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    import correlation_crosscheck
    import correlation_study

    rng = np.random.default_rng(21)
    bars_dir = tmp_path / "bars"
    bars_dir.mkdir()
    n = 700
    dates = pd.bdate_range("2019-01-01", periods=n)
    spy = pd.DataFrame({"date": dates, "open": 100.0, "close": 100.0, "adjOpen": 100.0,
                        "adjClose": 100 * np.cumprod(1 + rng.normal(0, 0.01, n))})
    spy.to_csv(bars_dir / "SPY.csv.gz", index=False)
    rows, shares = [], []
    for k in range(40):
        t = f"T{k}"
        px = 20 * np.cumprod(1 + rng.normal(0, 0.02, n))
        pd.DataFrame({"date": dates, "open": px, "close": px, "adjOpen": px,
                      "adjClose": px}).to_csv(bars_dir / f"{t}.csv.gz", index=False)
        shares.append({"cik": str(1000 + k), "end": "2018-06-30", "shares": 1e7,
                       "accn": "", "frame": ""})
        for i in rng.choice(np.arange(300, n - 70), 40, replace=False):
            side = "buy" if rng.uniform() < 0.5 else "sell"
            row = {"filer_id": int(rng.integers(1, 50)), "issuer_cik": str(1000 + k),
                   "ticker": t, "side": side,
                   "filing_date": dates[i - 1].date().isoformat(),
                   "entry": dates[i].date().isoformat(), "status": "ok", "is_10b5_1": 0,
                   "entry_open": px[i], "dollar_volume": 1e6,
                   "role": rng.choice(["CEO", "CFO", "officer", "director", "10% owner"]),
                   "value": float(rng.lognormal(11, 1)), "drift_pct": rng.normal(0, 3),
                   "cost": 0.004, "lag_days": int(rng.integers(0, 6)),
                   "cluster_count": int(rng.integers(1, 6)), "status_note": ""}
            for h in C.HORIZONS:
                j = i + h - 1
                row[f"exit_{h}"] = dates[j].date().isoformat()
                row[f"stock_{h}"] = px[j] / px[i] - 1
                row[f"spy_{h}"] = spy["adjClose"][j] / spy["adjClose"][i] - 1
                row[f"iwm_{h}"] = row[f"spy_{h}"]
            rows.append(row)
    ev = pd.DataFrame(rows)
    ev.to_csv(tmp_path / "events.csv.gz", index=False)
    pd.DataFrame(shares).to_csv(tmp_path / "shares.csv", index=False)
    feats = ev[ev["side"] == "buy"][["filer_id", "issuer_cik", "filing_date"]].copy()
    feats["stake_change"] = rng.uniform(0, 1, len(feats))
    feats["drawdown"] = np.nan
    feats["log_market_cap"] = np.nan
    feats.to_csv(tmp_path / "features.csv.gz", index=False)

    common = ["--events", str(tmp_path / "events.csv.gz"), "--bars", str(bars_dir),
              "--shares", str(tmp_path / "shares.csv")]
    out, cache = tmp_path / "out", tmp_path / "cache"
    res = correlation_study.main([*common, "--features", str(tmp_path / "features.csv.gz"),
                                  "--out", str(out), "--cache", str(cache), "--no-charts"])
    assert set(res["baseline"]["side"]) == {"buy", "sell"}
    assert (out / "tables" / "ic.csv").exists() and (out / "summary.json").exists()
    assert len(res["event_paths"]) == 2 * C.PATH_SESSIONS + 1
    correlation_crosscheck.FAILS.clear()
    status = correlation_crosscheck.main(["--study", str(out), "--cache", str(cache),
                                          *common, "--sample", "60"])
    failed = [f for f in correlation_crosscheck.FAILS
              if "phase 4e" not in f]       # the synthetic file has no 4e features
    assert failed == [], failed
    assert status in (0, 1)
