"""Gradient-boosted models over the per-trade features (tradetracker.ml)."""

import math
import random
from datetime import date, timedelta

import numpy as np
import pytest
from test_scoring import Source, bars_from, ev, res, sig

from tradetracker.backtest import prices as bp
from tradetracker.features.calendar import NYSE
from tradetracker.ml import features as F
from tradetracker.ml import study as ML

pytest.importorskip("lightgbm")


# ---------------------------------------------------------------- features

def series(n=300, start=date(2020, 1, 2), seed=1):
    rng = random.Random(seed)
    closes, c = [], 50.0
    for _ in range(n):
        c *= 1 + rng.gauss(0, 0.02)
        closes.append(c)
    return bars_from(start, closes)


def test_price_features_by_hand():
    closes = [100.0] * 200 + [80.0] * 50 + [60.0, 66.0]
    b = bars_from(date(2020, 1, 2), closes)
    entry = NYSE.add(b.dates[-1], 1)
    f = F.price_features(b, entry)
    assert f["ret_5"] == pytest.approx(66 / 80 - 1)
    assert f["ret_250"] == pytest.approx(66 / 100 - 1)
    assert f["low_distance"] == pytest.approx(66 / 60 - 1)
    assert f["max_ret_20"] == pytest.approx(0.1)
    # The 52-week high is the last 100 of the first block, 52 sessions back.
    assert f["sessions_since_high"] == 52
    assert f["mom_12_1"] == pytest.approx(80 / 100 - 1)
    assert f["log_price"] == pytest.approx(math.log(66))
    assert f["dollar_volume_trend"] == pytest.approx(math.log(80 / 100))


def test_price_features_never_see_the_entry_day_or_later():
    b = series()
    entry = b.dates[250]
    f = F.price_features(b, entry)
    cut = bp.Bars([x for x in b.bars if x.date < entry])
    wrecked = bp.Bars([x if x.date < entry else bp.Bar(x.date, 1, 1, 1, 1, 1) for x in b.bars])
    assert F.price_features(cut, entry) == f
    assert F.price_features(wrecked, entry) == f


def test_short_history_gives_missing_values_not_errors():
    b = series(n=30)
    f = F.price_features(b, NYSE.add(b.dates[-1], 1))
    assert math.isnan(f["ret_60"]) and math.isnan(f["low_distance"]) and math.isnan(f["mom_12_1"])
    assert not math.isnan(f["ret_20"])
    assert all(math.isnan(v) for v in F.price_features(None, date(2020, 1, 1)).values())


def test_market_features_point_in_time():
    spy, iwm = series(seed=2), series(seed=3)
    entry = spy.dates[280]
    m = F.Market(spy, iwm).features(entry)
    cut = F.Market(bp.Bars([x for x in spy.bars if x.date < entry]),
                   bp.Bars([x for x in iwm.bars if x.date < entry])).features(entry)
    assert m == cut
    assert m["spy_drawdown"] <= 0
    i = spy.index_before(entry)
    assert m["spy_ret_20"] == pytest.approx(spy.bars[i].adj_close / spy.bars[i - 20].adj_close - 1)


def test_activity_counts_only_earlier_days():
    d = date(2024, 6, 10)
    evs = [
        ev(filer_id=1, side="buy", filing_date=d),                       # this event
        ev(filer_id=2, side="buy", filing_date=d),                       # same day: unknown
        ev(filer_id=3, side="buy", filing_date=d - timedelta(days=10)),
        ev(filer_id=3, side="buy", filing_date=d - timedelta(days=20)),  # same buyer twice
        ev(filer_id=4, side="sell", filing_date=d - timedelta(days=30)),
        ev(filer_id=1, side="sell", filing_date=d - timedelta(days=200)),
        ev(filer_id=5, side="sell", filing_date=d - timedelta(days=120)),
        ev(filer_id=6, side="buy", filing_date=d + timedelta(days=1)),    # later
        ev(filer_id=7, side="buy", filing_date=d - timedelta(days=5), issuer_cik="99"),
    ]
    a = F.Activity(evs)
    f = a.features(evs[0])
    assert f["firm_buyers_90"] == 1
    assert f["firm_sellers_90"] == 1
    # 180 days: buys by 3 (twice), sells by 4 and 5 -> (2 - 2) / 4
    assert f["firm_net_purchase_180"] == 0.0
    assert f["insider_sells_365"] == 1
    assert f["insider_prior_buys"] == 0
    assert f["days_since_firm_buy"] == 10
    other = a.features(evs[-1])
    assert other["firm_buyers_90"] == 0 and other["days_since_firm_buy"] == F.DAYS_SINCE_CAP


def test_market_buying_is_relative_to_the_previous_year():
    d = date(2024, 6, 10)
    evs = [ev(filer_id=i, filing_date=d - timedelta(days=40 + i)) for i in range(365)]
    evs += [ev(filer_id=1000 + i, filing_date=d - timedelta(days=1 + i % 29)) for i in range(60)]
    a = F.Activity(evs)
    # 60 buys in the last 30 days; the year before (days 30 to 395 back) holds the 356
    # filed 40 to 395 days back, the rest are older.
    assert a.market_buying(d) == pytest.approx(60 / (356 * 30 / 365))
    assert math.isnan(F.Activity([]).market_buying(d))


def test_event_features_and_ratios():
    e = ev(lag_days=4, n_lines=3, filing_date=date(2024, 3, 5))
    x = res(e, r20=0.01, cost=0.01)
    s = sig(value=1e6, market_cap=1e8, dollar_volume=2e6, role="CFO")
    f = F.event_features(x, s)
    assert f["value_to_cap"] == pytest.approx(0.01)
    assert f["value_to_dollar_volume"] == pytest.approx(0.5)
    assert f["lag_days"] == 4 and f["n_lines"] == 3 and f["month"] == 3 and f["cost"] == 0.01
    assert f["role"] == "CFO" and f["insider_type"] == "unclassified"
    assert math.isnan(F.event_features(x, sig(market_cap=None))["value_to_cap"])


def test_feature_lists_are_consistent():
    assert len(set(F.FEATURES)) == len(F.FEATURES)
    assert set(F.GROUPS) == set(F.FEATURES)
    b = series()
    entry = b.dates[260]
    e = ev(filing_date=NYSE.add(entry, -1), ticker="X")
    x = res(e, entry=entry)
    rows = F.build_rows([(x, sig(market_cap=1e9))], Source({"X": b, "SPY": b, "IWM": b}),
                        F.Activity([e]))
    assert set(rows[0].features) == set(F.FEATURES)


# ---------------------------------------------------------------- walk-forward

NOISE = [f for f in F.NUMERIC if f not in ("drawdown", "ret_20")]


def planted(seed=7, per_year=1500, years=range(2014, 2021), edge=0.04):
    """Returns are noise except buys 40%+ below the high AND falling over 20 days earn
    `edge`: an interaction a linear score of either input alone can't express."""
    rng = random.Random(seed)
    rows = []
    for y in years:
        for i in range(per_year):
            d = NYSE.on_or_before(date(y, 1, 5) + timedelta(days=int(340 * i / per_year)))
            dd, r20 = rng.uniform(-0.8, 0), rng.uniform(-0.3, 0.3)
            r = rng.gauss(0, 0.05) + (edge if dd <= -0.4 and r20 < 0 else 0)
            e = ev(filing_date=d, first_trade_date=d, last_trade_date=d, ticker=f"T{i}",
                   filer_id=i)
            x = res(e, r20=r)
            f = {k: rng.gauss(0, 1) for k in NOISE}
            f.update(drawdown=dd, ret_20=r20, role=rng.choice(["CEO", "director"]),
                     sector="SPY", insider_type="unclassified")
            rows.append(F.Row(x, sig(drawdown=dd), f, x.outcome.entry.year))
    return rows


FAST = ML.MLConfig("test", h=20, seeds=1)


def test_split_purges_and_embargoes():
    rows = planted(per_year=400)
    ys = ML.target(rows, FAST)
    for y in (2016, 2018, 2020):
        tr, fi, va = ML.split(rows, ys, y, 20, 14)
        start = date(y, 1, 1)
        assert all(ML.exit_day(rows[i], 20) < start - timedelta(days=14) for i in tr)
        assert all(rows[i].result.outcome.entry.year == y - 1 for i in va)
        assert all(ML.exit_day(rows[i], 20) < date(y - 1, 1, 1) - timedelta(days=14) for i in fi)
        assert set(fi) | set(va) <= set(tr) and not set(fi) & set(va)
        # Purge and embargo bite: some year y-1 buys exit too late to train on.
        late = [i for i, r in enumerate(rows) if r.year == y - 1 and i not in tr]
        assert late


def test_gbm_finds_a_planted_interaction():
    rows = planted()
    folds = ML.walk_forward(rows, FAST, first_test_year=2017)
    assert [f.year for f in folds] == [2017, 2018, 2019, 2020]
    above = [x for f in folds for x in f.scored if x.above]
    inside = [x for x in above if x.signals.drawdown <= -0.4]
    assert len(inside) / len(above) > 0.85
    mean_above = np.mean([x.r() for x in above])
    mean_all = np.mean([x.r() for f in folds for x in f.scored])
    assert mean_above - mean_all > 0.02
    # SHAP: the two planted features matter most, and contributions add up to the score.
    X, S, scored = ML.pooled_shap(folds)
    imp = ML.importance(X, S, [x.year for x in scored])
    assert {imp[0]["feature"], imp[1]["feature"]} == {"drawdown", "ret_20"}
    assert imp[[d["feature"] for d in imp].index("drawdown")]["direction"] < 0
    assert np.allclose(S.sum(axis=1), [x.score for x in scored], atol=1e-9)
    pairs = ML.interactions(folds[-1:], per_fold=200)
    assert {pairs[0][0], pairs[0][1]} == {"drawdown", "ret_20"}


def test_shuffled_labels_find_nothing():
    rows = planted()
    real = ML.walk_forward(rows, FAST, first_test_year=2017, explain=False)
    fake = ML.walk_forward(rows, FAST, first_test_year=2017, explain=False, shuffle_seed=1)
    gap = lambda fs: (np.mean([x.r() for f in fs for x in f.scored if x.above])
                      - np.mean([x.r() for f in fs for x in f.scored]))
    assert gap(real) > 0.02 and abs(gap(fake)) < 0.01


def test_later_returns_cannot_change_earlier_folds():
    rows = planted(per_year=900)
    base = ML.walk_forward(rows, FAST, first_test_year=2017, explain=False)
    for r in rows:
        if r.result.outcome.entry >= date(2018, 12, 1):
            r.result.outcome.legs[20]["stock"] = random.Random(id(r)).gauss(0, 1)
    again = ML.walk_forward(rows, FAST, first_test_year=2017, explain=False)
    for a, b in zip(base, again):
        if a.year <= 2019:
            assert [x.score for x in a.scored] == [x.score for x in b.scored]
            assert a.cutoff == b.cutoff
    assert [x.score for x in base[-1].scored] != [x.score for x in again[-1].scored]


def test_rank_target_and_shuffles():
    rows = planted(per_year=200, years=range(2015, 2017))
    ys = ML.target(rows, FAST)
    lab = ML.train_labels(rows, ys, ML.MLConfig("r", target="rank"))
    assert lab.min() >= -0.5 and lab.max() <= 0.5
    # Within a month the label orders like the return.
    jan = [i for i, r in enumerate(rows) if r.result.outcome.entry.month == 3 and r.year == 2015]
    assert np.argsort(lab[jan]).tolist() == np.argsort(np.array(ys)[jan]).tolist()
    w = ML.train_labels(rows, ys, FAST)
    assert w.max() <= np.quantile(ys, 0.99) + 1e-12
    sh = ML.shuffle_within_year(rows, w, 3)
    for y in (2015, 2016):
        idx = [i for i, r in enumerate(rows) if r.year == y]
        assert sorted(sh[idx]) == sorted(w[idx]) and list(sh[idx]) != list(w[idx])


def test_setups_and_placebo_table():
    names = [c.name for c in ML.SETUPS]
    assert len(set(names)) == 8 and ML.PRIMARY.h == 60 and ML.PRIMARY.min_cap == 50e6
    assert ML.PRIMARY.baseline().components == ("drawdown",)
    real = {"top_mean": 0.02, "rank_mean": 0.01, "mean_ic": 0.05, "ir": 0.4}
    shuffles = [{"top_mean": v / 100, "rank_mean": 0.0, "mean_ic": 0.0, "ir": None}
                for v in range(-3, 3)]
    text, summ = ML.placebo_text(real, shuffles)
    assert summ["top_mean"]["beaten"] == 5      # a tie (+2%) does not count
    assert "ir" not in summ
    assert "real beats" in text


def test_dependence_and_grids_render():
    rows = planted(per_year=900)
    folds = ML.walk_forward(rows, FAST, first_test_year=2019)
    X, S, scored = ML.pooled_shap(folds)
    t = ML.dependence(X, S, scored, "drawdown", 20, "spy")
    assert t.count("\n") >= 6
    assert "role" in ML.dependence(X, S, scored, "role", 20, "spy")
    g = ML.interaction_grid(scored, X, "drawdown", "ret_20", 20, "spy")
    assert "Terciles" in g
    w, n, _ = ML.year_stability(X, scored, "drawdown", 20, "spy", high_is_good=False)
    assert n == 2 and w == 2


def test_feature_and_event_files_round_trip(tmp_path):
    import pandas as pd
    rows = planted(per_year=50, years=range(2016, 2018))
    ML.write_rows(rows, tmp_path / "f.csv.gz")
    df = pd.read_csv(tmp_path / "f.csv.gz", float_precision="round_trip")
    assert len(df) == len(rows) and list(df.columns) == ML.ROW_META + F.FEATURES
    assert df.drawdown.tolist() == [r.features["drawdown"] for r in rows]
    assert df.net_20.notna().all() and df.net_60.isna().all()
    ML.write_events([r.result.event for r in rows], tmp_path / "e.csv.gz")
    assert len(pd.read_csv(tmp_path / "e.csv.gz")) == len(rows)


def test_cli_has_ml_study(capsys):
    from tradetracker import cli
    with pytest.raises(SystemExit) as e:
        cli.main(["ml-study", "--help"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    assert "--placebos" in out and "--cache" in out and "--quick" in out
