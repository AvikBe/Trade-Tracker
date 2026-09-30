"""Phase 4 scoring: signals, the model fit, the walk-forward study, portfolio and ranking."""

import json
import math
import random
from datetime import date, timedelta

import pytest

from tradetracker.backtest import prices as bp
from tradetracker.backtest.events import Event
from tradetracker.backtest.returns import Outcome
from tradetracker.backtest.study import Result
from tradetracker.features.calendar import NYSE
from tradetracker.scoring import model as M
from tradetracker.scoring import rank as R
from tradetracker.scoring import signals as S
from tradetracker.scoring import study as ST

# ---------------------------------------------------------------- helpers


def days_from(start: date, n: int) -> list[date]:
    d = start if NYSE.is_business_day(start) else NYSE.add(start, 1)
    out = []
    for _ in range(n):
        out.append(d)
        d = NYSE.add(d, 1)
    return out


def bars_from(start: date, closes: list[float], opens: list[float] | None = None,
              volume: float = 1e6) -> bp.Bars:
    ds = days_from(start, len(closes))
    return bp.Bars([bp.Bar(d, (opens or closes)[i], c, (opens or closes)[i], c, volume)
                    for i, (d, c) in enumerate(zip(ds, closes))])


class Source:
    name = "test"

    def __init__(self, bars: dict):
        self.bars = bars

    def get(self, t):
        return self.bars.get(t)


def ev(**kw) -> Event:
    base = dict(filer_id=1, issuer_cik="1", ticker="X", side="buy",
                filing_date=date(2024, 1, 10), accepted_at=None,
                first_trade_date=date(2024, 1, 8), last_trade_date=date(2024, 1, 8),
                lag_days=2, lag_ratio=1.0, value=10_000.0, shares=1000.0, role="CEO",
                cluster_count=1)
    base.update(kw)
    return Event(**base)


def res(e: Event, r20: float | None = 0.0, entry: date | None = None, cost: float = 0.004,
        drift_pct: float | None = 0.0, dv: float = 5e6, open_: float = 10.0,
        status: str = "ok") -> Result:
    entry = entry or e.entry_date
    o = Outcome(status, entry=entry, entry_open=open_, drift_pct=drift_pct,
                dollar_volume=dv, cost=cost, benchmark="SPY")
    if r20 is not None:
        o.legs[20] = {"stock": r20, "spy": 0.0, "sector": 0.0, "iwm": 0.0,
                      "exit": NYSE.add(entry, 19), "flags": []}
    return Result(e, o, "SPY")


def sig(**kw) -> S.Signals:
    base = dict(role="CEO", value=100_000.0, stake_change=0.1, cluster_count=1,
                repeat_buys=0, drawdown=-0.2, track_record=None, track_n=0,
                dollar_volume=5e6, drift=0.0)
    base.update(kw)
    return S.Signals(**base)


# ---------------------------------------------------------------- signals

@pytest.mark.parametrize("shares, before, expected", [
    (100, 400, 0.25),
    (100, 0, S.NEW_STAKE),        # held nothing before
    (100, -5, S.NEW_STAKE),       # rounding in the filing: treat as a new position
    (100, None, None),            # holdings not reported
    (0, 100, None),
    (5000, 1, S.NEW_STAKE),       # capped
])
def test_stake_change(shares, before, expected):
    assert S.stake_change(shares, before) == expected


def test_holdings_before_sum_the_accounts_that_bought():
    e = ev()
    e.held_before = {"self": 1000.0, "trust": 500.0}
    assert e.shares_held_before == 1500.0
    e.holdings_missing = True
    assert e.shares_held_before is None


def test_drawdown_uses_only_closes_before_entry():
    closes = [100.0] * 100 + [150.0] + [120.0] * 99 + [60.0] * 50 + [70.0] * 10
    b = bars_from(date(2023, 1, 3), closes)
    src = Source({"X": b})
    entry = b.dates[250]
    assert S.drawdown(src, "X", entry) == pytest.approx(60 / 150 - 1)
    # Changing the entry day's and later bars changes nothing.
    scrambled = bp.Bars(b.bars[:250] + [bp.Bar(d, 999, 999, 999, 999, 1) for d in b.dates[250:]])
    assert S.drawdown(Source({"X": scrambled}), "X", entry) == pytest.approx(60 / 150 - 1)


def test_drawdown_window_is_250_sessions_and_needs_60():
    closes = [200.0] + [100.0] * 300
    b = bars_from(date(2022, 1, 3), closes)
    src = Source({"X": b})
    assert S.drawdown(src, "X", b.dates[250]) == pytest.approx(100 / 200 - 1)  # peak in window
    assert S.drawdown(src, "X", b.dates[251]) == 0.0                           # peak just left
    assert S.drawdown(src, "X", b.dates[59]) is None                           # 59 sessions
    assert S.drawdown(src, "X", b.dates[60]) is not None
    assert S.drawdown(src, "Y", b.dates[60]) is None


def test_repeat_buys_count_earlier_buys_in_the_same_stock_within_two_years():
    rs = [
        res(ev(filing_date=date(2021, 1, 4))),                  # > 730 days before
        res(ev(filing_date=date(2022, 6, 1))),
        res(ev(filing_date=date(2023, 6, 1))),
        res(ev(filing_date=date(2023, 7, 1), side="sell")),     # sells don't count
        res(ev(filing_date=date(2023, 8, 1), ticker="Y")),      # other stock
        res(ev(filing_date=date(2023, 9, 1), filer_id=2)),      # other insider
        res(ev(filing_date=date(2024, 1, 10))),                 # this one
        res(ev(filing_date=date(2024, 2, 1))),                  # later: never counted
    ]
    sigs = S.build(rs, Source({}))
    assert sigs[6].repeat_buys == 2
    assert sigs[0].repeat_buys == 0
    assert sigs[4].repeat_buys == 0 and sigs[5].repeat_buys == 0


def test_track_record_uses_only_buys_that_exited_before_entry():
    a = res(ev(filing_date=date(2023, 1, 10)), r20=0.10)
    b = res(ev(filing_date=date(2023, 3, 10)), r20=-0.04)
    plan = res(ev(filing_date=date(2023, 4, 10), is_10b5_1=True), r20=0.50)  # excluded
    other = res(ev(filing_date=date(2023, 4, 10), filer_id=9), r20=0.50)     # other filer
    sell = res(ev(filing_date=date(2023, 4, 12), side="sell"), r20=0.50)     # not a buy
    # c's exit is the next event's entry day: a close after the open, so not yet known.
    c = res(ev(filing_date=date(2023, 6, 1)), r20=0.30)
    now = res(ev(filing_date=date(2023, 6, 1) + timedelta(days=40)),
              entry=c.outcome.legs[20]["exit"])
    sigs = S.build([a, b, plan, other, sell, c, now], Source({}))
    assert sigs[-1].track_n == 2
    assert sigs[-1].track_record == pytest.approx((0.10 - 0.04) / 2)
    assert sigs[1].track_record is None and sigs[1].track_n == 1   # needs 2


def test_track_record_for_an_unmatured_event_uses_its_planned_entry():
    a = res(ev(filing_date=date(2023, 1, 10)), r20=0.10)
    b = res(ev(filing_date=date(2023, 3, 10)), r20=0.20)
    live = res(ev(filing_date=date(2024, 1, 10)), r20=None, status="not_matured")
    live.outcome.entry = None
    sigs = S.build([a, b, live], Source({}))
    assert sigs[-1].track_record == pytest.approx(0.15)


# ---------------------------------------------------------------- buckets and model

@pytest.mark.parametrize("comp, kw, expected", [
    ("value", {"value": 9_999.99}, "<$10k"),
    ("value", {"value": 10_000}, "$10k-50k"),
    ("value", {"value": 1_000_000}, ">=$1M"),
    ("stake", {"stake_change": None}, "unknown"),
    ("stake", {"stake_change": 0.05}, "5-20%"),
    ("stake", {"stake_change": 1.0}, ">=100%"),
    ("stake", {"stake_change": S.NEW_STAKE}, "new stake"),
    ("cluster", {"cluster_count": 0}, "1"),
    ("cluster", {"cluster_count": 2}, "2"),
    ("cluster", {"cluster_count": 7}, "3+"),
    ("repeat", {"repeat_buys": 3}, "2-3"),
    ("repeat", {"repeat_buys": 4}, "4+"),
    ("drawdown", {"drawdown": -0.45}, "-45 to -25%"),
    ("drawdown", {"drawdown": -0.4501}, "<=-45%"),
    ("drawdown", {"drawdown": None}, "unknown"),
    ("drawdown", {"drawdown": 0.0}, "within 10%"),
    ("track", {"track_record": None}, "none yet"),
    ("track", {"track_record": 0.05}, ">= 5%"),
    ("role", {"role": None}, "none"),
    ("liquidity", {"dollar_volume": None}, "<$1M/day"),
    ("liquidity", {"dollar_volume": 5e7}, ">=$50M/day"),
])
def test_bucket_edges(comp, kw, expected):
    b = M.bucket(comp, sig(**kw))
    assert b == expected
    assert b in M.COMPONENTS[comp][1]


def test_nnls_recovers_non_negative_weights_and_clips_negative_ones():
    rng = random.Random(1)
    X = [[rng.gauss(0, 1), rng.gauss(0, 1), rng.gauss(0, 1)] for _ in range(2000)]
    y = [2 * a + 0.5 * b - 1.0 * c for a, b, c in X]
    w = M.nnls(X, y)
    assert w[0] == pytest.approx(2.0, abs=0.05)
    assert w[1] == pytest.approx(0.5, abs=0.05)
    assert w[2] == 0.0
    assert M.nnls([[0.0], [0.0]], [1.0, 2.0]) == [0.0]


def planted(n=6000, seed=3, ceo_edge=0.03, drift_edge=0.0):
    """CEO buys earn ceo_edge more; each 1% of drift costs drift_edge; noise elsewhere."""
    rng = random.Random(seed)
    sigs, ys = [], []
    for _ in range(n):
        s = sig(role=rng.choice(["CEO", "director", "officer"]),
                value=rng.choice([5e3, 3e4, 1e5, 5e5, 2e6]),
                cluster_count=rng.choice([1, 2, 3]),
                drift=rng.uniform(-0.1, 0.2))
        y = rng.gauss(0, 0.05) + (ceo_edge if s.role == "CEO" else 0) - drift_edge * 100 * s.drift
        sigs.append(s)
        ys.append(y)
    return sigs, ys


def test_fit_finds_a_planted_role_effect_and_q_is_between_0_and_1():
    sigs, ys = planted()
    m = M.fit(sigs, ys, components=["role", "value", "cluster"], beta=0.0)
    t = m.tables["role"]
    assert t["CEO"] > t["director"] and t["CEO"] > t["officer"]
    assert t["CEO"] == pytest.approx(0.02, abs=0.006)        # 3% edge minus the mean, shrunk
    spread = {c: m.weights[c] * (max(t.values()) - min(t.values())) for c, t in m.tables.items()}
    assert m.weights["role"] > 0.5
    assert spread["role"] > 3 * max(spread["value"], spread["cluster"])
    assert all(0 <= w <= 1 for w in m.weights.values())
    qs = [m.quality(s) for s in sigs]
    assert min(qs) >= 0 and max(qs) <= 1
    assert m.quality(sig(role="CEO", value=5e3, cluster_count=1)) > \
        m.quality(sig(role="director", value=5e3, cluster_count=1))


def test_small_buckets_are_shrunk_toward_zero():
    sigs = [sig(role="director")] * 5000 + [sig(role="CFO")] * 100
    ys = [0.0] * 5000 + [0.10] * 100
    m = M.fit(sigs, ys, components=["role"], beta=0.0)
    raw = 0.10 - (0.10 * 100 / 5100)
    assert m.tables["role"]["CFO"] == pytest.approx(raw * 100 / (100 + M.SHRINK_N))


def test_fit_winsorizes_the_target():
    sigs = [sig(role="director")] * 999 + [sig(role="CFO")]
    ys = [0.0] * 999 + [1000.0]      # one 1,000x stock
    m = M.fit(sigs, ys, components=["role"], beta=0.0)
    assert m.tables["role"]["CFO"] < 0.001


def test_beta_is_picked_from_the_grid_when_drift_hurts():
    sigs, ys = planted(drift_edge=0.004)
    m = M.fit(sigs, ys, components=["role"])
    assert m.beta in M.BETA_GRID and m.beta > 0
    sigs, ys = planted(drift_edge=-0.004)   # run-ups help: no penalty
    assert M.fit(sigs, ys, components=["role"]).beta == 0.0


def test_threshold_is_the_training_top_decile():
    sigs, ys = planted()
    m = M.fit(sigs, ys, components=["role", "value"])
    scores = [m.score(s) for s in sigs]
    share = sum(v >= m.threshold for v in scores) / len(scores)
    assert 0.10 <= share <= 0.45   # ties at the cutoff can only add events


def test_drift_factor_and_decay():
    m = M.Model(tables={}, counts={}, weights={}, beta=5.0, half_life=10)
    assert m.drift_factor(None) == 1.0
    assert m.drift_factor(0.1) == pytest.approx(0.5)
    assert m.drift_factor(0.25) == 0.0
    assert m.drift_factor(-0.5) == pytest.approx(1 + 5 * 0.2)   # bonus floored at -20%
    assert m.decay(0) == 1.0
    assert m.decay(10) == pytest.approx(0.5)
    assert m.decay(20) == pytest.approx(0.25)


def test_model_json_round_trip_scores_identically():
    sigs, ys = planted()
    m = M.fit(sigs, ys)
    m2 = M.Model.from_json(m.to_json())
    assert [m2.score(s, 3) for s in sigs[:200]] == [m.score(s, 3) for s in sigs[:200]]
    bad = json.loads(m.to_json())
    bad["version"] = 99
    with pytest.raises(ValueError):
        M.Model.from_json(json.dumps(bad))


def test_fit_rejects_mismatched_input():
    with pytest.raises(ValueError):
        M.fit([sig()], [])


def test_reason_line():
    s = sig(role="CFO", value=3_100_000, stake_change=0.35, cluster_count=3, repeat_buys=0,
            drawdown=-0.48, track_record=0.021, track_n=4, drift=-0.012)
    assert M.reason(s, "XYZ", 2) == (
        "CFO bought $3.1M of XYZ, stake +35%, 3 insiders buying within 14 days, first buy in "
        "2 years, stock 48% below its 52-week high, past buys +2.1% vs SPY (n=4), -1.2% since "
        "the trade, filed 2 trading days ago")
    s = sig(role=None, value=900, stake_change=S.NEW_STAKE, repeat_buys=2, drawdown=0.0,
            drift=None)
    assert M.reason(s, "A") == ("Insider bought $900 of A, a new stake, 2 earlier buys in 2 "
                                "years, stock at its 52-week high")
    assert "1 earlier buy in 2 years," in M.reason(sig(repeat_buys=1), "A")


# ---------------------------------------------------------------- study

def synthetic_history(seed=5, per_year=400, years=range(2015, 2021), edge=0.03):
    """CEO buys earn `edge`; everything else is noise. Returns (results, signals)."""
    rng = random.Random(seed)
    rs, ss = [], []
    for y in years:
        for i in range(per_year):
            d = NYSE.on_or_before(date(y, 1, 5) + timedelta(days=int(360 * i / per_year)))
            role = rng.choice(["CEO", "director", "director", "officer"])
            e = ev(filing_date=d, first_trade_date=d, last_trade_date=d, ticker=f"T{i % 97}",
                   filer_id=i, role=role)
            r = rng.gauss(0, 0.05) + (edge if role == "CEO" else 0)
            x = res(e, r20=r)
            rs.append(x)
            ss.append(sig(role=role, drift=rng.uniform(-0.05, 0.1)))
    return rs, ss


def test_shown_applies_the_hide_rules():
    base = res(ev())
    assert ST.shown(base, sig())
    assert not ST.shown(base, sig(drift=0.2001))
    assert ST.shown(base, sig(drift=0.20))
    assert not ST.shown(res(ev(is_10b5_1=True)), sig())
    assert not ST.shown(res(ev(side="sell")), sig())
    assert not ST.shown(res(ev(), open_=1.99), sig())
    assert not ST.shown(res(ev(), dv=99_999), sig())
    assert not ST.shown(res(ev(), status="no_prices"), sig())


def test_walk_forward_finds_the_planted_edge_out_of_sample():
    rs, ss = synthetic_history()
    folds = ST.walk_forward(rs, ss, min_train_years=2)
    assert [f.year for f in folds] == [2017, 2018, 2019, 2020]
    for f in folds:
        top = [x for x in f.scored if x.above]
        assert top and all(x.signals.role == "CEO" for x in top)
        assert f.model.weights["role"] > 0


def test_walk_forward_is_purged_and_never_sees_the_test_year():
    rs, ss = synthetic_history()
    folds = ST.walk_forward(rs, ss)
    f17 = next(f for f in folds if f.year == 2017)
    # Only events whose 20-day exit was before 1 Jan 2017 trained the 2017 model.
    n = sum(1 for x in rs if x.outcome.legs[20]["exit"] < date(2017, 1, 1))
    late_2016 = sum(1 for x in rs if x.outcome.entry.year == 2016
                    and x.outcome.legs[20]["exit"] >= date(2017, 1, 1))
    assert late_2016 > 0
    assert f17.n_train == n
    # Wrecking every 2017-and-later return leaves the 2017 model (and its scores) unchanged.
    for x in rs:
        if x.outcome.entry.year >= 2017 or x.outcome.legs[20]["exit"] >= date(2017, 1, 1):
            x.outcome.legs[20]["stock"] = -9.0
    again = next(f for f in ST.walk_forward(rs, ss) if f.year == 2017)
    assert again.model.tables == f17.model.tables
    assert [x.score for x in again.scored] == [x.score for x in f17.scored]


def test_dedupe_keeps_the_best_scored_insider_per_stock_and_day():
    d = date(2024, 1, 10)
    a = ST.Scored(res(ev(filer_id=1)), sig(), 0.3, 0.3, 2024, False)
    b = ST.Scored(res(ev(filer_id=2)), sig(), 0.7, 0.7, 2024, True)
    c = ST.Scored(res(ev(filer_id=3, ticker="Y")), sig(), 0.1, 0.1, 2024, False)
    out = ST.dedupe([a, b, c])
    assert [x.result.event.filer_id for x in out] == [2, 3]
    assert out[0].result.outcome.entry == NYSE.add(d, 1)


# ---------------------------------------------------------------- portfolio

def test_portfolio_by_hand():
    start = date(2024, 1, 2)
    n = 30
    spy = bars_from(start, [100 * 1.001 ** i for i in range(n)])     # +0.1% a day
    a = bars_from(start, [50 * 1.011 ** i for i in range(n)])        # +1.1% a day
    b = bars_from(start, [20.0] * n)
    src = Source({"SPY": spy, "A": a, "B": b})
    days = spy.dates

    def cand(t, day_i, score, h=3, cost=0.002):
        e = ev(ticker=t)
        o = Outcome("ok", entry=days[day_i], entry_open=10, cost=cost)
        o.legs[h] = {"exit": days[day_i + h - 1], "stock": 0, "spy": 0, "sector": 0, "iwm": 0,
                     "flags": []}
        return ST.Scored(Result(e, o, "SPY"), sig(), score, score, 2024, True)

    # Opens equal closes here, so the entry day's open-to-close return is 0 for everyone.
    pf = ST.simulate([cand("A", 1, 0.9), cand("B", 1, 0.5), cand("A", 2, 0.99)], src, h=3,
                     slots=2)
    assert pf.days == days[1:4]              # the rejected A's later exit adds no days
    assert pf.trades == 2                       # the second A is already held
    assert pf.positions == [2, 2, 2]
    # Day 1: both enter, pay half the round trip; no moves.
    assert pf.excess[0] == pytest.approx((-0.001 - 0.001) / 2)
    # Day 2: A +1.1% vs SPY +0.1%; B 0% vs +0.1%.
    assert pf.excess[1] == pytest.approx((0.010 - 0.001) / 2)
    # Day 3: same moves, and both exit (half the round trip each).
    assert pf.excess[2] == pytest.approx((0.010 - 0.001 - 0.002) / 2)
    assert pf.spy[1] == pytest.approx(0.001)


def test_portfolio_respects_slots_and_prefers_higher_scores():
    start = date(2024, 1, 2)
    spy = bars_from(start, [100.0] * 20)
    src = Source({"SPY": spy, **{t: bars_from(start, [10.0] * 20) for t in "ABC"}})
    days = spy.dates

    def cand(t, score):
        o = Outcome("ok", entry=days[0], entry_open=10, cost=0.0)
        o.legs[20] = {"exit": days[4], "stock": 0, "spy": 0, "sector": 0, "iwm": 0, "flags": []}
        return ST.Scored(Result(ev(ticker=t), o, "SPY"), sig(), score, score, 2024, True)

    pf = ST.simulate([cand("A", 0.1), cand("B", 0.9), cand("C", 0.5)], src, slots=2)
    assert pf.trades == 2 and max(pf.positions) == 2


def test_portfolio_metrics():
    pf = ST.Portfolio([], [0.001, -0.0005] * 50, [0.0] * 100, [1] * 100, 1)
    m = sum(pf.excess) / 100
    sd = math.sqrt(sum((v - m) ** 2 for v in pf.excess) / 99)
    assert pf.ir() == pytest.approx(m / sd * math.sqrt(252))
    assert ST.Portfolio.max_drawdown([0.1, -0.5, 0.2]) == pytest.approx(-0.5)
    assert ST.Portfolio.max_drawdown([0.1, 0.1]) == 0.0


def test_delayed_excess():
    start = date(2024, 1, 2)
    spy = bars_from(start, [100.0] * 40)
    x_closes = [10 + i for i in range(40)]
    x = bars_from(start, x_closes)
    src = Source({"SPY": spy, "X": x})
    o = Outcome("ok", entry=x.dates[0], entry_open=10, cost=0.01)
    s = ST.Scored(Result(ev(), o, "SPY"), sig(), 1, 1, 2024, True)
    # Enter 2 sessions late at the open of day 2 (12), exit at day 21's close (31).
    assert ST.delayed_excess(s, src, 2, 20) == pytest.approx(31 / 12 - 1 - 0.01)
    assert ST.delayed_excess(s, src, 25, 20) is None


def test_criteria_pass_and_fail():
    start = date(2024, 1, 2)
    good = [ST.Scored(res(ev(), r20=0.03, cost=0.004), sig(), 1, 1, y, True)
            for y in (2020, 2021, 2022) for _ in range(30)]
    pooled = {"top": good, "yearly": [0.026, 0.026, 0.026]}
    pf = ST.Portfolio([start] * 100, [0.001, 0.0008] * 50, [0.001, -0.001] * 50, [5] * 100, 10)
    _, ok = ST.criteria(pooled, pf)
    assert ok == [True, True, True, True]
    bad = [ST.Scored(res(ev(), r20=-0.01), sig(), 1, 1, 2020, True)]
    _, ok = ST.criteria({"top": bad, "yearly": [-0.014]}, pf)
    assert ok[0] is False and ok[2] is False


# ---------------------------------------------------------------- ranking

def test_rank_hides_run_ups_stale_and_illiquid_filings_and_applies_decay():
    as_of = date(2024, 3, 1)
    start = date(2023, 1, 3)
    n = NYSE.business_days_between(start, as_of) + 1
    flat = bars_from(start, [10.0] * n)
    runup = bars_from(start, [10.0] * (n - 5) + [13.0] * 5)
    penny = bars_from(start, [1.5] * n)
    thin = bars_from(start, [10.0] * n, volume=1000)
    src = Source({"A": flat, "B": runup, "C": penny, "D": thin, "E": flat, "F": flat})
    trade = date(2024, 2, 20)
    fresh = NYSE.add(trade, 2)
    rows = [
        res(ev(ticker="A", first_trade_date=trade, filing_date=fresh), r20=None),
        res(ev(ticker="B", first_trade_date=trade, filing_date=fresh), r20=None),
        res(ev(ticker="C", first_trade_date=trade, filing_date=fresh, value=1500.0), r20=None),
        res(ev(ticker="D", first_trade_date=trade, filing_date=fresh), r20=None),
        res(ev(ticker="E", first_trade_date=date(2023, 12, 1), filing_date=date(2023, 12, 5)),
            r20=None),                                                  # > 30 sessions ago
        res(ev(ticker="F", first_trade_date=trade, filing_date=fresh, is_10b5_1=True),
            r20=None),
        res(ev(ticker="A", filer_id=2, role="director", first_trade_date=trade,
               filing_date=fresh), r20=None),                          # same stock
        res(ev(ticker="A", first_trade_date=trade, filing_date=date(2024, 3, 4)), r20=None),
    ]
    sigs = [sig() for _ in rows]
    sigs[6] = sig(role="director")
    m = M.Model(tables={"role": {"CEO": 0.02, "director": -0.01}}, counts={},
                weights={"role": 1.0}, beta=2.0, half_life=10)
    out, hidden = R.rank(rows, sigs, src, m, as_of)
    assert [r.result.event.ticker for r in out] == ["A"]
    assert out[0].result.event.role == "CEO"
    assert hidden == {"drift": 1, "illiquid": 2, "no_prices": 0, "ticker_mismatch": 0,
                      "10b5-1": 1}
    d = NYSE.business_days_between(fresh, as_of)
    assert out[0].days_since_filing == d
    assert out[0].score == pytest.approx(1.0 * M.Model.decay(m, d))
    assert out[0].signals.drift == 0.0 and out[0].signals.drawdown == 0.0
    assert "filed" in out[0].reason


def test_rank_drops_a_symbol_that_now_belongs_to_another_company():
    as_of = date(2024, 3, 1)
    start = date(2023, 1, 3)
    n = NYSE.business_days_between(start, as_of) + 1
    src = Source({"A": bars_from(start, [10.0] * n), "B": bars_from(start, [10.0] * n)})
    trade = date(2024, 2, 20)
    fresh = NYSE.add(trade, 2)
    # Form 4 prices of $4.99 and $20.01 against a $10 close: outside 0.5x to 2x.
    low = ev(ticker="A", first_trade_date=trade, last_trade_date=trade, filing_date=fresh,
             value=4990.0, shares=1000.0)
    high = ev(ticker="B", first_trade_date=trade, last_trade_date=trade, filing_date=fresh,
              value=20010.0, shares=1000.0)
    ok = ev(ticker="B", filer_id=2, first_trade_date=trade, last_trade_date=trade,
            filing_date=fresh, value=19990.0, shares=1000.0)
    m = M.Model(tables={"role": {"CEO": 0.0}}, counts={}, weights={"role": 1.0}, beta=0.0)
    out, hidden = R.rank([res(low, r20=None), res(high, r20=None), res(ok, r20=None)],
                         [sig(), sig(), sig()], src, m, as_of)
    assert hidden["ticker_mismatch"] == 2
    assert [(r.result.event.ticker, r.result.event.filer_id) for r in out] == [("B", 2)]
