"""The new approach after phase 4: market-cap floor, longer holds, beta hedge and
volatility-scaled positions. Hand-computed cases, plus real SEC share counts and bars."""

import csv
import math
import random
from datetime import date, timedelta
from pathlib import Path

import pytest

from tradetracker import fundamentals as F
from tradetracker.backtest import prices as bp
from tradetracker.backtest.returns import Outcome
from tradetracker.backtest.study import Result
from tradetracker.features.calendar import NYSE
from tradetracker.scoring import model as M
from tradetracker.scoring import rank as R
from tradetracker.scoring import signals as S
from tradetracker.scoring import study as ST

from test_scoring import Source, bars_from, ev, res, sig

REAL = Path(__file__).parent / "fixtures" / "real"


# ---------------------------------------------------------------- share counts

def test_frame_names_cover_every_quarter_between_the_dates():
    assert F.frame_names(date(2023, 11, 5), date(2024, 7, 1)) == [
        (2023, 4), (2024, 1), (2024, 2), (2024, 3)]
    assert F.frame_names(date(2024, 3, 31), date(2024, 3, 31)) == [(2024, 1)]


def test_parse_frame_strips_zeros_and_drops_bad_values():
    payload = {"data": [
        {"cik": 1041859, "end": "2023-08-24", "val": 12478857, "accn": "a"},
        {"cik": "0000350852", "end": "2023-07-31", "val": "17991419"},
        {"cik": 1, "end": "2023-07-31", "val": 0},
        {"cik": 2, "end": "2023-07-31", "val": "n/a"},
        {"cik": 3, "end": "2023-07-31"},
    ]}
    rows = F.parse_frame(payload, "CY2023Q3I")
    assert rows == [
        {"cik": "1041859", "end": "2023-08-24", "shares": 12478857.0, "accn": "a",
         "frame": "CY2023Q3I"},
        {"cik": "350852", "end": "2023-07-31", "shares": 17991419.0, "accn": "",
         "frame": "CY2023Q3I"},
    ]


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload


class FakeClient:
    def __init__(self, missing=()):
        self.urls, self.missing = [], set(missing)

    def get(self, url):
        self.urls.append(url)
        frame = url.rsplit("/", 1)[1].removesuffix(".json")
        if frame in self.missing:
            raise RuntimeError("404")
        return FakeResponse({"data": [{"cik": 7, "end": "2020-01-01", "val": 100}]})


def test_fetch_appends_only_frames_not_cached_and_skips_unpublished_ones(tmp_path):
    out = tmp_path / "shares.csv"
    c = FakeClient(missing={"CY2020Q2I"})
    got = F.fetch(c, out, date(2020, 1, 1), date(2020, 6, 30))
    assert got == {"frames": 1, "facts": 1, "skipped": 0}
    assert len(c.urls) == 2 and c.urls[0] == F.FRAME_URL.format(year=2020, q=1)
    c2 = FakeClient()
    got = F.fetch(c2, out, date(2020, 1, 1), date(2020, 6, 30))
    assert got == {"frames": 1, "facts": 1, "skipped": 1}
    assert [u.rsplit("/", 1)[1] for u in c2.urls] == ["CY2020Q2I.json"]
    rows = list(csv.DictReader(out.open()))
    assert [r["frame"] for r in rows] == ["CY2020Q1I", "CY2020Q2I"]   # one header


def test_share_count_is_used_only_100_days_after_its_date():
    t = F.SharesTable([("0001", date(2023, 3, 31), 1e6), ("1", date(2023, 6, 30), 2e6),
                       ("1", date(2023, 6, 30), 2.5e6)])      # a repeat: the last one wins
    assert len(t) == 1
    assert t.as_of("1", date(2023, 7, 8)) is None               # 3/31 + 100 days = 7/9
    assert t.as_of("1", date(2023, 7, 9)) == (date(2023, 3, 31), 1e6)
    assert t.as_of("0001", date(2023, 10, 7)) == (date(2023, 3, 31), 1e6)
    assert t.as_of("1", date(2023, 10, 8)) == (date(2023, 6, 30), 2.5e6)
    assert t.as_of("2", date(2024, 1, 1)) is None and t.as_of(None, date(2024, 1, 1)) is None


def split_bars(start, raw, factor_from: int, ratio: float):
    """Raw closes with a split of `ratio` new shares per old from bar `factor_from` on.
    Adjusted closes divide the earlier bars by the ratio."""
    ds = [start + timedelta(days=i) for i in range(len(raw))]
    return bp.Bars([bp.Bar(d, c, c, c / ratio if i < factor_from else c,
                           c / ratio if i < factor_from else c, 1e6)
                    for i, (d, c) in enumerate(zip(ds, raw))])


def test_market_cap_scales_the_count_across_splits():
    start = date(2023, 1, 2)
    t = F.SharesTable([("9", start, 1e6)], lag_days=0)
    # 2-for-1 on day 3: a $100 stock trades at $50, and 1M shares became 2M.
    fwd = split_bars(start, [100, 100, 100, 50, 50, 50], 3, 2.0)
    assert F.market_cap(t, "9", fwd, start + timedelta(days=5)) == pytest.approx(2e6 * 50)
    # 1-for-10 on day 3: a $1 stock trades at $10, and 1M shares became 100k.
    rev = split_bars(start, [1, 1, 1, 10, 10, 10], 3, 0.1)
    assert F.market_cap(t, "9", rev, start + timedelta(days=5)) == pytest.approx(1e5 * 10)
    # No split: count x the last close before entry (not the entry day's).
    flat = split_bars(start, [5, 6, 7, 8], 0, 1.0)
    assert F.market_cap(t, "9", flat, start + timedelta(days=3)) == pytest.approx(1e6 * 7)
    assert F.market_cap(t, "9", flat, start) is None           # no bar before entry
    assert F.market_cap(None, "9", flat, start + timedelta(days=3)) is None
    assert F.market_cap(t, "8", flat, start + timedelta(days=3)) is None


def test_market_cap_with_a_count_dated_before_the_first_bar_uses_the_first_bar():
    t = F.SharesTable([("9", date(2022, 12, 1), 1e6)], lag_days=0)
    rev = split_bars(date(2023, 1, 2), [1, 1, 10, 10], 2, 0.1)
    assert F.market_cap(t, "9", rev, date(2023, 1, 6)) == pytest.approx(1e5 * 10)


def real_bars(ticker):
    return bp.open_source(str(REAL / "backtest_bars"), None).get(ticker)


def test_real_market_caps_by_hand():
    t = F.SharesTable.load(REAL / "shares_outstanding.csv")
    # PLCE, entering 15 Feb 2024: the latest count dated by 7 Nov 2023 is 12,478,857
    # (24 Aug 2023, from the 10-Q). Last close before entry: 14.515 on 14 Feb. The
    # fixture bars have no split or dividend (close = adjusted close).
    plce = real_bars("PLCE")
    assert F.market_cap(t, "0001041859", plce, date(2024, 2, 15)) == pytest.approx(
        12_478_857 * 14.515)
    # From 6 Mar 2024 (27 Nov + 100 days) the 27 Nov 2023 count (12,477,325) is used.
    assert t.as_of("1041859", date(2024, 3, 5)) == (date(2023, 8, 24), 12_478_857)
    assert t.as_of("1041859", date(2024, 3, 6)) == (date(2023, 11, 27), 12_477_325)
    # RMCF, entering 15 Feb 2024: 6,302,185 (10 Oct 2023) x 4.20 = $26.5M, under the floor.
    rmcf = F.market_cap(t, "1616262", real_bars("RMCF"), date(2024, 2, 15))
    assert rmcf == pytest.approx(6_302_185 * 4.20) and rmcf < 50e6
    # CTBI, entering 26 Jan 2024: 17,991,419 (31 Jul 2023) x 42.07. CTBI paid a dividend
    # in December, which moves the raw/adjusted ratio by about 1%: the factor is
    # (37.22 / 33.2229) / (42.07 / 37.9509) from the 1 Nov 2023 and 25 Jan 2024 bars.
    factor = (37.22 / 33.222893791633716) / (42.07 / 37.950889893797715)
    assert factor == pytest.approx(1.0106, abs=1e-4)
    assert F.market_cap(t, "350852", real_bars("CTBI"), date(2024, 1, 26)) == pytest.approx(
        17_991_419 * factor * 42.07)


# ---------------------------------------------------------------- beta and volatility

def test_risk_recovers_a_planted_beta_from_closes_before_entry():
    rng = random.Random(1)
    start = date(2022, 1, 3)
    spy_r = [rng.gauss(0, 0.01) for _ in range(300)]
    spy, stock = [100.0], [50.0]
    for r in spy_r:
        spy.append(spy[-1] * (1 + r))
        stock.append(stock[-1] * (1 + 2 * r))
    # The entry day's bar moves wildly; it must not count.
    stock[-1] = stock[-2] * 3
    src = Source({"SPY": bars_from(start, spy), "X": bars_from(start, stock)})
    entry = src.get("X").dates[-1]
    beta, vol = S.risk(src, "X", entry)
    assert beta == pytest.approx(2.0)
    rs = spy_r[-251:-1]                        # the 250 returns ending the day before entry
    m = sum(rs) / len(rs)
    sd = math.sqrt(sum((r - m) ** 2 for r in rs) / (len(rs) - 1))
    assert vol == pytest.approx(2 * sd * math.sqrt(252))


def test_risk_needs_60_sessions():
    start = date(2022, 1, 3)
    src = Source({"SPY": bars_from(start, [100 + i % 3 for i in range(80)]),
                  "X": bars_from(start, [10 + i % 5 for i in range(80)])})
    d = src.get("X").dates
    assert S.risk(src, "X", d[59]) == (None, None)
    b, v = S.risk(src, "X", d[61])
    assert b is not None and v > 0
    assert S.risk(src, "Y", d[70]) == (None, None)


# ---------------------------------------------------------------- setups

def test_config_labels():
    assert ST.PHASE4.label() == "20 days, vs SPY"
    c = ST.Config("x", h=60, bench="beta", min_cap=50e6, vol_target=0.4)
    assert c.label() == "60 days, beta-hedged, cap >= $50M, vol-sized to 40%"
    assert [v.name for v in ST.VARIANTS][0] == "phase 4"
    assert len({v.name for v in ST.VARIANTS}) == len(ST.VARIANTS) == 5


def test_cap_floor_hides_small_and_unknown_caps():
    x = res(ev())
    cfg = ST.Config("floor", min_cap=50e6)
    assert ST.shown(x, sig(market_cap=None))                  # phase 4 ignores cap
    assert not ST.shown(x, sig(market_cap=None), cfg)
    assert not ST.shown(x, sig(market_cap=49.9e6), cfg)
    assert ST.shown(x, sig(market_cap=50e6), cfg)


def test_hedge_beta_is_clipped_and_defaults_to_1():
    assert ST.hedge_beta(sig()) == 1.0
    assert ST.hedge_beta(sig(beta=-0.4)) == 0.0
    assert ST.hedge_beta(sig(beta=5.0)) == 3.0
    assert ST.hedge_beta(sig(beta=1.7)) == 1.7


def test_beta_legs_hedge_each_horizon():
    x = res(ev(), r20=0.05)
    x.outcome.legs[20]["spy"] = 0.02
    x.outcome.legs[60] = {"stock": 0.1, "spy": None, "exit": date(2024, 5, 1)}
    ST.add_beta_legs([x], [sig(beta=1.5)])
    assert x.outcome.legs[20]["beta"] == pytest.approx(0.03)
    assert x.r(20, "beta") == pytest.approx(0.02)
    assert x.r(20, "beta", net=True) == pytest.approx(0.02 - 0.004)
    assert x.outcome.legs[60]["beta"] is None


def test_position_weight():
    assert ST.position_weight(sig(vol=0.8), 20, None) == 1 / 20
    assert ST.position_weight(sig(vol=0.8), 20, 0.4) == pytest.approx(0.5 / 20)
    assert ST.position_weight(sig(vol=0.1), 20, 0.4) == pytest.approx(2 / 20)   # capped
    assert ST.position_weight(sig(vol=None), 20, 0.4) == 1 / 20


def test_hedged_vol_sized_portfolio_by_hand():
    start = date(2024, 1, 2)
    n = 10
    spy = bars_from(start, [100 * 1.01 ** i for i in range(n)])      # +1% a day
    a = bars_from(start, [50 * 1.02 ** i for i in range(n)])         # +2% a day
    b = bars_from(start, [20.0] * n)                                  # flat
    src = Source({"SPY": spy, "A": a, "B": b})
    days = spy.dates

    def cand(t, score, beta, vol):
        o = Outcome("ok", entry=days[1], entry_open=10, cost=0.0)
        o.legs[3] = {"exit": days[3], "stock": 0, "spy": 0, "flags": []}
        return ST.Scored(Result(ev(ticker=t), o, "SPY"), sig(beta=beta, vol=vol),
                         score, score, 2024, True)

    cands = [cand("A", 0.9, 2.0, 0.2), cand("B", 0.5, 0.5, 0.8)]
    plain = ST.simulate(cands, src, h=3, slots=2)
    pf = ST.simulate(cands, src, h=3, slots=2, hedged=True, vol_target=0.4)
    assert pf.days == plain.days == days[1:4]
    # Day 1 opens equal closes: nothing moves. Day 2: A +2%, B 0%, SPY +1%.
    # Unhedged, equal slots: ((0.02 - 0.01) + (0 - 0.01)) / 2 = 0.
    assert plain.excess[1] == pytest.approx(0.0)
    # Hedged, vol-sized: A shorts 2 x SPY and gets min(2, 0.4/0.2) = 2 slots; B shorts
    # 0.5 x SPY at 0.4/0.8 = 0.5 slots. (0.02 - 0.02) x 2/2 + (0 - 0.005) x 0.5/2.
    assert pf.excess[1] == pytest.approx(-0.005 * 0.25)


def test_criteria_scale_a_longer_hold_to_20_days():
    start = date(2024, 1, 2)
    pf = ST.Portfolio([start] * 100, [0.001, 0.0008] * 50, [0.001, -0.001] * 50, [5] * 100, 10)
    cfg = ST.Config("long", h=60)

    def pooled(r):
        top = []
        for y in (2020, 2021, 2022):
            for _ in range(30):
                x = res(ev(), r20=None, cost=0.0)
                x.outcome.legs[60] = {"stock": r, "spy": 0.0, "exit": date(y, 12, 1)}
                top.append(ST.Scored(x, sig(), 1, 1, y, True))
        return {"top": top, "yearly": [r] * 3}

    _, ok = ST.criteria(pooled(0.029), pf, cfg)                   # 0.97% per 20 days
    assert ok[0] is False
    text, ok = ST.criteria(pooled(0.031), pf, cfg)                # 1.03% per 20 days
    assert ok[0] is True and "over 60 days = 1.03% per 20" in text


def history_60(seed=7, per_year=300, years=range(2015, 2021)):
    """CEO buys earn 6% over 60 days, and only buys with a market cap of $50M+ count."""
    rng = random.Random(seed)
    rs, ss = [], []
    for y in years:
        for i in range(per_year):
            d = NYSE.on_or_before(date(y, 1, 5) + timedelta(days=int(300 * i / per_year)))
            role = rng.choice(["CEO", "director", "officer"])
            e = ev(filing_date=d, first_trade_date=d, last_trade_date=d, ticker=f"T{i % 89}",
                   filer_id=i, role=role)
            x = res(e, r20=rng.gauss(0, 0.05))
            entry = x.outcome.entry
            x.outcome.legs[60] = {"stock": rng.gauss(0, 0.05) + (0.06 if role == "CEO" else 0),
                                  "spy": 0.01, "sector": 0.0, "iwm": 0.0,
                                  "exit": NYSE.add(entry, 59), "flags": []}
            rs.append(x)
            ss.append(sig(role=role, beta=rng.uniform(0.5, 1.5), vol=0.4,
                          market_cap=rng.choice([None, 20e6, 80e6, 5e9])))
    ST.add_beta_legs(rs, ss)
    return rs, ss


def test_walk_forward_on_60_day_beta_hedged_returns_is_purged_by_the_60_day_exit():
    rs, ss = history_60(per_year=700)
    cfg = ST.Config("t", h=60, bench="beta", min_cap=50e6)
    folds = ST.walk_forward(rs, ss, cfg=cfg)
    f = next(f for f in folds if f.year == 2018)
    pool = [(x, s) for x, s in zip(rs, ss) if ST.shown(x, s, cfg)]
    assert all(s.market_cap >= 50e6 for _, s in pool)
    assert f.n_train == sum(1 for x, _ in pool if x.outcome.legs[60]["exit"] < date(2018, 1, 1))
    assert all(x.signals.market_cap >= 50e6 for x in f.scored)
    top = [x for f in folds for x in f.scored if x.above]
    assert top and all(x.signals.role == "CEO" for x in top)
    # Wrecking 60-day returns that exit in or after 2018 leaves the 2018 model unchanged.
    for x in rs:
        if x.outcome.legs[60]["exit"] >= date(2018, 1, 1):
            x.outcome.legs[60]["stock"] = -9.0
    ST.add_beta_legs(rs, ss)
    again = next(g for g in ST.walk_forward(rs, ss, cfg=cfg) if g.year == 2018)
    assert again.model.tables == f.model.tables


def test_variants_table_runs_every_setup():
    rs, ss = history_60(per_year=120)
    start = date(2014, 6, 2)
    n = NYSE.business_days_between(start, date(2021, 6, 1))
    src = Source({"SPY": bars_from(start, [100.0] * n),
                  **{f"T{i}": bars_from(start, [10.0] * n) for i in range(89)}})
    text, sums = ST.variants_table(rs, ss, src)
    assert list(sums) == [v.name for v in ST.VARIANTS]
    assert text.count("\n") == len(ST.VARIANTS) + 1
    assert sums["cap floor, 60 days, vol control"]["config"]["bench"] == "beta"
    # The cap floor keeps only half the events (80M and 5B of four cap choices).
    assert sums["cap floor"]["pooled"]["above"] < sums["phase 4"]["pooled"]["above"]


# ---------------------------------------------------------------- ranking

def test_rank_applies_the_cap_floor_as_of_today():
    as_of = date(2024, 3, 1)
    start = date(2023, 1, 3)
    n = NYSE.business_days_between(start, as_of) + 1
    src = Source({"SPY": bars_from(start, [100.0] * n),
                  "BIG": bars_from(start, [10.0] * n, volume=1e6),
                  "SMALL": bars_from(start, [10.0] * n, volume=1e6)})
    filed = NYSE.add(as_of, -3)
    xs = [res(ev(ticker="BIG", issuer_cik="1", filing_date=filed, first_trade_date=filed,
                 last_trade_date=filed), r20=None),
          res(ev(ticker="SMALL", issuer_cik="2", filing_date=filed, first_trade_date=filed,
                 last_trade_date=filed, filer_id=2), r20=None)]
    shares = F.SharesTable([("1", date(2023, 6, 30), 10e6), ("2", date(2023, 6, 30), 4e6)])
    m = M.Model(tables={"role": {"CEO": 0.01, "none": 0.0}}, counts={}, weights={"role": 1.0},
                beta=0.0, half_life=10)
    out, hidden = R.rank(xs, [sig(), sig()], src, m, as_of)
    assert {r.result.event.ticker for r in out} == {"BIG", "SMALL"}
    out, hidden = R.rank(xs, [sig(), sig()], src, m, as_of, shares=shares, min_cap=50e6)
    assert [r.result.event.ticker for r in out] == ["BIG"]
    assert out[0].signals.market_cap == pytest.approx(100e6)
    assert hidden["small_cap"] == 1
