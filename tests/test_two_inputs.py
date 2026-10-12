"""The stripped-down score: the 52-week drawdown and the run-up penalty only."""

import random
from datetime import date, timedelta

import pytest
from test_scoring import ev, res, sig

from tradetracker.features.calendar import NYSE
from tradetracker.scoring import study as ST


def test_two_input_setups():
    names = [c.name for c in ST.TWO_INPUT_VARIANTS]
    assert names[0] == "phase 4" and len(set(names)) == len(names) == 9
    two = ST.TWO_INPUT_VARIANTS[1:]
    assert all(c.comps() == ["drawdown"] for c in two)
    # Every combination of hold, cap floor and hedge, once.
    assert {(c.h, c.min_cap, c.bench) for c in two} == {
        (h, cap, b) for h in (20, 60) for cap in (None, 50e6) for b in ("spy", "beta")}
    assert two[-1].label() == "60 days, beta-hedged, cap >= $50M, drawdown and run-up only"


def deep_history(seed=3, per_year=600, years=range(2015, 2021), edge=0.03, runup_cost=0.3):
    """Buys 25%+ below the 52-week high earn `edge`; a run-up since the trade costs
    `runup_cost` x the run-up. Role and the rest are noise."""
    rng = random.Random(seed)
    rs, ss = [], []
    for y in years:
        for i in range(per_year):
            d = NYSE.on_or_before(date(y, 1, 5) + timedelta(days=int(350 * i / per_year)))
            dd = rng.uniform(-0.7, 0.0)
            drift = rng.uniform(-0.1, 0.15)
            r = rng.gauss(0, 0.04) + (edge if dd <= -0.25 else 0) - runup_cost * max(drift, 0)
            e = ev(filing_date=d, first_trade_date=d, last_trade_date=d, ticker=f"T{i % 97}",
                   filer_id=i, role=rng.choice(["CEO", "director"]))
            rs.append(res(e, r20=r))
            ss.append(sig(role=e.role, drawdown=dd, drift=drift))
    return rs, ss


def test_two_input_fit_uses_only_the_drawdown_and_the_penalty():
    rs, ss = deep_history()
    cfg = ST.Config("t", components=ST.TWO_INPUTS)
    folds = ST.walk_forward(rs, ss, cfg=cfg)
    assert folds
    for f in folds:
        assert set(f.model.weights) == {"drawdown"} and f.model.weights["drawdown"] > 0
        t = f.model.tables["drawdown"]
        assert t["<=-45%"] > t["within 10%"] and t["-45 to -25%"] > t["-25 to -10%"]
        assert f.model.beta > 0                       # the planted run-up cost
        top = [x for x in f.scored if x.above]
        assert top and all(x.signals.drawdown <= -0.25 for x in top)
    final = ST.fit_final(rs, ss, cfg=cfg)
    assert set(final.weights) == {"drawdown"}


def test_ablation_of_a_one_component_model_doesnt_fall_back_to_phase_4():
    rs, ss = deep_history(per_year=300)
    cfg = ST.Config("t", components=ST.TWO_INPUTS)
    folds = ST.walk_forward(rs, ss, cfg=cfg)
    text = ST.ablation(rs, ss, [x for f in folds for x in f.scored if x.above], cfg)
    labels = [ln.split("|")[1].strip() for ln in text.splitlines()[2:]]
    assert labels == ["all components", "no drift penalty (beta 0)"]


def test_rule_table_by_hand():
    def one(y, dd, drift, r):
        d = date(y, 3, 2)
        e = ev(filing_date=d, first_trade_date=d, last_trade_date=d)
        return res(e, r20=r, cost=0.0), sig(drawdown=dd, drift=drift)

    pairs = [one(2017, -0.30, -0.01, 0.05),     # in the rule
             one(2017, -0.25, 0.0, 0.03),       # edges count (<= -25%, <= 0)
             one(2017, -0.30, 0.01, -0.02),     # rose since the trade: out
             one(2017, -0.10, -0.05, -0.04),    # not deep enough: out
             one(2017, None, -0.05, 0.10),      # no drawdown: out
             one(2023, -0.50, -0.02, -0.01)]
    text = ST.rule_table([x for x, _ in pairs], [s for _, s in pairs])
    rows = {ln.split("|")[1].strip(): [c.strip() for c in ln.split("|")[2:-1]]
            for ln in text.splitlines()[2:]}
    # 2017: all five average 0.024; the rule's two average 0.04.
    assert rows["2017"][:5] == ["5", "2.40%", "2", "4.00%", "1.60%"]
    assert rows["2017"][5] == "100.0%"
    assert rows["2023"][:4] == ["1", "-1.00%", "1", "-1.00%"]
    assert rows["**all years**"][:4] == ["6", "1.83%", "3", "2.33%"]
    assert rows["**2015-2016 (before any test year)**"][0] == "0"


@pytest.mark.parametrize("h", [20, 60])
def test_two_input_variants_table_runs(h):
    from test_scoring import Source, bars_from

    rs, ss = deep_history(per_year=400)
    for x in rs:
        x.outcome.legs[60] = dict(x.outcome.legs[20], exit=NYSE.add(x.outcome.entry, 59))
    for s in ss:
        s.market_cap, s.beta = 1e9, 1.0
    ST.add_beta_legs(rs, ss)
    start = date(2014, 6, 2)
    n = NYSE.business_days_between(start, date(2021, 6, 1))
    src = Source({"SPY": bars_from(start, [100.0] * n),
                  **{f"T{i}": bars_from(start, [10.0] * n) for i in range(97)}})
    configs = [c for c in ST.TWO_INPUT_VARIANTS if c.h == h]
    text, sums = ST.variants_table(rs, ss, src, configs)
    assert list(sums) == [c.name for c in configs]
    for c in configs[1 if h == 20 else 0:]:
        assert list(sums[c.name]["config"]["components"]) == ["drawdown"]
        assert sums[c.name]["pooled"]["net_mean"] > 0.02
