"""Routine vs opportunistic insiders (Cohen, Malloy and Pomorski 2012): the classifier,
the trade history, and their use in the score backtest."""

import random
from datetime import date, timedelta
from pathlib import Path

import pytest
from test_scoring import ev, res, sig

from tradetracker import insiders as I
from tradetracker.edgar.bulk import parse_quarter
from tradetracker.features.calendar import NYSE
from tradetracker.scoring import model as M
from tradetracker.scoring import study as ST

REAL = Path(__file__).parent / "fixtures" / "real"


def rows(owner, *dates, filed_lag=2):
    """(owner, trade date, filed date) rows, each filed `filed_lag` days after the trade."""
    return [(owner, d, d + timedelta(days=filed_lag)) for d in dates]


def test_same_month_three_years_running_is_routine():
    c = I.Classifier(rows("1", date(2020, 3, 5), date(2021, 3, 20), date(2022, 3, 1),
                          date(2022, 8, 1)))
    assert c.kind_in_year("1", 2023) == I.ROUTINE
    # Its 2023 buys and sells keep that type all year, whatever they are.
    assert c.kind("1", date(2023, 12, 31)) == I.ROUTINE


def test_trading_every_year_in_different_months_is_opportunistic():
    c = I.Classifier(rows("1", date(2020, 3, 5), date(2021, 4, 20), date(2022, 3, 1)))
    assert c.kind_in_year("1", 2023) == I.OPPORTUNISTIC


def test_a_missing_year_leaves_the_insider_unclassified():
    c = I.Classifier(rows("1", date(2019, 3, 5), date(2021, 3, 20), date(2022, 3, 1)))
    assert c.kind_in_year("1", 2023) is None
    assert c.kind_in_year("2", 2023) is None and c.kind_in_year(None, 2023) is None
    # Two of three matching months aren't enough either.
    c = I.Classifier(rows("1", date(2020, 3, 5), date(2021, 3, 20), date(2022, 5, 1),
                          date(2022, 6, 1)))
    assert c.kind_in_year("1", 2023) == I.OPPORTUNISTIC


def test_only_trades_filed_before_1_january_count():
    # The December 2022 trade is filed on 3 Jan 2023: the 2023 type can't use it.
    late = [("1", date(2022, 12, 29), date(2023, 1, 3))]
    c = I.Classifier(rows("1", date(2020, 12, 5), date(2021, 12, 20)) + late)
    assert c.kind_in_year("1", 2023) is None
    # In 2024 it is public, but 2021-2023 are the years that count: no 2023 trade.
    assert c.kind_in_year("1", 2024) is None
    c = I.Classifier(rows("1", date(2020, 12, 5), date(2021, 12, 20)) + late
                     + rows("1", date(2023, 12, 1)))
    assert c.kind_in_year("1", 2024) == I.ROUTINE        # December in 2021, 2022, 2023


def test_trades_in_the_year_itself_and_earlier_years_dont_count():
    c = I.Classifier(rows("1", date(2019, 3, 5), date(2020, 4, 5), date(2021, 5, 5),
                          date(2022, 6, 5), date(2023, 3, 5)))
    assert c.kind_in_year("1", 2023) == I.OPPORTUNISTIC   # 2020, 2021, 2022 only


def test_cik_leading_zeros_and_repeats_dont_matter():
    c = I.Classifier(rows("0001", date(2020, 3, 5), date(2020, 3, 5), date(2021, 3, 9),
                          date(2022, 3, 9)))
    assert len(c) == 1
    assert c.kind_in_year("1", 2023) == c.kind_in_year("00001", 2023) == I.ROUTINE


def test_covers_needs_three_years_of_history():
    c = I.Classifier(rows("1", date(2012, 1, 3)))
    assert c.covers(2015) and not c.covers(2014)
    assert not I.Classifier([]).covers(2020)


def test_history_from_real_filings_round_trips(tmp_path):
    filings = list(parse_quarter(REAL / "2024q1_features_form345.zip"))
    rows_ = I.history_rows(filings)
    assert len(rows_) == sum(len(f.trades) for f in filings) > 0
    assert {r["side"] for r in rows_} <= {"buy", "sell"}
    path = tmp_path / "h.csv.gz"
    I.write_history(rows_, path)
    c = I.Classifier.load(path)
    owners = {f.filer_key for f in filings if f.trades}
    assert set(c.months) == owners
    # One quarter of history can't classify anyone.
    assert all(c.kind_in_year(o, 2025) is None for o in owners)


# ---------------------------------------------------------------- in the score

def test_insider_type_bucket_and_reason():
    assert M.bucket("insider type", sig()) == "unclassified"
    assert M.bucket("insider type", sig(insider_type="routine")) == "routine"
    assert "insider type" not in M.BASE_COMPONENTS and "insider type" in M.COMPONENTS
    assert M.reason(sig(insider_type="opportunistic"), "X").startswith(
        "CEO (opportunistic trader) bought $100k of X")


def test_config_type_filter_and_components():
    x = res(ev())
    cfg = ST.Config("o", only_type="opportunistic")
    assert ST.shown(x, sig(insider_type="opportunistic"), cfg)
    assert not ST.shown(x, sig(insider_type="routine"), cfg)
    assert not ST.shown(x, sig(), cfg)
    assert ST.shown(x, sig(), ST.PHASE4)
    assert ST.PHASE4.comps() == M.BASE_COMPONENTS
    assert ST.Config("t", components=ST.WITH_TYPE).comps()[-1] == "insider type"
    assert ST.Config("t", components=ST.WITH_TYPE).label() == "20 days, vs SPY, insider type in Q"
    assert cfg.label() == "20 days, vs SPY, opportunistic insiders only"
    assert [c.name for c in ST.CMP_VARIANTS][0] == "phase 4"


def typed_history(seed=11, per_year=500, years=range(2015, 2022), edge=0.03):
    """Opportunistic buys earn `edge`, routine and unclassified ones nothing."""
    rng = random.Random(seed)
    rs, ss = [], []
    for y in years:
        for i in range(per_year):
            d = NYSE.on_or_before(date(y, 1, 5) + timedelta(days=int(350 * i / per_year)))
            kind = rng.choice(["opportunistic", "routine", None])
            e = ev(filing_date=d, first_trade_date=d, last_trade_date=d,
                   ticker=f"T{i % 97}", filer_id=i)
            rs.append(res(e, r20=rng.gauss(0, 0.05) + (edge if kind == "opportunistic" else 0)))
            ss.append(sig(insider_type=kind, drift=rng.uniform(-0.05, 0.1)))
    return rs, ss


def test_walk_forward_with_insider_type_in_q_finds_the_opportunistic_edge():
    rs, ss = typed_history()
    folds = ST.walk_forward(rs, ss, cfg=ST.Config("t", components=ST.WITH_TYPE))
    assert folds
    for f in folds:
        assert f.model.weights["insider type"] > 0
        t = f.model.tables["insider type"]
        assert t["opportunistic"] > t["routine"] and t["opportunistic"] > t["unclassified"]
        top = [x for x in f.scored if x.above]
        assert top and all(x.signals.insider_type == "opportunistic" for x in top)
    # The phase 4 model never sees the type.
    assert all("insider type" not in f.model.weights for f in ST.walk_forward(rs, ss))


def test_opportunistic_only_scores_only_opportunistic_buys():
    rs, ss = typed_history()
    cfg = ST.Config("o", only_type="opportunistic")
    folds = ST.walk_forward(rs, ss, cfg=cfg)
    assert folds and all(x.signals.insider_type == "opportunistic"
                         for f in folds for x in f.scored)
    pool = [x for x, s in zip(rs, ss) if ST.shown(x, s, cfg)]
    f = folds[0]
    assert f.n_train == sum(1 for x in pool if x.outcome.legs[20]["exit"] < date(f.year, 1, 1))


def test_type_table_splits_every_shown_buy():
    rs, ss = typed_history(per_year=200, years=range(2019, 2021))
    text = ST.type_table(rs, ss, ST.Config("o", only_type="routine"))   # filter ignored
    lines = text.splitlines()
    assert lines[0].startswith("| insider type | n |") and "| 2019 | 2020 |" in lines[0]
    n = {ln.split("|")[1].strip(): int(ln.split("|")[2]) for ln in lines[2:]}
    assert sum(n.values()) == sum(ST.shown(x, s) for x, s in zip(rs, ss))
    assert n["opportunistic"] == sum(s.insider_type == "opportunistic" for s in ss)
    opp = next(ln for ln in lines if ln.startswith("| opportunistic"))
    rou = next(ln for ln in lines if ln.startswith("| routine"))
    assert float(opp.split("|")[3].strip(" %")) > float(rou.split("|")[3].strip(" %")) + 1.5


def test_cmp_variants_table_runs():
    from test_scoring import Source, bars_from

    rs, ss = typed_history(per_year=900)
    start = date(2014, 6, 2)
    n = NYSE.business_days_between(start, date(2022, 6, 1))
    for x in rs:
        x.outcome.legs[60] = dict(x.outcome.legs[20], exit=NYSE.add(x.outcome.entry, 59))
    src = Source({"SPY": bars_from(start, [100.0] * n),
                  **{f"T{i}": bars_from(start, [10.0] * n) for i in range(97)}})
    text, sums = ST.variants_table(rs, ss, src, ST.CMP_VARIANTS)
    assert list(sums) == [c.name for c in ST.CMP_VARIANTS]
    opp, rou = (sums[k]["pooled"]["net_mean"] for k in ("opportunistic only",
                                                        "routine only (control)"))
    assert opp > 0.015 and abs(rou) < 0.015 and opp > rou + 0.015
    assert sums["insider type in Q"]["config"]["components"][-1] == "insider type"
    assert text.count("\n") == len(ST.CMP_VARIANTS) + 1


@pytest.mark.parametrize("year", [2015, 2020])
def test_classifier_cache_matches_a_fresh_classification(year):
    r = rows("1", *(date(y, m, 2) for y in range(2011, 2021) for m in (2, 7)))
    c = I.Classifier(r)
    first = c.kind_in_year("1", year)
    assert first == I.ROUTINE and c.kind_in_year("1", year) == first
