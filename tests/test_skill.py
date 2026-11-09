"""Insider skill records (tradetracker.skill) and the skill study (scoring.skill_study)."""

import gzip
import json
import random
from dataclasses import replace
from datetime import date, timedelta

import pytest

from tradetracker import skill as SK
from tradetracker.backtest import prices as bp
from tradetracker.backtest.returns import Outcome
from tradetracker.backtest.study import Result
from tradetracker.features.calendar import NYSE
from tradetracker.scoring import model as M
from tradetracker.scoring import signals as SG
from tradetracker.scoring import skill_run as SR
from tradetracker.scoring import skill_study as SS
from tradetracker.scoring import study as ST

from test_scoring import Source, bars_from, ev, sig

D = date


def ep(owner, known, excess, issuer="1", filed=None):
    return SK.Episode(owner, issuer, filed or known - timedelta(days=90), known, excess)


# ---------------------------------------------------------------- episodes


def test_owner_key_strips_leading_zeros_and_blanks():
    assert SK.owner_key("0001234") == "1234"
    assert SK.owner_key(1234) == "1234"
    assert SK.owner_key("") is None and SK.owner_key(None) is None and SK.owner_key("000") is None


def test_collapse_keeps_one_episode_per_owner_issuer_and_month():
    rows = [
        ("a", "1", D(2020, 3, 20), D(2020, 6, 15), 0.10),
        ("a", "1", D(2020, 3, 2), D(2020, 5, 28), 0.30),     # earliest in March: kept
        ("a", "1", D(2020, 4, 1), D(2020, 6, 25), 0.05),     # April: its own episode
        ("a", "2", D(2020, 3, 5), D(2020, 6, 1), -0.02),     # other issuer
        ("b", "1", D(2020, 3, 5), D(2020, 6, 1), 2.50),      # clipped to +100%
        (None, "1", D(2020, 3, 5), D(2020, 6, 1), 0.0),      # no owner: dropped
        ("c", "1", D(2020, 3, 5), D(2020, 6, 1), None),      # no outcome: dropped
    ]
    eps = SK.collapse(rows)
    got = {(e.owner, e.issuer, e.filed): e.excess for e in eps}
    assert got == {("a", "1", D(2020, 3, 2)): 0.30, ("a", "1", D(2020, 4, 1)): 0.05,
                   ("a", "2", D(2020, 3, 5)): -0.02, ("b", "1", D(2020, 3, 5)): 1.0}
    assert [e.known for e in eps] == sorted(e.known for e in eps)
    assert SK.clip(-3) == -1.0


# ---------------------------------------------------------------- records


def test_record_counts_only_episodes_that_exited_before_the_date():
    eps = [ep("a", D(2020, 1, 10), 0.10), ep("a", D(2020, 2, 10), -0.05),
           ep("a", D(2020, 3, 10), 0.20), ep("a", D(2020, 4, 10), 0.40)]
    book = SK.SkillBook(eps, k=3.0)
    assert book.record("a", D(2020, 1, 10)).n == 0           # exit on the day: not known
    r = book.record("a", D(2020, 3, 10))
    assert r.n == 2 and r.shrunk is None                      # below MIN_EPISODES
    assert r.raw == pytest.approx(0.025) and r.hit == 0.5
    r = book.record("a", D(2020, 3, 11))
    assert r.n == 3 and r.raw == pytest.approx(0.25 / 3) and r.hit == pytest.approx(2 / 3)
    # Fewer than MIN_PRIOR_EPISODES known overall: the prior is 0.
    assert r.prior == 0.0
    assert r.shrunk == pytest.approx(3 / (3 + 3.0) * (0.25 / 3))
    assert book.record("zzz", D(2021, 1, 1)) == SK.Record(0, None, None, 0.0, None)
    assert book.record(None, D(2021, 1, 1)).n == 0


def test_prior_is_the_mean_of_every_known_episode_once_there_are_enough():
    eps = [ep(f"o{i}", D(2019, 1, 1) + timedelta(days=i), 0.01 * (i % 3)) for i in range(150)]
    eps += [ep("a", D(2019, 9, 1), 0.2), ep("a", D(2019, 9, 2), 0.2), ep("a", D(2019, 9, 3), 0.2)]
    book = SK.SkillBook(eps, k=7.0)
    d = D(2019, 1, 1) + timedelta(days=120)      # 120 episodes known
    assert book.prior(d) == pytest.approx(sum(0.01 * (i % 3) for i in range(120)) / 120)
    assert book.prior(D(2019, 1, 1) + timedelta(days=50)) == 0.0
    r = book.record("a", D(2020, 1, 1))
    p = (sum(0.01 * (i % 3) for i in range(150)) + 0.6) / 153
    assert r.prior == pytest.approx(p)
    assert r.shrunk == pytest.approx(p + 3 / 10 * (0.2 - p))


def test_records_are_point_in_time_whatever_happens_later():
    rng = random.Random(1)
    owners = [f"o{i}" for i in range(40)]
    eps = [ep(rng.choice(owners), D(2016, 1, 1) + timedelta(days=rng.randrange(2000)),
              rng.gauss(0, 0.2), issuer=str(rng.randrange(30))) for _ in range(3000)]
    book = SK.SkillBook(eps, k=10.0)
    cut = D(2019, 6, 1)
    later = [replace(e, excess=-1.0) if e.known >= cut else e for e in eps]
    later += [ep("o1", cut + timedelta(days=i), 1.0) for i in range(30)]
    other = SK.SkillBook(later, k=10.0)
    for o in owners:
        for d in (D(2017, 1, 1), D(2018, 7, 4), cut):
            assert book.record(o, d) == other.record(o, d)


def test_estimate_k_recovers_planted_skill_and_caps_no_skill():
    rng = random.Random(4)
    eps = []
    for i in range(600):
        mu = rng.gauss(0, 0.05)                  # tau = 0.05
        for j in range(6):
            eps.append(ep(f"s{i}", D(2015, 1, 1) + timedelta(days=j * 30 + i % 30),
                          mu + rng.gauss(0, 0.15)))      # sigma = 0.15 -> k = 9
    k = SK.estimate_k(eps, D(2017, 1, 1))
    assert 6 < k < 13
    noise = [replace(e, excess=rng.gauss(0, 0.15)) for e in eps]
    assert SK.estimate_k(noise, D(2017, 1, 1)) >= 40
    # Only episodes known before the cutoff count; too few groups: the default.
    assert SK.estimate_k(eps, D(2015, 1, 1)) == SK.DEFAULT_K


# ---------------------------------------------------------------- sources


def res(e, legs: dict, entry=None, status="ok", cost=0.004):
    entry = entry or e.entry_date
    o = Outcome(status, entry=entry, entry_open=10.0, drift_pct=0.0, dollar_volume=5e6,
                cost=cost, benchmark="SPY")
    for h, r in legs.items():
        o.legs[h] = {"stock": r, "spy": 0.0, "sector": 0.0, "iwm": 0.0,
                     "exit": NYSE.add(entry, h - 1), "flags": []}
    return Result(e, o, "SPY")


def test_rows_from_results_keep_priced_non_plan_buys_with_their_exit_day():
    a = res(ev(filer_id=1, issuer_cik="0007"), {20: 0.05, 60: 0.1})
    rows = list(SK.rows_from_results(
        [a, res(ev(is_10b5_1=True), {20: 1.0}), res(ev(side="sell"), {20: 1.0}),
         res(ev(), {20: 1.0}, status="no_prices"), res(ev(filer_id=9), {60: 0.3})],
        20, owner_keys={1: "00042", 9: "43"}))
    assert rows == [("42", "7", a.event.filing_date, a.outcome.legs[20]["exit"], 0.05)]
    rows = list(SK.rows_from_results([a], 60))
    assert rows[0][0] == "filer:1" and rows[0][4] == 0.1


def test_price_outcome_by_hand():
    # Filed on Friday 2014-03-07: bought at Monday's open, sold at the 3rd session's close.
    stock = bars_from(D(2014, 3, 3), [10, 10, 10, 10, 10, 11, 12, 13, 14],
                      opens=[10, 10, 10, 10, 10, 10, 12, 13, 14])
    spy = bars_from(D(2014, 3, 3), [100] * 6 + [101, 102, 103],
                    opens=[100] * 9)
    src = Source({"X": stock, "SPY": spy})
    known, x = SK.price_outcome(src, "X", D(2014, 3, 7), 3)
    assert known == D(2014, 3, 12)
    assert x == pytest.approx((13 / 10 - 1) - (102 / 100 - 1))
    assert SK.price_outcome(src, "X", D(2014, 3, 7), 10) is None     # not enough bars
    assert SK.price_outcome(src, "NOPE", D(2014, 3, 7), 3) is None
    assert SK.price_outcome(src, "X", D(2014, 1, 2), 3) is None      # entry too far away


def test_history_buys_before_the_database_start(tmp_path):
    p = tmp_path / "h.csv.gz"
    with gzip.open(p, "wt") as fh:
        fh.write("owner_cik,issuer_cik,trade_date,filed,side\n"
                 "0001,0005,2014-03-03,2014-03-05,buy\n"
                 "1,5,2014-03-04,2014-03-05,buy\n"         # same filing day: once
                 "1,5,2014-03-03,2014-03-05,sell\n"
                 "2,6,2014-12-30,2015-01-02,buy\n"         # filed after the cutoff
                 ",6,2014-05-01,2014-05-02,buy\n")
    assert SK.load_history_buys(p, D(2015, 1, 1)) == [("1", "5", D(2014, 3, 5))]


def test_build_books_adds_priced_history_buys(tmp_path):
    p = tmp_path / "h.csv"
    p.write_text("owner_cik,issuer_cik,trade_date,filed,side\n"
                 "77,5,2014-03-03,2014-03-07,buy\n")
    closes = [10.0] * 5 + [10.0 + 0.1 * i for i in range(1, 80)]
    src = Source({"X": bars_from(D(2014, 3, 3), closes, opens=[10.0] * len(closes)),
                  "SPY": bars_from(D(2014, 3, 3), [100.0] * len(closes))})
    a = res(ev(filer_id=3, issuer_cik="5", filing_date=D(2016, 2, 1)), {20: 0.02, 60: 0.04})
    books = SK.build_books([a], src, {3: "77"}, history=p, tickers={"5": "X"})
    assert set(books) == {20, 60}
    e20 = books[20].episodes
    assert [(e.owner, e.filed) for e in e20] == [("77", D(2014, 3, 7)), ("77", D(2016, 2, 1))]
    assert e20[0].excess == pytest.approx(closes[5 + 19] / 10 - 1)
    assert books[60].episodes[0].known == NYSE.add(D(2014, 3, 10), 59)


# ---------------------------------------------------------------- signals


def test_signals_build_and_attach_fill_the_same_skill_fields():
    rs = []
    for i, r in enumerate([0.1, 0.2, 0.3, -0.1]):
        d = NYSE.on_or_before(D(2018, 1, 10) + timedelta(days=70 * i))
        rs.append(res(ev(filer_id=5, issuer_cik=str(i), filing_date=d, first_trade_date=d,
                         last_trade_date=d), {20: r, 60: r}))
    books = SK.build_books(rs, Source({}), {5: "0099"})
    built = SG.build(rs, Source({}), filer_keys={5: "0099"}, skills=books)
    att = SR.attach([SG.Signals(None, 0, None, 0, 0, None, None, 0, None, None)] * 4, rs,
                    books, {5: "0099"})
    for b, a in zip(built, att):
        for f in ("skill_20", "skill_60", "skill_n20", "skill_n60", "skill_raw60", "skill_hit60"):
            assert getattr(b, f) == getattr(a, f)
    assert [s.skill_n20 for s in built] == [0, 1, 2, 3]
    assert built[3].skill_20 is not None and built[3].skill_hit20 == 1.0
    assert built[3].skill(20) == built[3].skill_20 and built[3].skill_n(60) == built[3].skill_n60
    assert built[3].skill(5) is None


@pytest.mark.parametrize("h,v,expected", [
    (60, None, "no record"), (60, -0.03, "< -2%"), (60, -0.01, "-2 to 0%"),
    (60, 0.0, "0 to 2%"), (60, 0.02, ">= 2%"), (20, -0.011, "< -1%"), (20, 0.015, ">= 1%"),
])
def test_skill_buckets(h, v, expected):
    assert M.bucket(f"skill {h}", sig(**{f"skill_{h}": v})) == expected


def test_skill_gate_semantics():
    proven = ST.Config(h=60, skill_filter="proven")
    no_losers = ST.Config(h=60, skill_filter="no losers")
    assert ST.skill_ok(sig(skill_60=0.01), proven)
    assert not ST.skill_ok(sig(skill_60=0.0), proven)
    assert not ST.skill_ok(sig(), proven)
    assert ST.skill_ok(sig(), no_losers) and ST.skill_ok(sig(skill_60=0.0), no_losers)
    assert not ST.skill_ok(sig(skill_60=-0.001), no_losers)
    assert ST.skill_ok(sig(), ST.Config())
    assert not ST.skill_ok(sig(skill_20=-1, skill_60=1), replace(no_losers, h=20))
    with pytest.raises(ValueError):
        ST.skill_ok(sig(), ST.Config(skill_filter="nope"))
    assert "proven insiders only" in proven.label()
    assert "skill 60 in Q" in ST.Config(components=("drawdown", "skill 60")).label()


# ---------------------------------------------------------------- study


def skilled_history(seed=11, edge=0.04, years=range(2015, 2022), owners=300, per_owner_year=3):
    """Half the insiders earn `edge` on every buy (at 20 and 60 days), half earn -edge/2.
    Records come from these same buys, so the study has to find persistence."""
    rng = random.Random(seed)
    rs, ss = [], []
    for y in years:
        for o in range(owners):
            mu = edge if o % 2 == 0 else -edge / 2
            for j in range(per_owner_year):
                d = NYSE.on_or_before(D(y, 1, 5) + timedelta(days=(o * 7 + j * 120) % 355))
                e = ev(filer_id=o, issuer_cik=str(1000 + o), ticker=f"T{o}", filing_date=d,
                       first_trade_date=d, last_trade_date=d)
                r = mu + rng.gauss(0, 0.06)
                rs.append(res(e, {20: r / 2, 60: r}))
                ss.append(sig(drawdown=rng.uniform(-0.6, 0), drift=rng.uniform(-0.05, 0.1),
                              market_cap=1e9))
    return rs, ss


def flat_source(rs):
    start = D(2014, 12, 1)
    n = 1900
    bars = {x.event.ticker: bars_from(start, [10.0] * n) for x in rs}
    bars["SPY"] = bars_from(start, [100.0] * n)
    return Source(bars)


@pytest.fixture(scope="module")
def skilled():
    rs, ss = skilled_history()
    keys = {o: str(5000 + o) for o in range(300)}
    books = SK.build_books(rs, flat_source(rs), keys)
    return rs, SR.attach(ss, rs, books, keys), books, keys


def test_persistence_finds_planted_skill(skilled):
    rs, ss, books, _ = skilled
    cfg = ST.Config(h=60, min_cap=50e6)
    _, out = SS.persistence(rs, ss, cfg)
    assert out[">= 2%"]["net_mean"] > 0.02 > out["all shown"]["net_mean"]
    assert out["< -2%"]["net_mean"] < 0
    assert out["consistent winners"]["net_mean"] > out["record, not consistent"]["net_mean"]
    _, sp = SS.spread_by_year(rs, ss, cfg)
    pos, cnt = sp["years_positive"]["top"]
    assert cnt >= 5 and pos == cnt
    assert books[60].k < 20


def test_gate_flags_a_subset_of_the_ungated_model(skilled):
    rs, ss, _, _ = skilled
    base = ST.Config(h=60, min_cap=50e6, components=ST.TWO_INPUTS)
    gated = replace(base, skill_filter="proven")
    fb, fg = ST.walk_forward(rs, ss, cfg=base), ST.walk_forward(rs, ss, cfg=gated)
    for a, b in zip(fb, fg):
        assert a.model.tables == b.model.tables and a.model.threshold == b.model.threshold
        flagged = {(x.result.event.ticker, x.result.outcome.entry) for x in a.scored if x.above}
        for x in b.scored:
            if x.above:
                assert (x.result.event.ticker, x.result.outcome.entry) in flagged
                assert x.signals.skill_60 is not None and x.signals.skill_60 > 0


def test_gate_placebo_beats_random_gates_with_planted_skill(skilled):
    rs, ss, _, _ = skilled
    src = flat_source(rs)
    cfg = ST.Config(h=60, min_cap=50e6, components=ST.TWO_INPUTS, skill_filter="proven")
    _, out = SS.gate_placebo(rs, ss, src, cfg, draws=60)
    assert 0 < out["n_gated"] < out["n_base"]
    assert out["gated_mean"] > out["placebo_mean_median"]
    assert out["p_mean"] < 0.05


def test_gate_placebo_finds_nothing_without_skill():
    rs, ss = skilled_history(edge=0.0, seed=12)
    keys = {o: str(5000 + o) for o in range(300)}
    src = flat_source(rs)
    books = SK.build_books(rs, src, keys)
    ss = SR.attach(ss, rs, books, keys)
    assert books[60].k >= 40
    cfg = ST.Config(h=60, min_cap=50e6, components=ST.TWO_INPUTS, skill_filter="proven")
    _, out = SS.gate_placebo(rs, ss, src, cfg, draws=60)
    assert out["p_mean"] > 0.02


def test_skill_in_q_gets_weight_when_skill_is_planted(skilled):
    rs, ss, _, _ = skilled
    cfg = ST.Config(h=60, min_cap=50e6, components=("drawdown", "skill 60"))
    folds = ST.walk_forward(rs, ss, cfg=cfg)
    assert folds and all(f.model.weights["skill 60"] > 0 for f in folds[1:])


def test_skill_run_writes_the_outputs(skilled, tmp_path):
    rs, _, _, keys = skilled
    _, ss = skilled_history()
    configs = [ST.SKILL_VARIANTS[0], ST.SKILL_VARIANTS[2]]
    s = SR.run(rs, ss, flat_source(rs), keys, tmp_path, placebo_draws=20, configs=configs)
    for f in ("report.md", "summary.json", "skill-pool.csv.gz", "episodes.csv.gz",
              "scored-proven-60-days.csv.gz"):
        assert (tmp_path / f).exists()
    text = (tmp_path / "report.md").read_text()
    assert "Persistence at 60 days" in text and "Random-gate check" in text
    js = json.loads((tmp_path / "summary.json").read_text())
    assert "40-slot IR" in text and js["variants"][configs[1].name]["portfolio_extra"]["slots"] == 40
    with gzip.open(tmp_path / "skill-pool.csv.gz", "rt") as fh:
        head = fh.readline().strip().split(",")
        n = sum(1 for _ in fh)
    assert head == SR.POOL_COLUMNS and n == s["pool_rows"] > 0


def test_cli_registers_skill_study():
    from tradetracker import cli
    with pytest.raises(SystemExit) as e:
        cli.main(["skill-study", "--help"])
    assert e.value.code == 0


def test_issuer_tickers_take_the_first_filing_and_break_ties_by_use():
    rs = [res(ev(issuer_cik="0001099574", ticker="ETBI", filing_date=D(2015, 1, 21)), {20: 0}),
          res(ev(issuer_cik="1099574", ticker="EBTI", filing_date=D(2015, 1, 21)), {20: 0}),
          res(ev(issuer_cik="1099574", ticker="EBTI", filing_date=D(2016, 1, 21)), {20: 0}),
          res(ev(issuer_cik="5", ticker="NEW", filing_date=D(2020, 1, 2)), {20: 0}),
          res(ev(issuer_cik="5", ticker="OLD", filing_date=D(2015, 1, 2)), {20: 0})]
    assert SR.issuer_tickers(rs) == {"1099574": "EBTI", "5": "OLD"}
    assert SR.issuer_tickers(list(reversed(rs))) == {"1099574": "EBTI", "5": "OLD"}
