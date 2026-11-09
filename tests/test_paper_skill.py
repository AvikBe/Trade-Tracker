"""The proven-insider gate of the shadow paper book (paper.skill_gate)."""

import gzip
from datetime import date, timedelta

import pytest

from tradetracker import skill as SK
from tradetracker.features.calendar import NYSE
from tradetracker.paper import engine as E
from tradetracker.paper import skill_gate as G

from test_paper import book, buy_filing, flat_market, price_on, write_day


def write_episodes(path, rows):
    with gzip.open(path, "wt") as fh:
        fh.write("horizon,owner,issuer,filed,known,excess\n")
        for r in rows:
            fh.write(",".join(str(v) for v in r) + "\n")


class Counting:
    def __init__(self, bars):
        self.bars, self.calls = bars, 0

    def get(self, t):
        self.calls += 1
        return self.bars.get(t)


def test_study_episodes_load_one_horizon(tmp_path):
    p = tmp_path / "e.csv.gz"
    write_episodes(p, [(20, "7", "1", "2020-01-02", "2020-01-31", 0.1),
                       (60, "7", "1", "2020-01-02", "2020-03-30", -0.2)])
    (e,) = G.load_study_episodes(p)
    assert (e.owner, e.known, e.excess) == ("7", date(2020, 3, 30), -0.2)


def test_matured_needs_the_60th_close_before_today():
    filed = date(2026, 7, 1)
    exit_day = NYSE.add(NYSE.add(filed, 1), 59)
    assert not G.matured(filed, exit_day)
    assert G.matured(filed, NYSE.add(exit_day, 1))


def test_live_rows_price_matured_cached_buys_once(tmp_path):
    bars, ds = flat_market(["AAA", "BBB"])
    d0, d1, d2 = ds[100], ds[101], ds[400]
    cache = tmp_path / "filings"
    write_day(cache, d0, [buy_filing("AAA", 5, d0, price=price_on(bars, "AAA", d0))])
    write_day(cache, d1, [buy_filing("BBB", 6, d1, price=price_on(bars, "BBB", d1), plan=True)])
    write_day(cache, d2, [buy_filing("AAA", 5, d2, price=price_on(bars, "AAA", d2))])  # not matured
    done: set = set()
    prices = Counting(bars)
    rows = G.live_rows(cache, ds[50], ds[420], prices, done)
    assert [(o, f) for o, _, f, _, _ in rows] == [("5", d0)]
    known, x = SK.price_outcome(prices, "AAA", d0, 60)
    assert rows[0][3:] == (known, x)
    assert done == {d0.isoformat(), d1.isoformat()}
    assert G.live_rows(cache, ds[50], ds[420], prices, done) == []
    # Days on or before the study's last episode are the study's.
    assert [f for _, _, f, _, _ in G.live_rows(cache, d1, ds[480], prices, set())] == [d2]


def test_gate_passes_only_positive_records_and_keeps_live_episodes(tmp_path):
    bars, ds = flat_market(["AAA"])
    p = tmp_path / "e.csv.gz"
    # Insider 9 beat SPY three times, insider 8 lost three times, insider 7 has two.
    rows = []
    for owner, x, n in (("9", 0.2, 3), ("8", -0.2, 3), ("7", 0.5, 2)):
        for i in range(n):
            rows.append((60, owner, "1", ds[10 + i * 30], ds[70 + i * 30], x))
    write_episodes(p, rows)
    state = tmp_path / "book"
    state.mkdir()
    cache = tmp_path / "filings"
    d = ds[200]
    write_day(cache, d, [buy_filing("AAA", 7, d, price=price_on(bars, "AAA", d))])
    gate = G.SkillGate.load(p, state, cache, ds[300], Counting(bars))
    when = ds[299]
    ok, note = gate.passes("0009", when)
    assert ok and note.startswith("record +") and "n=3" in note
    assert gate.passes("8", when)[0] is False
    # Insider 7 now has a third, live episode (flat stock: excess 0, still positive mean).
    r = gate.record("7", when)
    assert r.n == 3 and r.shrunk is not None and r.shrunk > 0
    assert gate.passes("7", ds[150]) == (False, "no record (n=2)")
    assert gate.passes("123", when) == (False, "no record (n=0)")
    # A second load reads the saved live episodes instead of pricing again.
    prices = Counting(bars)
    again = G.SkillGate.load(p, state, cache, ds[300], prices)
    assert prices.calls == 0 and again.record("7", when) == r
    assert (state / G.LIVE_FILE).exists() and (state / G.DONE_FILE).exists()


def test_book_with_gate_buys_only_proven_insiders(tmp_path):
    bars, ds = flat_market(["AAA", "BBB"])
    filed, entry = ds[320], ds[321]
    write_day(tmp_path / "filings", filed, [
        buy_filing("AAA", 1, filed, price=price_on(bars, "AAA", filed)),
        buy_filing("BBB", 2, filed, price=price_on(bars, "BBB", filed)),
    ])
    eps = [SK.Episode("1", "x", ds[10 + 30 * i], ds[70 + 30 * i], 0.1) for i in range(3)]
    gate = G.SkillGate(SK.SkillBook(eps, G.DEFAULT_K), ds[200])
    b = book(tmp_path, bars)
    b.cfg.skill_gate = gate
    b.cfg.state_dir.mkdir(parents=True)
    E.save_state(b.state_path, E.new_state("dry-run", 100_000.0, filed - timedelta(days=1)))
    r = b.run(entry)
    assert [d.ticker for d in r.candidates] == ["AAA"]
    assert [o["ticker"] for o in r.submitted] == ["AAA"]
    dec = (b.cfg.state_dir / "decisions.csv").read_text()
    assert "unproven_insider" in dec
    # Without the gate both are candidates.
    b2 = book(tmp_path / "plain", bars)
    b2.cfg.state_dir.mkdir(parents=True)
    E.save_state(b2.state_path, E.new_state("dry-run", 100_000.0, filed - timedelta(days=1)))
    b2.filings_cache = tmp_path / "filings"
    assert {d.ticker for d in b2.run(entry).candidates} == {"AAA", "BBB"}


def test_cli_paper_trade_has_the_skill_flag():
    from tradetracker import cli
    with pytest.raises(SystemExit):
        cli.main(["paper-trade", "--help"])
