"""The backtest on real SEC filings and real prices, checked by hand.

Filings: tests/fixtures/real/2024q1_features_form345.zip (CTBI, PLCE, NBIX, RMCF).
Prices: tests/fixtures/real/backtest_bars, Nov 2023 to Jul 2024, from the Tiingo cache
(CTBI, PLCE, RMCF, SPY and the sector ETFs) and Yahoo (NBIX, IWM). Every expected number
below is typed from those CSV rows, not produced by the code under test.
"""

from datetime import date
from pathlib import Path

import pytest

from tradetracker import store
from tradetracker.backtest import events, prices, sectors, study
from tradetracker.edgar.bulk import parse_quarter
from tradetracker.features import job

REAL = Path(__file__).parent / "fixtures" / "real"


@pytest.fixture
def results(conn):
    for f in parse_quarter(REAL / "2024q1_features_form345.zip"):
        store.save_filing(conn, f)
    store.link_amendments(conn)
    conn.commit()
    job.run(conn, as_of=date(2024, 3, 28))
    evs, ex = events.load_events(conn)
    src = prices.CachePrices(REAL / "backtest_bars")
    res = study.run_study(evs, src, sectors.load_sic(REAL / "backtest_sic.csv"))
    return res, ex


def _find(res, ticker, filing_date, **kw):
    out = [r for r in res if r.event.ticker == ticker and r.event.filing_date == filing_date
           and all(getattr(r.event, k) == v for k, v in kw.items())]
    return out


def test_sample_counts_leave_out_amendments_and_repeat_filings(results):
    res, ex = results
    # 99 lines: 15 restate an earlier line (RMCF joint filers, 4/A repeats), 10 more are
    # 4/A lines; the other 74 lines make 32 events.
    assert ex.counts["lines"] == 99
    assert ex.counts["duplicate"] == 15 and ex.counts["amendment"] == 10
    assert len(res) == ex.counts["events"] == 32
    assert all(r.ok for r in res)


def test_ctbi_board_buy_filed_25_jan(results):
    res, _ = results
    ctbi = _find(res, "CTBI", date(2024, 1, 25))
    assert len(ctbi) == 11  # one event per insider
    one = [r for r in ctbi if r.event.lag_days == 1][0]
    o = one.outcome
    # Filed Thursday 25 Jan: entry at the open of Friday 26 Jan.
    assert o.entry == date(2024, 1, 26)
    assert o.benchmark == "XLF"  # SIC 6022, state commercial bank
    # CTBI adjOpen 26 Jan 38.338788221687736; adjClose 1 Feb 37.26530215148048,
    # 23 Feb 36.020419145693914, 22 Apr 38.43560010337344.
    assert [o.legs[h]["exit"] for h in (5, 20, 60)] == [
        date(2024, 2, 1), date(2024, 2, 23), date(2024, 4, 22)]
    assert o.legs[5]["stock"] == pytest.approx(37.26530215148048 / 38.338788221687736 - 1)
    assert o.legs[20]["stock"] == pytest.approx(36.020419145693914 / 38.338788221687736 - 1)
    assert o.legs[60]["stock"] == pytest.approx(38.43560010337344 / 38.338788221687736 - 1)
    # SPY adjOpen 26 Jan 472.1560339551387, adjClose 23 Feb 491.77473255012865.
    spy20 = 491.77473255012865 / 472.1560339551387 - 1
    assert o.legs[20]["spy"] == pytest.approx(spy20)
    # XLF adjOpen 26 Jan 36.89565615992035, adjClose 23 Feb 38.67086978530013.
    assert o.legs[20]["sector"] == pytest.approx(38.67086978530013 / 36.89565615992035 - 1)
    # IWM (Yahoo) adjOpen 26 Jan 191.14009634, adjClose 23 Feb 194.05247498.
    assert o.legs[20]["iwm"] == pytest.approx(194.05247498 / 191.14009634 - 1)
    assert one.r(20) == pytest.approx(36.020419145693914 / 38.338788221687736 - 1 - spy20)
    # Drift: adjClose 24 Jan (trade) 37.59907513129282 to 25 Jan 37.950889893797715.
    assert o.drift_pct == pytest.approx(100 * (37.950889893797715 / 37.59907513129282 - 1))
    # Form 4 price vs CTBI's raw close of 41.68 on 24 Jan.
    assert o.price_ratio == pytest.approx(one.event.avg_price / 41.68)


def test_ctbi_late_line_sets_the_event_lag_and_drift(results):
    res, _ = results
    # Smith's filing has a 2 Jan purchase (16 SEC days late) and two 24 Jan purchases.
    smith = [r for r in _find(res, "CTBI", date(2024, 1, 25)) if r.event.lag_days == 16][0]
    assert smith.event.n_lines == 3 and smith.event.first_trade_date == date(2024, 1, 2)
    assert study.lag_bucket(smith.event.lag_days) == 3
    # Drift from the 2 Jan close (feature test: 39.6738801409) to the 25 Jan close.
    assert smith.outcome.drift_pct == pytest.approx(100 * (37.950889893797715 / 39.6738801409 - 1),
                                                    abs=1e-6)


def test_plce_run_up_before_disclosure(results):
    res, _ = results
    (plce,) = _find(res, "PLCE", date(2024, 2, 15))
    o = plce.outcome
    # Bought 13 Feb at close 11.29; the filing-date close (15 Feb) was 26.29.
    assert o.drift_pct == pytest.approx(100 * (26.29 / 11.29 - 1))
    assert study.stats.bucket_of(o.drift_pct, [0.0, 1.0, 2.0, 5.0]) == 4
    # Entry 16 Feb open 26.37; 20-day exit 15 Mar close 13.01 (no dividends, adj = raw).
    assert o.entry == date(2024, 2, 16) and o.legs[20]["exit"] == date(2024, 3, 15)
    assert o.legs[20]["stock"] == pytest.approx(13.01 / 26.37 - 1)
    # XLY (SIC 5641) adjOpen 16 Feb 87.62825209000549, adjClose 15 Mar 87.01249391905347.
    assert o.benchmark == "XLY"
    assert o.legs[20]["sector"] == pytest.approx(87.01249391905347 / 87.62825209000549 - 1)
    assert o.legs[20]["spy"] == pytest.approx(495.23651001800016 / 485.8194020289446 - 1)


def test_plce_amendments_are_not_events(results):
    res, _ = results
    plce_dates = sorted(r.event.filing_date for r in res if r.event.ticker == "PLCE")
    # Three original Form 4s; the 4/As filed later (and dated by the 4/A) are out.
    assert plce_dates == [date(2024, 2, 13), date(2024, 2, 14), date(2024, 2, 15)]


def test_nbix_plan_sales_are_flagged_sells(results):
    res, _ = results
    nbix = [r for r in res if r.event.ticker == "NBIX"]
    assert len(nbix) == 4 and all(r.event.side == "sell" and r.event.is_10b5_1 for r in nbix)
    # A sell's signal is the negative excess return: NBIX rose, so the sale "lost".
    r = _find(res, "NBIX", date(2024, 2, 22))[0]
    assert r.sign == -1 and r.r(20) == pytest.approx(-(r.outcome.legs[20]["stock"]
                                                       - r.outcome.legs[20]["spy"]))
    assert r.r(20) < 0


def test_rmcf_joint_filers_are_one_event(results):
    res, _ = results
    # The same buys were filed twice by joint filers; the repeat is a duplicate.
    rmcf = _find(res, "RMCF", date(2024, 2, 14))
    assert len(rmcf) == 1 and rmcf[0].event.n_lines == 3


def test_report_on_real_events(results, tmp_path):
    res, ex = results
    text = study.render(res, ex, "fixture")
    assert "Buys, liquid (primary)" in text
    study.write_events(res, tmp_path / "events.csv.gz")
    assert (tmp_path / "events.csv.gz").stat().st_size > 0


def _snapshot(r, horizons=(5, 20)):
    o = r.outcome
    return (o.status, o.entry, o.drift_pct, o.price_ratio, o.dollar_volume, o.benchmark,
            {h: dict(o.legs[h]) for h in horizons if h in o.legs})


def test_cutting_prices_after_a_date_changes_no_earlier_result(results, tmp_path):
    """Truncate every price file at 1 Apr 2024: events whose 20-day exit is earlier match."""
    import gzip

    res, _ = results
    (tmp_path / "bars").mkdir()
    for f in (REAL / "backtest_bars" / "bars").glob("*.csv.gz"):
        lines = gzip.decompress(f.read_bytes()).decode().splitlines()
        keep = [lines[0]] + [x for x in lines[1:] if x[:10] < "2024-04-01"]
        (tmp_path / "bars" / f.name).write_bytes(gzip.compress("\n".join(keep).encode()))
    cut = prices.CachePrices(tmp_path)
    again = study.run_study([r.event for r in res], cut,
                            sectors.load_sic(REAL / "backtest_sic.csv"))
    checked = 0
    for a, b in zip(res, again):
        if a.outcome.legs[20]["exit"] < date(2024, 4, 1):
            assert _snapshot(a) == _snapshot(b)
            checked += 1
        else:
            assert 60 not in b.outcome.legs
    assert checked >= 25


def test_filings_made_later_change_no_earlier_event(conn):
    """Load only filings up to 14 Feb, then everything: events up to 14 Feb match."""
    filings = list(parse_quarter(REAL / "2024q1_features_form345.zip"))
    src = prices.CachePrices(REAL / "backtest_bars")
    sic = sectors.load_sic(REAL / "backtest_sic.csv")

    def run(fs):
        conn.execute("TRUNCATE trades, filings, filers, trade_features CASCADE")
        for f in fs:
            store.save_filing(conn, f)
        store.link_amendments(conn)
        conn.commit()
        job.run(conn, recompute=True, as_of=date(2024, 3, 28))
        evs, _ = events.load_events(conn)
        return {(r.event.ticker, r.event.side, r.event.filing_date, r.event.first_trade_date,
                 r.event.n_lines, round(r.event.value, 2)): _snapshot(r)
                for r in study.run_study(evs, src, sic)}

    early = run([f for f in filings if f.filed_at.date() <= date(2024, 2, 14)])
    full = run(filings)
    assert len(early) >= 15
    for k, v in early.items():
        assert full[k] == v
