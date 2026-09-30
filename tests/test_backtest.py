"""Backtest unit tests: entry timing, returns, costs, point-in-time checks, statistics."""

import gzip
import math
import random
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from tradetracker.backtest import prices as bp
from tradetracker.backtest import returns, stats, study
from tradetracker.backtest.events import Event, entry_date
from tradetracker.backtest.sectors import sector_etf
from tradetracker.backtest.yahoo import parse_chart, yahoo_symbol
from tradetracker.features.calendar import NYSE

ET = ZoneInfo("America/New_York")


# ---------------------------------------------------------------- helpers

def series(start: date, closes: list[float], opens: list[float] | None = None,
           volume: float = 1e6, adj: float = 1.0) -> bp.Bars:
    """Bars on consecutive NYSE days from `start`; adj scales adjusted vs raw prices."""
    d = start if NYSE.is_business_day(start) else NYSE.add(start, 1)
    out = []
    for i, c in enumerate(closes):
        o = opens[i] if opens else c
        out.append(bp.Bar(d, o, c, o * adj, c * adj, volume))
        d = NYSE.add(d, 1)
    return bp.Bars(out)


def flat(start: date, n: int, price: float = 100.0) -> bp.Bars:
    return series(start, [price] * n)


def event(**kw) -> Event:
    base = dict(filer_id=1, issuer_cik="1", ticker="X", side="buy",
                filing_date=date(2024, 1, 10), accepted_at=None,
                first_trade_date=date(2024, 1, 8), last_trade_date=date(2024, 1, 8),
                lag_days=2, lag_ratio=1.0, value=10_000.0, shares=100.0)
    base.update(kw)
    return Event(**base)


# ---------------------------------------------------------------- entry timing

@pytest.mark.parametrize("filed, expected", [
    (date(2024, 1, 10), date(2024, 1, 11)),   # Wednesday -> Thursday
    (date(2024, 1, 12), date(2024, 1, 16)),   # Friday -> Tuesday (MLK Day)
    (date(2024, 3, 28), date(2024, 4, 1)),    # Thursday before Good Friday -> Monday
    (date(2024, 3, 29), date(2024, 4, 1)),    # filed on Good Friday (EDGAR open)
    (date(2023, 10, 9), date(2023, 10, 10)),  # Columbus Day: NYSE open, next day anyway
])
def test_bulk_history_enters_at_the_next_trading_day_open(filed, expected):
    assert entry_date(filed, None) == expected


@pytest.mark.parametrize("accepted, expected", [
    (datetime(2024, 1, 10, 8, 0, tzinfo=ET), date(2024, 1, 10)),    # before the open
    (datetime(2024, 1, 10, 9, 29, 59, tzinfo=ET), date(2024, 1, 10)),
    (datetime(2024, 1, 10, 9, 30, tzinfo=ET), date(2024, 1, 11)),   # at the open: too late
    (datetime(2024, 1, 10, 15, 0, tzinfo=ET), date(2024, 1, 11)),
    (datetime(2024, 1, 10, 21, 45, tzinfo=ET), date(2024, 1, 11)),  # after hours
    (datetime(2024, 3, 29, 7, 0, tzinfo=ET), date(2024, 4, 1)),     # Good Friday morning
    (datetime(2024, 1, 13, 12, 0, tzinfo=ET), date(2024, 1, 16)),   # Saturday
    # UTC input is converted: 13:00 UTC is 08:00 Eastern in January.
    (datetime(2024, 1, 10, 13, 0, tzinfo=ZoneInfo("UTC")), date(2024, 1, 10)),
])
def test_live_filings_enter_at_the_first_open_after_acceptance(accepted, expected):
    assert entry_date(date(2024, 1, 10), accepted) == expected


# ---------------------------------------------------------------- returns

def test_returns_run_from_entry_open_to_the_close_h_sessions_later():
    start = date(2024, 1, 2)
    closes = [100 + i for i in range(80)]
    opens = [c - 0.5 for c in closes]
    stock = series(start, closes, opens)
    spy = series(start, [200 + 0.5 * i for i in range(80)], [200 + 0.5 * i - 0.25 for i in range(80)])
    e = event(filing_date=date(2024, 1, 10), first_trade_date=date(2024, 1, 8),
              last_trade_date=date(2024, 1, 8), value=10_400.0, shares=100.0)
    out = returns.evaluate(e, stock, spy, None, "SPY", None, horizons=(1, 5, 20))
    assert out.status == "ok" and out.entry == date(2024, 1, 11)
    k = stock.dates.index(date(2024, 1, 11))
    for h in (1, 5, 20):
        leg = out.legs[h]
        assert leg["exit"] == stock.dates[k + h - 1] == NYSE.add(date(2024, 1, 11), h - 1)
        assert leg["stock"] == pytest.approx(closes[k + h - 1] / opens[k] - 1)
        spy_open = 200 + 0.5 * k - 0.25
        assert leg["spy"] == pytest.approx((200 + 0.5 * (k + h - 1)) / spy_open - 1)
        assert out.signal(h) == pytest.approx(leg["stock"] - leg["spy"])
    # Sector falls back to SPY when no sector ETF is given.
    assert out.benchmark == "SPY" and out.legs[5]["sector"] == out.legs[5]["spy"]
    # Drift: close on 8 Jan (first trade) to the close before entry (10 Jan).
    i8, i10 = stock.dates.index(date(2024, 1, 8)), stock.dates.index(date(2024, 1, 10))
    assert out.drift_pct == pytest.approx(100 * (closes[i10] / closes[i8] - 1))
    assert out.price_ratio == pytest.approx(104.0 / closes[i8])


def test_sell_signal_is_the_negative_excess_return_and_drift_flips():
    start = date(2024, 1, 2)
    stock = series(start, [100 - 0.5 * i for i in range(40)])
    spy = flat(start, 40, 400)
    e = event(side="sell", value=9_950.0, shares=100.0)
    out = returns.evaluate(e, stock, spy, None, "SPY", None, horizons=(5,))
    assert out.legs[5]["stock"] < 0
    assert out.signal(5, sign=-1) == pytest.approx(-(out.legs[5]["stock"] - out.legs[5]["spy"]))
    assert out.drift_pct > 0  # the stock fell before disclosure: move already captured
    r = study.Result(e, out, "SPY")
    assert r.r(5) > 0


def test_adjusted_prices_drive_returns_and_raw_prices_drive_the_ticker_check():
    # A 2:1 split-adjusted history: raw prices are twice the adjusted ones.
    start = date(2024, 1, 2)
    stock = series(start, [50.0] * 40, adj=0.5)
    e = event(value=5_000.0, shares=100.0)  # Form 4 price $50, the raw price that day
    out = returns.evaluate(e, stock, flat(start, 40), None, "SPY", None, horizons=(5,))
    assert out.status == "ok" and out.price_ratio == pytest.approx(1.0)
    assert out.legs[5]["stock"] == pytest.approx(0.0)


@pytest.mark.parametrize("form4_price, status", [(100, "ok"), (51, "ok"), (199, "ok"),
                                                 (49, "ticker_mismatch"), (201, "ticker_mismatch")])
def test_a_reused_ticker_is_caught_by_the_form4_price(form4_price, status):
    start = date(2024, 1, 2)
    e = event(value=form4_price * 100.0, shares=100.0)
    out = returns.evaluate(e, flat(start, 40), flat(start, 40), None, "SPY", None, horizons=(5,))
    assert out.status == status


def test_history_that_ends_early_exits_at_the_last_close():
    start = date(2024, 1, 2)
    stock = series(start, [100.0] * 12 + [60.0])   # delisted on the 13th session
    spy = flat(start, 80)
    out = returns.evaluate(event(), stock, spy, None, "SPY", None, horizons=(5, 20))
    assert out.legs[5]["flags"] == []
    leg = out.legs[20]
    assert "ended_early" in leg["flags"] and leg["exit"] == stock.dates[-1]
    assert leg["stock"] == pytest.approx(-0.4)


def test_a_gap_inside_the_series_gives_no_return_for_that_horizon():
    start = date(2024, 1, 2)
    full = flat(start, 80)
    stock = bp.Bars([b for i, b in enumerate(full.bars) if not 10 <= i <= 30])
    out = returns.evaluate(event(), stock, full, None, "SPY", None, horizons=(5, 20, 60))
    assert 20 not in out.legs and 60 in out.legs
    # A close a few days stale still stands in for the exit day.
    assert out.legs[5]["exit"] == date(2024, 1, 16) != NYSE.add(out.entry, 4)


def test_entry_waits_for_the_first_bar_but_not_forever():
    start = date(2024, 1, 2)
    spy = flat(start, 80)
    thin = bp.Bars([b for b in flat(start, 80).bars if b.date not in (date(2024, 1, 11), date(2024, 1, 12))])
    out = returns.evaluate(event(), thin, spy, None, "SPY", None, horizons=(5,))
    assert out.entry == date(2024, 1, 16) and "late_entry" in out.flags
    # SPY is measured from the same (late) day.
    assert out.legs[5]["exit"] == NYSE.add(date(2024, 1, 16), 4)
    halted = bp.Bars([b for b in flat(start, 80).bars
                      if not date(2024, 1, 11) <= b.date <= date(2024, 1, 25)])
    assert returns.evaluate(event(), halted, spy, None, "SPY", None).status == "no_entry_bar"


def test_events_after_the_last_price_are_not_matured():
    start = date(2024, 1, 2)
    spy = flat(start, 10)  # ends 16 Jan
    e = event(filing_date=date(2024, 1, 12), first_trade_date=date(2024, 1, 10),
              last_trade_date=date(2024, 1, 10))
    out = returns.evaluate(e, flat(start, 10), spy, None, "SPY", None, horizons=(5,))
    assert out.status == "not_matured"
    e2 = event(filing_date=date(2024, 1, 20))
    assert returns.evaluate(e2, flat(start, 10), spy, None, "SPY", None).status == "not_matured"


def test_sector_benchmark_used_when_it_trades_on_entry_day():
    start = date(2024, 1, 2)
    stock = flat(start, 40)
    sector = series(start, [50 + i for i in range(40)])
    out = returns.evaluate(event(), stock, flat(start, 40), sector, "XLF", None, horizons=(5,))
    assert out.benchmark == "XLF" and out.legs[5]["sector"] > 0
    late_sector = bp.Bars([b for b in sector.bars if b.date > date(2024, 2, 1)])
    out = returns.evaluate(event(), stock, flat(start, 40), late_sector, "XLC", None, horizons=(5,))
    assert out.benchmark == "SPY"


def test_nothing_at_or_after_entry_changes_the_point_in_time_inputs():
    """Drift, the ticker check and the liquidity estimate only see bars before entry."""
    start = date(2023, 11, 1)
    rng = random.Random(7)
    closes = [100 * math.exp(sum(rng.gauss(0, 0.02) for _ in range(i))) for i in range(120)]
    base = series(start, closes)
    e = event(filing_date=date(2024, 1, 10), first_trade_date=date(2024, 1, 5),
              last_trade_date=date(2024, 1, 8), value=100 * closes[40], shares=100.0)
    a = returns.evaluate(e, base, flat(start, 120), None, "SPY", None)
    entry = a.entry
    scrambled = bp.Bars([b if b.date < entry else
                         bp.Bar(b.date, b.open * 3, b.close * 3, b.adj_open * 3, b.adj_close * 3, 1.0)
                         for b in base.bars])
    b = returns.evaluate(e, scrambled, flat(start, 120), None, "SPY", None)
    assert (a.drift_pct, a.price_ratio, a.dollar_volume, a.cost) == (
        b.drift_pct, b.price_ratio, b.dollar_volume, b.cost)
    assert a.legs[20]["stock"] != b.legs[20]["stock"]


def test_pre_open_acceptance_uses_the_previous_close_for_drift():
    """A filing accepted at 8 am enters at that day's open: that day's close is future."""
    start = date(2024, 1, 2)
    closes = [100.0] * 7 + [150.0] * 30   # jumps on the 8th session (11 Jan)
    stock = series(start, closes)
    e = event(filing_date=date(2024, 1, 11), first_trade_date=date(2024, 1, 9),
              last_trade_date=date(2024, 1, 9), accepted_at=datetime(2024, 1, 11, 8, tzinfo=ET))
    out = returns.evaluate(e, stock, flat(start, 37), None, "SPY", None, horizons=(5,))
    assert out.entry == date(2024, 1, 11)
    assert out.drift_pct == pytest.approx(0.0)


@pytest.mark.parametrize("dv, half", [(None, 150), (0, 150), (4e4, 150), (5e4, 75), (6e5, 30),
                                      (5e6, 10), (4.9e7, 10), (5e7, 2), (1e9, 2)])
def test_spread_tiers(dv, half):
    assert returns.half_spread_bps(dv) == half
    assert returns.round_trip_cost(dv) == pytest.approx(2 * (10 + half) / 1e4)


def test_net_return_subtracts_the_round_trip_cost():
    start = date(2024, 1, 2)
    stock = series(start, [10 + 0.1 * i for i in range(40)], volume=1_000)  # ~$10k a day
    out = returns.evaluate(event(value=1_060.0, shares=100.0), stock, flat(start, 40), None, "SPY",
                           None, horizons=(5,))
    assert out.cost == pytest.approx(2 * (10 + 150) / 1e4)
    assert out.signal(5, net=True) == pytest.approx(out.signal(5) - out.cost)


# ---------------------------------------------------------------- price sources

def test_cache_round_trip_and_adj_open_derivation(tmp_path):
    bars = [bp.Bar(date(2024, 1, 2), 10.0, 11.0, 5.0, 5.5, 1000.0),
            bp.Bar(date(2024, 1, 3), 11.0, 12.0, 5.5, 6.0, None)]
    (tmp_path / "bars").mkdir()
    bp.write_csv(tmp_path / "bars" / "BRK.B.csv.gz", bars)
    src = bp.CachePrices(tmp_path)
    got = src.get("BRK.B")
    assert [b.date for b in got.bars] == [date(2024, 1, 2), date(2024, 1, 3)]
    assert got.bars[0] == bars[0] and got.bars[1].volume is None
    assert src.get("NOPE") is None
    # Tiingo layout with no adjOpen: derived from open x adjClose / close.
    text = "date,open,high,low,close,volume,adjOpen,adjClose\n2024-01-02,10,12,9,11,5,,5.5\n"
    (tmp_path / "bars" / "T.csv.gz").write_bytes(gzip.compress(text.encode()))
    assert bp.CachePrices(tmp_path).get("T").bars[0].adj_open == pytest.approx(5.0)


def test_real_tiingo_cache_file_parses():
    text = (
        "date,open,high,low,close,volume,adjOpen,adjClose\n"
        "2014-01-02,56.65,57.3,55.35,55.87,519300,51.8419296861968,51.128130830852875\n"
    )
    (b,) = bp.parse_csv(text)
    assert (b.open, b.close, b.volume) == (56.65, 55.87, 519300)
    assert b.adj_close == pytest.approx(51.128130830852875)


def test_bars_lookups():
    s = flat(date(2024, 1, 2), 5)  # 2, 3, 4, 5, 8 Jan
    assert s.index_on_or_after(date(2024, 1, 6)) == 4
    assert s.index_on_or_before(date(2024, 1, 7)) == 3
    assert s.index_before(date(2024, 1, 2)) is None
    assert s.at(date(2024, 1, 6)) is None and s.at(date(2024, 1, 8)).date == date(2024, 1, 8)
    assert s.index_on_or_after(date(2024, 2, 1)) is None


# ---------------------------------------------------------------- yahoo

def _ts(d: date) -> int:
    return int(datetime(d.year, d.month, d.day, 9, 30, tzinfo=ET).timestamp())


def test_yahoo_raw_prices_are_rebuilt_from_split_events():
    days = [date(2020, 8, 27), date(2020, 8, 28), date(2020, 8, 31)]
    payload = {"chart": {"error": None, "result": [{
        "meta": {"exchangeTimezoneName": "America/New_York"},
        "timestamp": [_ts(d) for d in days],
        "events": {"splits": {"1": {"date": _ts(days[2]), "numerator": 4.0, "denominator": 1.0}}},
        "indicators": {
            # Split-adjusted like Yahoo: 500 raw before the split shows as 125.
            "quote": [{"open": [124.0, 126.0, 127.5], "close": [125.0, 124.8, 129.0],
                       "volume": [400.0, 800.0, 1000.0]}],
            "adjclose": [{"adjclose": [120.0, 119.8, 125.0]}],
        }}]}}
    bars = parse_chart(payload)
    assert [b.date for b in bars] == days
    assert bars[0].close == pytest.approx(500.0) and bars[0].open == pytest.approx(496.0)
    assert bars[0].volume == pytest.approx(100.0)
    assert bars[2].close == pytest.approx(129.0) and bars[2].volume == 1000.0
    assert bars[0].adj_open == pytest.approx(124.0 * 120.0 / 125.0)


def test_yahoo_skips_null_rows_and_reports_empty():
    payload = {"chart": {"result": [{"meta": {}, "timestamp": [_ts(date(2024, 1, 2))],
                                     "indicators": {"quote": [{"open": [None], "close": [None],
                                                               "volume": [None]}],
                                                    "adjclose": [{"adjclose": [None]}]}}]}}
    assert parse_chart(payload) == []
    assert parse_chart({"chart": {"result": None, "error": {"code": "Not Found"}}}) == []


def test_yahoo_symbols():
    assert yahoo_symbol("BRK.B") == "BRK-B" and yahoo_symbol("AAPL") == "AAPL"


# ---------------------------------------------------------------- sectors

@pytest.mark.parametrize("sic, etf", [
    ("6022", "XLF"), ("6798", "XLRE"), ("2834", "XLV"), ("1311", "XLE"), ("3674", "XLK"),
    ("7372", "XLK"), ("5812", "XLY"), ("4911", "XLU"), ("4813", "XLC"), ("2060", "XLP"),
    ("3711", "XLY"), ("3721", "XLI"), ("3841", "XLV"), ("6770", "SPY"), ("", "SPY"),
    (None, "SPY"), ("9999", "SPY"), ("5411", "XLP"), ("1040", "XLB"),
])
def test_sic_to_sector_etf(sic, etf):
    assert sector_etf(sic) == etf


# ---------------------------------------------------------------- statistics

def test_summary_matches_hand_computation():
    v = [0.01, -0.02, 0.03, 0.04]
    s = stats.summarize(v, keys=["a", "a", "b", "b"], winsor=(-0.01, 0.035))
    assert s.n == 4 and s.mean == pytest.approx(0.015)
    sd = math.sqrt(sum((x - 0.015) ** 2 for x in v) / 3)
    assert s.t == pytest.approx(0.015 / (sd / 2))
    assert s.hit == 0.75 and s.median == pytest.approx(0.02)
    assert s.wmean == pytest.approx((0.01 - 0.01 + 0.03 + 0.035) / 4)
    # Clustered: sums of deviations per cluster are -0.04 and +0.04.
    var = (0.04 ** 2 + 0.04 ** 2) / 16 * 2
    assert s.clusters == 2 and s.t_cluster == pytest.approx(0.015 / math.sqrt(var))


def test_clustering_shrinks_t_when_returns_move_together():
    rng = random.Random(1)
    vals, keys = [], []
    for m in range(24):
        shock = rng.gauss(0.01, 0.05)       # one common shock per month
        for _ in range(50):
            vals.append(shock + rng.gauss(0, 0.01))
            keys.append(m)
    s = stats.summarize(vals, keys)
    assert abs(s.t_cluster) < abs(s.t) / 3


def test_cluster_keys():
    assert stats.cluster_key(date(2024, 5, 3), 20) == (2024, 5)
    assert stats.cluster_key(date(2024, 5, 3), 60) == (2024, 1)


def test_quantiles_and_buckets():
    cut = stats.quantile_cutoffs(list(range(100)), 5)
    assert cut == [20, 40, 60, 80]
    assert [stats.bucket_of(x, cut) for x in (0, 19, 20, 79, 80, 99)] == [0, 0, 1, 3, 4, 4]


def test_rank_correlation_with_ties():
    assert stats.ranks([3, 1, 3, 2]) == [3.5, 1, 3.5, 2]
    assert stats.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert stats.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)


def test_lag_buckets_follow_the_form4_deadline():
    assert [study.lag_bucket(d) for d in (0, 1, 2, 3, 4, 5, 40)] == [0, 0, 1, 2, 2, 3, 3]


# ---------------------------------------------------------------- walk-forward

def _result(entry: date, lag: int, drift: float, r20: float, side="buy") -> study.Result:
    e = event(side=side, filing_date=entry - timedelta(days=1), lag_days=lag, lag_ratio=lag / 2)
    o = returns.Outcome("ok", entry=entry, entry_open=10.0, drift_pct=drift, cost=0.004,
                        dollar_volume=1e6)
    exit_day = NYSE.add(entry, 19)
    o.legs[20] = {"stock": r20, "spy": 0.0, "sector": 0.0, "iwm": 0.0, "exit": exit_day, "flags": []}
    return study.Result(e, o, "SPY")


def test_walk_forward_finds_a_planted_effect_and_purges_overlap():
    rng = random.Random(3)
    rs = []
    for y in range(2015, 2021):
        for i in range(400):
            entry = NYSE.add(date(y, 1, 2), rng.randrange(245))
            lag = rng.choice([1, 2, 2, 2, 5])
            drift = rng.uniform(-20, 20)
            # Returns fall with drift and rise with lateness, plus noise.
            r = -0.001 * drift + (0.02 if lag > 2 else 0.0) + rng.gauss(0, 0.02)
            rs.append(_result(entry, lag, drift, r))
    # A sub-penny stock up 1,000x in a prompt, low-drift event: winsorized away.
    rs.append(_result(date(2019, 6, 3), 1, -15.0, 1000.0))
    folds = study.walk_forward(rs)
    assert [f.year for f in folds] == [2017, 2018, 2019, 2020]
    for f in folds:
        assert f.lag_oos > 0 and f.lag_is > 0
        assert f.drift_oos < 0 and f.drift_is < 0
        assert f.ic > 0.3 and f.top_minus_bottom > 0
        # Purged: training holds only events that exited before the test year.
        train = [x for x in rs if x.outcome.legs[20]["exit"] < date(f.year, 1, 1)]
        assert f.n_train == len(train)
        assert f.n_train < sum(1 for x in rs if x.outcome.entry.year < f.year)
    verdict = "\n".join(study.walk_forward_verdict(folds))
    assert "held in 4 of 4" in verdict


def test_walk_forward_on_noise_has_no_consistent_edge():
    rng = random.Random(5)
    rs = [_result(NYSE.add(date(y, 1, 2), rng.randrange(245)), rng.choice([1, 2, 5]),
                  rng.uniform(-20, 20), rng.gauss(0, 0.05))
          for y in range(2012, 2024) for _ in range(300)]
    folds = study.walk_forward(rs)
    assert len(folds) == 10
    assert abs(sum(f.ic for f in folds) / len(folds)) < 0.05


def test_report_renders_on_synthetic_results():
    from tradetracker.backtest.events import Exclusions

    rng = random.Random(9)
    rs = [_result(NYSE.add(date(y, 1, 2), rng.randrange(245)), rng.choice([0, 2, 3, 6]),
                  rng.uniform(-10, 30), rng.gauss(0.01, 0.05), side=rng.choice(["buy", "sell"]))
          for y in range(2015, 2020) for _ in range(200)]
    text = study.render(rs, Exclusions(), "synthetic")
    assert "## Buys, liquid (primary)" in text
    assert "very late (5+ days, >2x)" in text and "Walk-forward" in text


def test_source_comparison_measures_agreement_and_survivorship():
    rng = random.Random(11)
    a, b = [], []
    for i in range(200):
        entry = NYSE.add(date(2020, 1, 2), i)
        r = rng.gauss(0.01, 0.05)
        x = _result(entry, 2, 1.0, r)
        x.event.filer_id = i
        y = _result(entry, 2, 1.0, r + 0.001)
        y.event.filer_id = i
        if i % 4 == 0:   # the primary source lacks a quarter of the names (delisted)
            x.outcome.status = "no_prices"
            y.outcome.legs[20]["stock"] = -0.2
        a.append(x)
        b.append(y)
    text = "\n".join(study.source_comparison(a, b, "yahoo", "tiingo"))
    assert "Events priced by both: 150" in text
    assert "100.0% within 0.5 pp" in text
    assert "| missing from yahoo | 50 |" in text
    assert "no_prices/ok 50" in text


def test_clustered_difference_matches_hand_computation():
    a, ka = [1.0, 3.0], ["m1", "m2"]
    b, kb = [0.0, 0.0, 3.0], ["m1", "m1", "m2"]
    d, t = stats.diff_summary(a, b, ka, kb)
    assert d == pytest.approx(2.0 - 1.0)
    # Per-cluster: m1 = (1-2)/2 - ((0-1)+(0-1))/3 = 1/6; m2 = (3-2)/2 - (3-1)/3 = -1/6.
    se = math.sqrt(((1 / 6) ** 2 + (1 / 6) ** 2) * 2)
    assert t == pytest.approx(1.0 / se)
    # Welch without keys.
    d2, t2 = stats.diff_summary(a, b)
    assert d2 == d and t2 == pytest.approx(1.0 / math.sqrt(2 / 2 + 3 / 3))
