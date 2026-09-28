"""Unit tests for the pure feature math. Dates are real 2024 calendar dates."""

import itertools
import random
from datetime import date, datetime, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from tradetracker.features import compute as fc
from tradetracker.features.compute import (
    PriceSeries,
    TradeInput,
    _LogHistogram,
    compute,
    drift_pct,
    filing_date,
    percentile,
    trade_value,
)

ET = ZoneInfo("America/New_York")
AS_OF = date(2024, 6, 28)
_ids = itertools.count(1)


def et(d: date, hour: int = 0, minute: int = 0) -> datetime:
    return datetime(d.year, d.month, d.day, hour, minute, tzinfo=ET)


def trade(trade_date, filed, *, side="buy", ticker="AAA", filer=1, filing=None, source="edgar",
          doc="4", amends=None, shares=100, price=10, role="director", plan=False, low=None,
          high=None, tid=None):
    tid = tid or next(_ids)
    return TradeInput(
        trade_id=tid, filing_id=filing or 1000 + tid, filer_id=filer, source=source,
        document_type=doc, amends_filing_id=amends,
        filed_at=filed if isinstance(filed, datetime) else et(filed),
        ticker=ticker, side=side, trade_date=trade_date,
        shares=None if shares is None else Decimal(shares),
        price=None if price is None else Decimal(str(price)),
        amount_low=low, amount_high=high, role=role, is_10b5_1=plan,
    )


def series(*pairs):
    return PriceSeries([(date.fromisoformat(d), c) for d, c in pairs])


def run(trades, prices=None, as_of=AS_OF):
    prices = prices or {}
    return {r.trade_id: r for r in compute(trades, lambda t: prices.get(t), as_of)}


# --- lag and lag ratio ----------------------------------------------------------------

@pytest.mark.parametrize("traded,filed,lag", [
    (date(2024, 3, 4), date(2024, 3, 4), 0),
    (date(2024, 3, 4), date(2024, 3, 6), 2),
    (date(2024, 3, 1), date(2024, 3, 5), 2),     # over a weekend
    (date(2024, 1, 12), date(2024, 1, 17), 2),   # over MLK Day
    (date(2024, 3, 28), date(2024, 4, 1), 2),    # Good Friday counts for the SEC
    (date(2024, 3, 4), date(2024, 3, 18), 10),
])
def test_lag_days_counts_sec_business_days(traded, filed, lag):
    t = trade(traded, filed)
    r = run([t])[t.trade_id]
    assert r.lag_days == lag
    assert r.lag_ratio == pytest.approx(lag / 2)


def test_lag_ratio_uses_each_sources_deadline():
    f4 = trade(date(2024, 3, 4), date(2024, 3, 7))
    house = trade(date(2024, 3, 4), date(2024, 3, 7), source="house", role="member")
    senate = trade(date(2024, 2, 1), date(2024, 3, 15), source="senate", role="member")
    other = trade(date(2024, 3, 4), date(2024, 3, 7), source="quiver")
    rows = run([f4, house, senate, other])
    assert rows[f4.trade_id].lag_ratio == pytest.approx(1.5)          # late
    assert rows[house.trade_id].lag_ratio == pytest.approx(3 / 31)
    assert rows[senate.trade_id].lag_ratio == pytest.approx(30 / 31)  # 31 weekdays less Presidents' Day
    assert rows[other.trade_id].lag_days == 3 and rows[other.trade_id].lag_ratio is None


def test_filing_date_is_the_eastern_date_of_filed_at():
    # 22:30 in New York is already the next day in UTC.
    late = datetime(2024, 3, 6, 22, 30, tzinfo=ET).astimezone(ZoneInfo("UTC"))
    assert filing_date(late) == date(2024, 3, 6)
    assert filing_date(datetime(2024, 3, 6, 23, 59)) == date(2024, 3, 6)  # naive stays put
    t = trade(date(2024, 3, 4), late)
    assert run([t])[t.trade_id].lag_days == 2


def test_trade_dated_after_its_filing_gets_no_lag_or_drift():
    t = trade(date(2024, 3, 8), date(2024, 3, 6))
    r = run([t], {"AAA": series(("2024-03-06", 10), ("2024-03-08", 11))})[t.trade_id]
    assert "trade_after_filing" in r.flags
    assert r.lag_days is None and r.lag_ratio is None and r.drift_pct is None


# --- drift ------------------------------------------------------------------------------

PRICES = {"AAA": series(("2024-03-01", 100), ("2024-03-04", 100), ("2024-03-05", 104),
                        ("2024-03-06", 110), ("2024-03-07", 90))}


def test_drift_is_trade_close_to_filing_close_for_buys():
    t = trade(date(2024, 3, 4), date(2024, 3, 6))
    assert run([t], PRICES)[t.trade_id].drift_pct == pytest.approx(10.0)


def test_drift_sign_flips_for_sells():
    up = trade(date(2024, 3, 4), date(2024, 3, 6), side="sell")
    down = trade(date(2024, 3, 4), date(2024, 3, 7), side="sell")
    rows = run([up, down], PRICES)
    assert rows[up.trade_id].drift_pct == pytest.approx(-10.0)   # rose after a sale
    assert rows[down.trade_id].drift_pct == pytest.approx(10.0)  # fell: move already captured


def test_drift_helper_symmetry():
    assert drift_pct("buy", 50, 55) == pytest.approx(10)
    assert drift_pct("sell", 50, 55) == pytest.approx(-10)
    assert drift_pct("buy", 50, 50) == 0


def test_same_day_filing_has_zero_drift():
    t = trade(date(2024, 3, 5), date(2024, 3, 5))
    assert run([t], PRICES)[t.trade_id].drift_pct == 0


def test_weekend_and_holiday_dates_use_the_last_close():
    prices = {"AAA": series(("2024-03-28", 100), ("2024-04-01", 120))}
    # Filed on Good Friday (market closed): the Thursday close stands in.
    t = trade(date(2024, 3, 28), date(2024, 3, 29))
    # Traded on a Saturday (odd, but filings say so) and filed Monday.
    s = trade(date(2024, 3, 30), date(2024, 4, 1))
    rows = run([t, s], prices)
    assert rows[t.trade_id].drift_pct == 0
    assert rows[s.trade_id].drift_pct == pytest.approx(20)


def test_missing_prices_are_flagged_not_guessed():
    prices = {"AAA": series(("2024-03-04", 100)), "GAP": series(("2024-02-20", 50), ("2024-03-06", 60))}
    no_filing_close = trade(date(2024, 3, 4), date(2024, 3, 12))    # last close 8 days earlier
    no_trade_close = trade(date(2024, 3, 1), date(2024, 3, 6), ticker="GAP")  # halted 20 Feb
    unpriced = trade(date(2024, 3, 4), date(2024, 3, 6), ticker="ZZZ")
    unmapped = trade(date(2024, 3, 4), date(2024, 3, 6), ticker=None)
    rows = run([no_filing_close, no_trade_close, unpriced, unmapped], prices)
    assert rows[no_filing_close.trade_id].flags == ["no_filing_price"]
    assert rows[no_trade_close.trade_id].flags == ["no_trade_price"]
    assert rows[unpriced.trade_id].flags == ["no_prices"]
    assert rows[unmapped.trade_id].flags == ["no_ticker"]
    assert all(r.drift_pct is None for r in rows.values())


def test_price_series_ignores_missing_and_non_positive_closes():
    s = PriceSeries([(date(2024, 3, 4), None), (date(2024, 3, 5), 0), (date(2024, 3, 1), Decimal("7.5"))])
    assert s.close_as_of(date(2024, 3, 5)) == (date(2024, 3, 1), 7.5)
    assert s.close_as_of(date(2024, 2, 29)) is None
    assert s.close_as_of(date(2024, 3, 6)) == (date(2024, 3, 1), 7.5)   # 5 days back: ok
    assert s.close_as_of(date(2024, 3, 7)) is None                       # 6 days: too stale


def test_split_adjusted_closes_keep_drift_honest():
    # A 2:1 split between trade and filing: adjusted closes show no real move.
    prices = {"AAA": series(("2024-03-04", 50), ("2024-03-06", 50))}
    t = trade(date(2024, 3, 4), date(2024, 3, 6), price=100)
    assert run([t], prices)[t.trade_id].drift_pct == 0


# --- staleness --------------------------------------------------------------------------

def test_days_since_filing_counts_trading_days_to_as_of():
    t = trade(date(2024, 3, 26), date(2024, 3, 27))
    assert run([t], as_of=date(2024, 4, 2))[t.trade_id].days_since_filing == 3  # skips Good Friday
    assert run([t], as_of=date(2024, 3, 27))[t.trade_id].days_since_filing == 0
    assert run([t], as_of=date(2024, 3, 1))[t.trade_id].days_since_filing == 0  # never negative


# --- role ---------------------------------------------------------------------------------

def test_role_weight_ladder():
    roles = ["CEO", "CFO", "officer", "director", "10% owner", None, "member"]
    ts = [trade(date(2024, 3, 4), date(2024, 3, 6), role=r) for r in roles]
    w = [run(ts)[t.trade_id].role_weight for t in ts]
    assert w[0] == w[1] > w[2] > w[3] > w[4] > 0
    assert w[5] is None and w[6] is None


# --- size -------------------------------------------------------------------------------

def test_trade_value():
    assert trade_value(trade(date(2024, 3, 4), date(2024, 3, 6), shares=1000, price=12.5)) == 12500
    ptr = trade(date(2024, 3, 4), date(2024, 3, 6), shares=None, price=None,
                low=Decimal(15001), high=Decimal(50000))
    assert trade_value(ptr) == 32500.5
    assert trade_value(trade(date(2024, 3, 4), date(2024, 3, 6), price=0)) is None
    assert trade_value(trade(date(2024, 3, 4), date(2024, 3, 6), price=None)) is None


def test_size_uses_filer_history_after_five_prior_trades():
    days = [date(2024, 3, d) for d in (4, 5, 6, 7, 8)]
    past = [trade(d, d, filer=7, shares=100 * (i + 1)) for i, d in enumerate(days)]  # $1k..$5k
    big = trade(date(2024, 3, 11), date(2024, 3, 11), filer=7, shares=450)           # $4.5k
    same_day = trade(date(2024, 3, 8), date(2024, 3, 8), filer=7, shares=1000)
    rows = run(past + [big, same_day])
    assert rows[big.trade_id].size_basis == "filer"
    # Six earlier trades ($1k..$5k and $10k); $4.5k beats four of them.
    assert rows[big.trade_id].size_score == pytest.approx(4 / 6)
    # A trade on the fifth day only sees four earlier trades: not enough.
    assert rows[same_day.trade_id].size_basis is None
    assert all(rows[t.trade_id].size_score is None for t in past)


def test_size_filer_history_is_per_side():
    buys = [trade(date(2024, 3, d), date(2024, 3, d), filer=8) for d in (4, 5, 6, 7, 8)]
    sell = trade(date(2024, 3, 11), date(2024, 3, 11), filer=8, side="sell")
    assert run(buys + [sell])[sell.trade_id].size_score is None


def test_size_percentile_ties_use_mid_rank():
    assert percentile([1, 2, 2, 3], 2) == pytest.approx(0.5)
    assert percentile([1, 2, 3], 0.5) == 0
    assert percentile([1, 2, 3], 9) == 1


def test_size_falls_back_to_trailing_year_market_rank():
    start = date(2023, 1, 3)
    crowd = [trade(start, start, filer=100 + i, shares=i + 1, price=100) for i in range(200)]
    newcomer = trade(date(2023, 6, 1), date(2023, 6, 1), filer=999, shares=51, price=100)
    much_later = trade(date(2024, 3, 4), date(2024, 3, 4), filer=998, shares=51, price=100)
    rows = run(crowd + [newcomer, much_later])
    assert rows[newcomer.trade_id].size_basis == "market"
    assert rows[newcomer.trade_id].size_score == pytest.approx(0.2525, abs=0.01)
    # The crowd is over a year old by then and has dropped out of the window.
    assert rows[much_later.trade_id].size_score is None


def test_log_histogram_rank_tracks_exact_percentile():
    rng = random.Random(4)
    values = [10 ** rng.uniform(3, 8) for _ in range(5000)]
    h = _LogHistogram()
    for v in values:
        h.add(v)
    ordered = sorted(values)
    for q in (1e3, 5e4, 1e6, 3e7, 1e8, 1e12, 0.5):
        assert h.rank(q) == pytest.approx(percentile(ordered, q), abs=0.01)
    for v in values[:2500]:
        h.remove(v)
    assert h.total == 2500
    assert h.rank(1e6) == pytest.approx(percentile(sorted(values[2500:]), 1e6), abs=0.01)


def test_zero_price_trade_is_flagged_no_value():
    t = trade(date(2024, 3, 4), date(2024, 3, 6), price=0)
    r = run([t])[t.trade_id]
    assert "no_value" in r.flags and r.trade_value is None and r.size_score is None


# --- filer lag z-score ------------------------------------------------------------------------

def _history(filer, lags, start=date(2024, 1, 2)):
    out, d = [], start
    for lag in lags:
        filed = fc.SEC.add(d, lag)
        out.append(trade(d, filed, filer=filer))
        d += timedelta(days=14)
    return out


def test_zscore_needs_three_prior_filings():
    hist = _history(1, [1, 2])
    now = trade(date(2024, 4, 1), date(2024, 4, 5), filer=1)
    assert run(hist + [now])[now.trade_id].filer_lag_zscore is None


def test_zscore_against_own_history():
    hist = _history(2, [1, 2, 3])      # mean 2, population std 0.816
    now = trade(date(2024, 4, 1), date(2024, 4, 5), filer=2)  # lag 4
    assert run(hist + [now])[now.trade_id].filer_lag_zscore == pytest.approx(2 / 0.8165, rel=1e-3)


def test_zscore_std_floor_for_perfectly_regular_filers():
    hist = _history(3, [2, 2, 2, 2])
    late = trade(date(2024, 4, 1), date(2024, 4, 5), filer=3)    # lag 4
    usual = trade(date(2024, 4, 15), date(2024, 4, 17), filer=3)  # lag 2
    rows = run(hist + [late, usual])
    assert rows[late.trade_id].filer_lag_zscore == pytest.approx(4.0)  # (4 - 2) / 0.5
    # History is now [2, 2, 2, 2, 4]: mean 2.4, std 0.8.
    assert rows[usual.trade_id].filer_lag_zscore == pytest.approx(-0.5)


def test_zscore_counts_each_filing_once_and_ignores_later_and_same_day_filings():
    # One filing with three lines dated 1, 2 and 3 days back counts once, at lag 3.
    multi = [trade(date(2024, 1, 2) + timedelta(days=i), date(2024, 1, 5), filer=4, filing=77)
             for i in range(3)]
    hist = _history(4, [0, 0], start=date(2024, 2, 1))
    same_day = trade(date(2024, 4, 3), date(2024, 4, 5), filer=4)
    now = trade(date(2024, 4, 1), date(2024, 4, 5), filer=4)
    future = trade(date(2024, 5, 1), date(2024, 5, 20), filer=4)
    rows = run(multi + hist + [same_day, now, future])
    # History is [3, 0, 0]: mean 1, std 1.414. The same-day and later filings are not in it.
    assert rows[now.trade_id].filer_lag_zscore == pytest.approx((4 - 1) / 1.41421, rel=1e-3)


def test_zscore_history_skips_amendments():
    hist = _history(5, [1, 1, 1])
    amendment = trade(date(2024, 3, 1), date(2024, 4, 1), filer=5, doc="4/A")  # lag 21
    now = trade(date(2024, 4, 10), date(2024, 4, 11), filer=5)
    assert run(hist + [amendment, now])[now.trade_id].filer_lag_zscore == pytest.approx(0)


# --- clusters --------------------------------------------------------------------------------

def test_cluster_counts_distinct_insiders_within_14_days():
    base = date(2024, 3, 15)
    ts = [
        trade(base, date(2024, 3, 18), filer=1),
        trade(base - timedelta(days=14), date(2024, 3, 4), filer=2),   # edge: included
        trade(base - timedelta(days=15), date(2024, 3, 4), filer=3),   # too early
        trade(base + timedelta(days=3), date(2024, 3, 18), filer=4),   # filed same day: known
        trade(base + timedelta(days=1), date(2024, 3, 18), filer=2),   # same insider again
        trade(base, date(2024, 3, 18), filer=5, side="sell"),          # other side
        trade(base, date(2024, 3, 18), filer=6, ticker="BBB"),         # other ticker
        trade(base + timedelta(days=4), date(2024, 3, 25), filer=7),   # filed after: unknown yet
    ]
    rows = run(ts)
    assert rows[ts[0].trade_id].cluster_count == 3   # filers 1, 2, 4
    assert rows[ts[7].trade_id].cluster_count == 4   # sees 1, 2, 4 and itself (3 too early)
    assert rows[ts[5].trade_id].cluster_count == 1
    assert rows[ts[6].trade_id].cluster_count == 1


def test_trade_without_ticker_has_no_cluster():
    t = trade(date(2024, 3, 4), date(2024, 3, 6), ticker=None)
    assert run([t])[t.trade_id].cluster_count is None


# --- amendments ------------------------------------------------------------------------------

def test_amendment_repeating_a_trade_keeps_the_original_filing_date():
    prices = {"AAA": series(("2024-03-04", 100), ("2024-03-06", 110), ("2024-03-20", 150))}
    orig = trade(date(2024, 3, 4), date(2024, 3, 6), filing=500, filer=9)
    repeat = trade(date(2024, 3, 4), date(2024, 3, 20), filing=501, filer=9, doc="4/A", amends=500)
    new_line = trade(date(2024, 3, 5), date(2024, 3, 20), filing=501, filer=9, doc="4/A",
                     amends=500, shares=999)
    other_buyer = trade(date(2024, 3, 4), date(2024, 3, 6), filer=10)
    rows = run([orig, repeat, new_line, other_buyer], prices)
    r = rows[repeat.trade_id]
    assert r.duplicate_of_trade_id == orig.trade_id and "amendment_duplicate" in r.flags
    assert r.filing_date == date(2024, 3, 6) and r.lag_days == 2
    assert r.drift_pct == pytest.approx(10)
    assert r.cluster_count == rows[orig.trade_id].cluster_count == 2
    n = rows[new_line.trade_id]
    assert n.duplicate_of_trade_id is None and n.flags == ["amendment"]
    assert n.filing_date == date(2024, 3, 20) and n.lag_days == 11
    assert n.drift_pct == pytest.approx(50)  # no 5 Mar close, so the 4 Mar close stands in
    # The repeat does not count twice anywhere.
    assert rows[other_buyer.trade_id].cluster_count == 2


def test_amendment_correcting_shares_or_price_is_linked_to_the_line_it_fixes():
    a = trade(date(2024, 3, 4), date(2024, 3, 6), filing=600, filer=11, shares=100)
    b = trade(date(2024, 3, 4), date(2024, 3, 6), filing=600, filer=11, shares=200)
    other_day = trade(date(2024, 3, 1), date(2024, 3, 6), filing=600, filer=11, shares=50)
    fix_b = trade(date(2024, 3, 4), date(2024, 3, 25), filing=601, filer=11, doc="4/A",
                  amends=600, shares=250)
    same_a = trade(date(2024, 3, 4), date(2024, 3, 25), filing=601, filer=11, doc="4/A",
                   amends=600, shares=100)
    sold = trade(date(2024, 3, 4), date(2024, 3, 25), filing=601, filer=11, doc="4/A",
                 amends=600, side="sell")
    rows = run([a, b, other_day, fix_b, same_a, sold])
    # The exact repeat claims line a first, so the correction pairs with line b.
    assert rows[same_a.trade_id].duplicate_of_trade_id == a.trade_id
    assert rows[same_a.trade_id].flags[0] == "amendment_duplicate"
    assert rows[fix_b.trade_id].duplicate_of_trade_id == b.trade_id
    assert rows[fix_b.trade_id].flags[0] == "amendment_correction"
    assert rows[fix_b.trade_id].filing_date == date(2024, 3, 6)
    assert rows[fix_b.trade_id].trade_value == 2500  # the corrected size
    # Nothing in the original matches a sell, so it is a new disclosure.
    assert rows[sold.trade_id].duplicate_of_trade_id is None
    assert rows[sold.trade_id].filing_date == date(2024, 3, 25)


def test_amendment_correcting_the_trade_date_is_linked_by_size_and_price():
    orig = trade(date(2024, 3, 5), date(2024, 3, 7), filing=700, filer=12, shares=300)
    fixed = trade(date(2024, 3, 4), date(2024, 4, 2), filing=701, filer=12, doc="4/A",
                  amends=700, shares=300)
    rows = run([orig, fixed])
    r = rows[fixed.trade_id]
    assert r.duplicate_of_trade_id == orig.trade_id and r.flags[0] == "amendment_correction"
    # Public since the original filing; the lag uses the corrected trade date.
    assert r.filing_date == date(2024, 3, 7) and r.lag_days == 3


def test_unlinked_amendment_is_flagged_but_kept():
    t = trade(date(2024, 3, 4), date(2024, 3, 20), doc="4/A")
    r = run([t])[t.trade_id]
    assert r.flags == ["amendment", "no_prices"] and r.lag_days == 12


# --- flags and bookkeeping ----------------------------------------------------------------

def test_10b5_1_trades_are_computed_and_flagged():
    t = trade(date(2024, 3, 4), date(2024, 3, 6), plan=True)
    r = run([t], PRICES)[t.trade_id]
    assert r.flags == ["10b5_1"] and r.drift_pct is not None


def test_output_order_matches_input_and_is_deterministic():
    ts = [trade(date(2024, 3, 4) + timedelta(days=i % 5), date(2024, 3, 11), filer=i % 3)
          for i in range(30)]
    shuffled = ts[:]
    random.Random(1).shuffle(shuffled)
    a = compute(ts, lambda _: None, AS_OF)
    b = {r.trade_id: r for r in compute(shuffled, lambda _: None, AS_OF)}
    assert [r.trade_id for r in a] == [t.trade_id for t in ts]
    assert all(r == b[r.trade_id] for r in a)


def test_price_lookup_called_once_per_ticker():
    calls = []
    ts = [trade(date(2024, 3, 4), date(2024, 3, 6), ticker=t) for t in "ABABAB"]
    compute(ts, lambda t: calls.append(t), AS_OF)
    assert sorted(calls) == ["A", "B"]
