"""Forward returns for one event: entry at an open, exit at a close, vs benchmarks.

For a horizon of h trading days, the position is bought at the open of the entry day
and sold at the close of the (h-1)th trading day after it, so it is held for exactly h
sessions. Benchmarks (SPY, the stock's sector ETF, IWM) are held over the same days.

Returns are signed so a positive number means the insider was right: the stock beat
the benchmark after a buy, or lagged it after a sell. Net returns subtract a round
trip of 10 bps per side plus half the estimated bid-ask spread per side.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import date

from ..features.calendar import NYSE
from .events import Event
from .prices import Bars

HORIZONS = (5, 20, 60)
COMMISSION_BPS = 10.0
MAX_ENTRY_DELAY = 5          # trading days a halted or thin stock may open late
MAX_EXIT_GAP_DAYS = 10       # calendar days the last close before exit may be stale
PRICE_GAP_DAYS = 5           # a trade-date close may come from up to 5 days before
TICKER_CHECK = (0.5, 2.0)    # Form 4 price / vendor raw close must fall inside
LIQUIDITY_WINDOW = 20        # trading days of dollar volume before entry

# Half-spread (bps) by median daily dollar volume over the 20 days before entry.
# A plain tiered estimate, conservative for small caps.
SPREAD_TIERS = [
    (50e6, 2.0),
    (5e6, 10.0),
    (5e5, 30.0),
    (5e4, 75.0),
    (0.0, 150.0),
]


def half_spread_bps(dollar_volume: float | None) -> float:
    if dollar_volume is None:
        return SPREAD_TIERS[-1][1]
    for floor, bps in SPREAD_TIERS:
        if dollar_volume >= floor:
            return bps
    return SPREAD_TIERS[-1][1]


def round_trip_cost(dollar_volume: float | None) -> float:
    """As a fraction: two sides of commission plus half-spread."""
    return 2 * (COMMISSION_BPS + half_spread_bps(dollar_volume)) / 10_000


@dataclass
class Outcome:
    status: str                         # 'ok' or why the event has no returns
    entry: date | None = None           # actual entry day
    entry_open: float | None = None     # raw
    drift_pct: float | None = None      # trade-date close to the last close before entry
    price_ratio: float | None = None    # Form 4 price / vendor raw close
    dollar_volume: float | None = None
    cost: float | None = None
    benchmark: str | None = None        # sector ETF actually used
    flags: list[str] = field(default_factory=list)
    # horizon -> {'stock', 'spy', 'sector', 'iwm', 'exit'}
    legs: dict[int, dict] = field(default_factory=dict)

    def signal(self, h: int, bench: str = "spy", sign: int = 1, net: bool = False) -> float | None:
        leg = self.legs.get(h)
        if not leg or leg.get(bench) is None:
            return None
        r = sign * (leg["stock"] - leg[bench])
        return r - self.cost if net and self.cost is not None else r


def _close_as_of(bars: Bars, d: date) -> tuple[date, float, float | None] | None:
    i = bars.index_on_or_before(d)
    if i is None or (d - bars.dates[i]).days > PRICE_GAP_DAYS:
        return None
    b = bars.bars[i]
    return b.date, b.adj_close, b.close


def _leg(bars: Bars, start: date, end: date) -> float | None:
    """Open of `start` to close of `end` for a benchmark that trades every session."""
    i = bars.index_on_or_after(start)
    j = bars.index_on_or_before(end)
    if i is None or j is None or j < i or bars.dates[i] != start:
        return None
    if (end - bars.dates[j]).days > MAX_EXIT_GAP_DAYS:
        return None
    o = bars.bars[i].adj_open
    return None if not o else bars.bars[j].adj_close / o - 1


def evaluate(
    e: Event,
    stock: Bars | None,
    spy: Bars,
    sector: Bars | None,
    sector_name: str,
    iwm: Bars | None,
    horizons=HORIZONS,
) -> Outcome:
    if stock is None:
        return Outcome("no_prices")
    planned = e.entry_date
    last_market_day = spy.dates[-1]
    if planned > last_market_day:
        return Outcome("not_matured")

    # Entry: the first bar on or after the planned day, if it opens soon enough.
    i = stock.index_on_or_after(planned)
    if i is None:
        return Outcome("ended_before_entry")
    if stock.dates[i] > NYSE.add(planned, MAX_ENTRY_DELAY):
        return Outcome("no_entry_bar")
    entry_bar = stock.bars[i]
    if not entry_bar.adj_open or not entry_bar.open:
        return Outcome("no_entry_bar")
    out = Outcome("ok", entry=entry_bar.date, entry_open=entry_bar.open)
    if entry_bar.date != planned:
        out.flags.append("late_entry")

    # Point-in-time checks, all from bars before entry.
    t0 = _close_as_of(stock, e.first_trade_date)
    pre = stock.index_before(entry_bar.date)
    if t0 is not None and pre is not None and stock.dates[pre] >= t0[0]:
        move = stock.bars[pre].adj_close / t0[1] - 1
        out.drift_pct = 100 * (-move if e.side == "sell" else move)
    else:
        out.flags.append("no_drift")
    t1 = _close_as_of(stock, e.last_trade_date)
    if e.avg_price and t1 is not None and t1[2]:
        out.price_ratio = e.avg_price / t1[2]
        lo, hi = TICKER_CHECK
        if not lo <= out.price_ratio <= hi:
            out.status = "ticker_mismatch"
            return out
    else:
        out.flags.append("no_price_check")
    window = [b for b in stock.bars[max(0, i - LIQUIDITY_WINDOW):i] if b.close and b.volume is not None]
    if window:
        out.dollar_volume = statistics.median(b.close * b.volume for b in window)
    out.cost = round_trip_cost(out.dollar_volume)

    use_sector = sector if sector is not None and sector.at(entry_bar.date) else spy
    out.benchmark = sector_name if use_sector is sector else "SPY"

    for h in horizons:
        target = NYSE.add(entry_bar.date, h - 1)
        if target > last_market_day:
            continue  # not matured yet
        j = stock.index_on_or_before(target)
        exit_bar = stock.bars[j]
        leg_flags = []
        if (target - exit_bar.date).days > MAX_EXIT_GAP_DAYS:
            if j == len(stock) - 1:
                leg_flags.append("ended_early")   # delisted or vendor history stops
            else:
                continue  # a long gap inside the series: no trustworthy exit price
        exit_day = exit_bar.date
        ret = exit_bar.adj_close / entry_bar.adj_open - 1
        leg = {
            "stock": ret,
            "exit": exit_day,
            "spy": _leg(spy, entry_bar.date, exit_day),
            "sector": _leg(use_sector, entry_bar.date, exit_day),
            "iwm": _leg(iwm, entry_bar.date, exit_day) if iwm is not None else None,
            "flags": leg_flags,
        }
        out.legs[h] = leg
    if not out.legs:
        out.status = "not_matured"
    return out
