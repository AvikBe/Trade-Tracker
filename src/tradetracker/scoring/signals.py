"""Point-in-time inputs to the score, one `Signals` per backtest result.

The phase 2 features (role, cluster count, trade value) come with each event. This adds
what scoring needs beyond them, each from data public before the event's entry:

- `stake_change`: shares bought as a fraction of the holding before the buy, from the
  Form 4's "owned following" column (per account, before each account's first line).
  None when a line does not report it; `NEW_STAKE` when the insider held nothing before.
- `repeat_buys`: earlier buy events by the same insider in the same stock filed in the
  previous `REPEAT_WINDOW_DAYS`. Routine buyers carry little news (Cohen, Malloy and
  Pomorski 2012); a first buy in years carries more.
- `drawdown`: the last close before entry against the highest close of the previous 250
  sessions (0 at a 52-week high, -0.5 when the stock has halved).
- `track_record`: the insider's mean 20-day excess return on earlier buys whose exit
  was before this entry, when there are at least `TRACK_MIN_EVENTS` of them.
- `dollar_volume` and `drift` are the backtest's own (median over the 20 sessions
  before entry, and trade date to last close before entry).
"""

from __future__ import annotations

import bisect
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta

from ..backtest.study import PRIMARY_H, Result

REPEAT_WINDOW_DAYS = 730
DRAWDOWN_SESSIONS = 250
DRAWDOWN_MIN_SESSIONS = 60   # less history than this and the 52-week high means little
TRACK_MIN_EVENTS = 2
NEW_STAKE = 10.0             # stake_change for a first purchase (nothing held before)


@dataclass
class Signals:
    role: str | None
    value: float
    stake_change: float | None
    cluster_count: int
    repeat_buys: int
    drawdown: float | None
    track_record: float | None
    track_n: int
    dollar_volume: float | None
    drift: float | None          # fraction, in the trade's direction

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def stake_change(shares: float, held_before: float | None) -> float | None:
    if held_before is None or shares <= 0:
        return None
    if held_before <= 0:
        return NEW_STAKE
    return min(shares / held_before, NEW_STAKE)


def drawdown(prices, ticker: str, entry: date) -> float | None:
    bars = prices.get(ticker)
    if bars is None:
        return None
    i = bars.index_before(entry)
    if i is None or i + 1 < DRAWDOWN_MIN_SESSIONS:
        return None
    window = bars.bars[max(0, i - DRAWDOWN_SESSIONS + 1): i + 1]
    peak = max(b.adj_close for b in window)
    return bars.bars[i].adj_close / peak - 1


class _History:
    """Earlier events per key, for 'what was known before date d' lookups."""

    def __init__(self):
        self.dates: dict[tuple, list[date]] = defaultdict(list)
        self.values: dict[tuple, list[float]] = defaultdict(list)

    def add(self, key, d: date, v: float = 0.0) -> None:
        i = bisect.bisect_right(self.dates[key], d)
        self.dates[key].insert(i, d)
        self.values[key].insert(i, v)

    def count_between(self, key, lo: date, hi: date) -> int:
        """Entries with lo <= date < hi."""
        ds = self.dates.get(key, [])
        return bisect.bisect_left(ds, hi) - bisect.bisect_left(ds, lo)

    def before(self, key, d: date) -> list[float]:
        ds = self.dates.get(key, [])
        return self.values.get(key, [])[: bisect.bisect_left(ds, d)]


def build(results: list[Result], prices, h: int = PRIMARY_H) -> list[Signals]:
    """Signals for every result, aligned with `results`.

    `results` must hold every event the history features should see (all buys, not just
    the ones being scored): a repeat buy or a track record counts earlier events whatever
    their liquidity or 10b5-1 status, as long as they were buys.
    """
    buys = _History()                # (filer, ticker) -> filing dates of buy events
    track = _History()               # filer -> exit dates of matured buy events
    for x in results:
        e = x.event
        if e.side != "buy":
            continue
        buys.add((e.filer_id, e.ticker), e.filing_date)
        r = x.r(h) if x.ok else None
        if r is not None and not e.is_10b5_1:
            track.add(e.filer_id, x.outcome.legs[h]["exit"], r)

    out = []
    for x in results:
        e, o = x.event, x.outcome
        rep = buys.count_between((e.filer_id, e.ticker),
                                 e.filing_date - timedelta(days=REPEAT_WINDOW_DAYS),
                                 e.filing_date) if e.side == "buy" else 0
        tr_vals = track.before(e.filer_id, o.entry or e.entry_date)
        tr = sum(tr_vals) / len(tr_vals) if len(tr_vals) >= TRACK_MIN_EVENTS else None
        out.append(Signals(
            role=e.role,
            value=e.value,
            stake_change=stake_change(e.shares, e.shares_held_before),
            cluster_count=e.cluster_count,
            repeat_buys=rep,
            drawdown=drawdown(prices, e.ticker, o.entry) if o.entry else None,
            track_record=tr,
            track_n=len(tr_vals),
            dollar_volume=o.dollar_volume,
            drift=None if o.drift_pct is None else o.drift_pct / 100,
        ))
    return out
