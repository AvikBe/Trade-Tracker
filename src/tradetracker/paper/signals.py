"""Score new buys exactly as the backtest scored them at entry.

The backtest computed every input from bars before the entry open: drift (trade-date
close to the last close before entry), the 52-week drawdown, the median dollar volume
of the 20 sessions before entry, the ticker check and the market cap. Here the entry
open hasn't happened yet, so the same functions run on bars that end before it
(`backtest.returns.pre_entry`, `scoring.signals.drawdown`, `fundamentals.market_cap`).

One input can't be known before the open: the backtest's liquidity rule uses the entry
open (at least $2). The last close before entry stands in for it, as in `tt rank`.

Hide rules (the backtest's `scoring.study.shown` for the chosen setup): buys only, no
10b5-1 plan, liquid, drift at most 20%, market cap at least `min_cap` and known.
A buy is a candidate when its score is at or above the model's training top-decile
cutoff, which is what the backtest's top-20 portfolio bought.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from ..backtest.prices import Bars
from ..backtest.returns import Outcome, pre_entry
from ..backtest.study import MIN_LIQUID_DOLLAR_VOLUME, MIN_LIQUID_PRICE
from ..fundamentals import market_cap
from ..scoring import model as M
from ..scoring.signals import Signals, drawdown, stake_change
from ..scoring.study import MAX_DRIFT_SHOWN
from .ingest import LiveEvent

STALE_DAYS = 7        # the last close before entry may be at most this old (calendar days)


@dataclass
class Decision:
    live: LiveEvent
    entry: date
    status: str                      # "candidate", "below_cutoff" or why it is hidden
    signals: Signals | None = None
    score: float | None = None
    last_close: float | None = None
    last_close_date: date | None = None
    cost: float | None = None
    flags: list[str] = field(default_factory=list)

    @property
    def ticker(self) -> str:
        return self.live.event.ticker


class _One:
    """A one-ticker price source, for the scoring helpers that take `prices`."""

    def __init__(self, ticker: str, bars: Bars | None):
        self.ticker, self.bars = ticker, bars

    def get(self, ticker: str) -> Bars | None:
        return self.bars if ticker == self.ticker else None


def evaluate(le: LiveEvent, bars: Bars | None, entry: date, m: M.Model,
             shares=None, min_cap: float | None = 50e6) -> Decision:
    e = le.event
    d = Decision(le, entry, "candidate")
    if e.side != "buy":
        d.status = "not_a_buy"
        return d
    if e.is_10b5_1:
        d.status = "10b5-1"
        return d
    if bars is None or not len(bars):
        d.status = "no_prices"
        return d
    # Only bars before entry: a bar dated on or after entry would be from the future
    # in the backtest's terms (and a partial bar when this runs during the session).
    bars = Bars([b for b in bars.bars if b.date < entry])
    pre = bars.index_before(entry)
    if pre is None or (entry - bars.dates[pre]).days > STALE_DAYS:
        d.status = "no_prices"             # no history, or it stopped (halted, delisted)
        return d
    last = bars.bars[pre]
    d.last_close, d.last_close_date = last.close, last.date
    o = pre_entry(e, bars, entry, Outcome("ok"))
    d.flags += o.flags
    d.cost = o.cost
    if o.status != "ok":
        d.status = o.status                # ticker_mismatch
        return d
    if not last.close or last.close < MIN_LIQUID_PRICE or not o.dollar_volume \
            or o.dollar_volume < MIN_LIQUID_DOLLAR_VOLUME:
        d.status = "illiquid"
        return d
    drift = None if o.drift_pct is None else o.drift_pct / 100
    if drift is not None and drift > MAX_DRIFT_SHOWN:
        d.status = "drift"
        return d
    one = _One(e.ticker, bars)
    cap = market_cap(shares, e.issuer_cik, bars, entry) if shares is not None else None
    if min_cap is not None and (cap is None or cap < min_cap):
        d.status = "small_cap" if cap is not None else "no_market_cap"
        return d
    d.signals = Signals(
        role=e.role, value=e.value, stake_change=stake_change(e.shares, e.shares_held_before),
        cluster_count=1, repeat_buys=0, drawdown=drawdown(one, e.ticker, entry),
        track_record=None, track_n=0, dollar_volume=o.dollar_volume, drift=drift,
        market_cap=cap)
    # The backtest scored at entry, with no decay (days since filing = 0).
    d.score = m.score(d.signals)
    if m.threshold is not None and d.score < m.threshold:
        d.status = "below_cutoff"
    return d


def best_per_ticker(decisions: list[Decision]) -> list[Decision]:
    """One candidate per stock (its best-scored insider), highest score first, as the
    backtest's portfolio took one position per stock."""
    best: dict[str, Decision] = {}
    for d in decisions:
        if d.status != "candidate":
            continue
        if d.ticker not in best or d.score > best[d.ticker].score:
            best[d.ticker] = d
    return sorted(best.values(), key=lambda d: (-d.score, d.ticker))
