"""Rank recent buys with a fitted model: the spec's "top 20 buys" list.

Recent filings have no forward returns yet, so their point-in-time inputs (drift,
dollar volume, 52-week drawdown) are measured up to the last close on or before
`as_of` instead of up to an entry. History inputs (repeat buys, track record) come from
the full event history, as in the backtest.

Hide rules from the spec: drift above 20%, filed more than 30 trading days ago, and
(standing in for "under $50M market cap", which needs share counts we don't load yet)
a last close under $2 or median dollar volume under $100k.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from datetime import date, timedelta

from ..backtest.returns import LIQUIDITY_WINDOW, TICKER_CHECK, round_trip_cost
from ..backtest.study import MIN_LIQUID_DOLLAR_VOLUME, MIN_LIQUID_PRICE, Result
from ..features.calendar import NYSE
from ..fundamentals import market_cap
from . import model as M
from .signals import DRAWDOWN_MIN_SESSIONS, DRAWDOWN_SESSIONS, Signals

MAX_DAYS_SINCE_FILING = 30
MAX_DRIFT = 0.20


@dataclass
class Ranked:
    result: Result
    signals: Signals
    days_since_filing: int
    last_close: float
    score: float
    quality: float
    reason: str


def _live_inputs(x: Result, prices, as_of: date) -> tuple | str:
    """(drift, dollar volume, drawdown, last raw close) as of the close of `as_of`,
    or the reason the event can't be priced."""
    e = x.event
    bars = prices.get(e.ticker)
    if bars is None:
        return "no_prices"
    i = bars.index_on_or_before(as_of)
    t0 = bars.index_on_or_before(e.first_trade_date)
    t1 = bars.index_on_or_before(e.last_trade_date)
    if i is None or t0 is None or i < t0:
        return "no_prices"
    # The backtest's ticker check: the Form 4 price must be within 0.5x to 2x of the
    # vendor's raw close on the trade date, or the symbol may now be another company.
    raw = bars.bars[t1].close
    if e.avg_price and raw and not TICKER_CHECK[0] <= e.avg_price / raw <= TICKER_CHECK[1]:
        return "ticker_mismatch"
    last = bars.bars[i]
    drift = last.adj_close / bars.bars[t0].adj_close - 1
    window = [b for b in bars.bars[max(0, i - LIQUIDITY_WINDOW + 1): i + 1]
              if b.close and b.volume is not None]
    dv = statistics.median(b.close * b.volume for b in window) if window else None
    dd = None
    if i + 1 >= DRAWDOWN_MIN_SESSIONS:
        dd = last.adj_close / max(b.adj_close for b in
                                  bars.bars[max(0, i - DRAWDOWN_SESSIONS + 1): i + 1]) - 1
    return drift, dv, dd, last.close


def rank(results: list[Result], signals: list[Signals], prices, m: M.Model, as_of: date,
         top: int = 20, shares=None, min_cap: float | None = None) -> tuple[list[Ranked], dict]:
    """With `min_cap`, hide stocks whose market cap as of today (share counts from
    `shares`) is below it or unknown: the spec's floor."""
    hidden = {"drift": 0, "illiquid": 0, "no_prices": 0, "ticker_mismatch": 0, "10b5-1": 0,
              "small_cap": 0}
    out = []
    for x, s in zip(results, signals):
        e = x.event
        if e.side != "buy" or e.filing_date > as_of:
            continue
        days = NYSE.business_days_between(e.filing_date, as_of)
        if days > MAX_DAYS_SINCE_FILING:
            continue
        if e.is_10b5_1:
            hidden["10b5-1"] += 1
            continue
        live = _live_inputs(x, prices, as_of)
        if isinstance(live, str):
            hidden[live] += 1
            continue
        drift, dv, dd, close = live
        if drift > MAX_DRIFT:
            hidden["drift"] += 1
            continue
        if not close or close < MIN_LIQUID_PRICE or dv is None or dv < MIN_LIQUID_DOLLAR_VOLUME:
            hidden["illiquid"] += 1
            continue
        cap = s.market_cap
        if min_cap is not None:
            cap = market_cap(shares, e.issuer_cik, prices.get(e.ticker), as_of + timedelta(days=1))
            if cap is None or cap < min_cap:
                hidden["small_cap"] += 1
                continue
        s = Signals(**{**s.as_dict(), "drift": drift, "dollar_volume": dv, "drawdown": dd,
                       "market_cap": cap})
        sc = m.score(s, days)
        out.append(Ranked(x, s, days, close, sc, m.quality(s), M.reason(s, e.ticker, days)))
    # One line per stock: its best-scored insider.
    best: dict[str, Ranked] = {}
    for r in out:
        t = r.result.event.ticker
        if t not in best or r.score > best[t].score:
            best[t] = r
    ranked = sorted(best.values(), key=lambda r: (-r.score, r.result.event.ticker))
    return ranked[:top], hidden


def cost_note(r: Ranked) -> str:
    return f"{100 * round_trip_cost(r.signals.dollar_volume):.2f}%"
