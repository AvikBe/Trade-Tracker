"""Turn trades into backtest events, using only what was public at the time.

One event is one insider's disclosure of trades in one stock on one side on one filing
date: the lines of a Form 4 (and any other Form 4 the same insider filed that day for
the same stock) are one signal with one entry, not ten.

Lines left out, each counted in `Exclusions`:

- `duplicate`: an amendment or repeat filing that restates a trade already disclosed
  (`duplicate_of_trade_id`, from the feature job). The original stays in.
- `amendment`: other 4/A lines. Their filing date is the amendment's, so their lag
  and drift describe the correction, not the trade.
- `trade_after_filing`, `no_ticker`, `no_features`.

Lag comes from `latest_trade_features` (a pure function of the two dates). Drift is
recomputed from the backtest's own price source, ending at the last close before entry,
so it never includes a price the backtest could not have seen.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, time

import psycopg

from ..features.calendar import NYSE
from ..features.compute import EASTERN

MARKET_OPEN = time(9, 30)

EVENTS_SQL = """
    SELECT t.trade_id, f.filing_id, f.filer_id, f.issuer_cik, t.ticker, t.side, t.trade_date,
           t.shares, t.price, t.role, t.is_10b5_1, coalesce(f.document_type, ''),
           f.accepted_at, x.filing_date, x.lag_days, x.lag_ratio, x.role_weight,
           x.cluster_count, x.trade_value, x.duplicate_of_trade_id, x.flags
    FROM trades t
    JOIN filings f USING (filing_id)
    LEFT JOIN latest_trade_features x USING (trade_id)
    WHERE f.source = 'edgar'
"""


@dataclass
class Event:
    filer_id: int
    issuer_cik: str | None
    ticker: str
    side: str
    filing_date: date
    accepted_at: datetime | None
    first_trade_date: date
    last_trade_date: date
    lag_days: int
    lag_ratio: float
    value: float = 0.0
    shares: float = 0.0
    role: str | None = None
    role_weight: float | None = None
    cluster_count: int = 0
    is_10b5_1: bool = False
    n_lines: int = 0
    filing_ids: set[int] = field(default_factory=set)
    trade_ids: list[int] = field(default_factory=list)

    @property
    def avg_price(self) -> float | None:
        return self.value / self.shares if self.shares > 0 and self.value > 0 else None

    @property
    def entry_date(self) -> date:
        return entry_date(self.filing_date, self.accepted_at)


def entry_date(filing_date: date, accepted_at: datetime | None) -> date:
    """The first NYSE open the filing could be traded at.

    With an EDGAR acceptance time: that day's open if accepted before 9:30 Eastern on a
    trading day, else the next trading day's open. Without one (all bulk history), the
    next trading day after the filing date: Form 4s count as filed that day up to 10 pm,
    so a same-day open could be before the filing existed.
    """
    if accepted_at is not None:
        local = accepted_at.astimezone(EASTERN) if accepted_at.tzinfo else accepted_at
        d = local.date()
        if local.time() < MARKET_OPEN and NYSE.is_business_day(d):
            return d
        return NYSE.add(d, 1)
    return NYSE.add(filing_date, 1)


@dataclass
class Exclusions:
    counts: Counter = field(default_factory=Counter)

    def add(self, reason: str) -> None:
        self.counts[reason] += 1


def load_events(conn: psycopg.Connection) -> tuple[list[Event], Exclusions]:
    ex = Exclusions()
    events: dict[tuple, Event] = {}
    with conn.cursor(name="bt_trades") as cur:
        cur.itersize = 50_000
        cur.execute(EVENTS_SQL)
        for row in cur:
            (trade_id, filing_id, filer_id, cik, ticker, side, trade_date, shares, price, role,
             is_10b5_1, doc_type, accepted_at, filing_date, lag_days, lag_ratio, role_weight,
             cluster_count, value, duplicate_of, flags) = row
            ex.add("lines")
            if filing_date is None:
                ex.add("no_features")
                continue
            if duplicate_of is not None:
                ex.add("duplicate")
                continue
            if doc_type.endswith("/A"):
                ex.add("amendment")
                continue
            if lag_days is None:
                ex.add("trade_after_filing")
                continue
            if not ticker:
                ex.add("no_ticker")
                continue
            key = (filer_id, ticker, side, filing_date)
            e = events.get(key)
            if e is None:
                e = events[key] = Event(
                    filer_id=filer_id, issuer_cik=cik, ticker=ticker, side=side,
                    filing_date=filing_date, accepted_at=accepted_at,
                    first_trade_date=trade_date, last_trade_date=trade_date,
                    lag_days=lag_days, lag_ratio=float(lag_ratio or 0),
                )
            if accepted_at is not None and (e.accepted_at is None or accepted_at < e.accepted_at):
                e.accepted_at = accepted_at
            if trade_date < e.first_trade_date:
                e.first_trade_date = trade_date
            e.last_trade_date = max(e.last_trade_date, trade_date)
            if lag_days > e.lag_days:
                e.lag_days, e.lag_ratio = lag_days, float(lag_ratio or 0)
            if value is not None and shares is not None and float(shares) > 0:
                e.value += float(value)
                e.shares += float(shares)
            if role_weight is not None and (e.role_weight is None or role_weight > e.role_weight):
                e.role_weight, e.role = float(role_weight), role
            e.cluster_count = max(e.cluster_count, cluster_count or 0)
            e.is_10b5_1 = e.is_10b5_1 or bool(is_10b5_1)
            e.n_lines += 1
            e.filing_ids.add(filing_id)
            e.trade_ids.append(trade_id)
    out = sorted(events.values(), key=lambda e: (e.filing_date, e.ticker, e.side, e.filer_id))
    ex.counts["events"] = len(out)
    return out, ex
