"""Per-trade features, computed point in time.

Everything here is pure: `compute` takes trades and a price lookup and returns one
FeatureRow per trade. History-based features (filer lag z-score, size percentile,
cluster count) only look at trades that were public by the time this trade was, so
the same code serves the live ranking and the backtest.

"Public" means the trade's filing date: the Eastern date of `filed_at`. A trade that
an amendment repeats or corrects keeps the original filing's date, as the spec asks,
and is left out of every history so it never counts twice.
"""

import bisect
import math
from collections import defaultdict, deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

from .calendar import NYSE, SEC

FEATURE_VERSION = 1
EASTERN = ZoneInfo("America/New_York")

# Legal deadline in SEC business days: Form 4 is 2; a PTR's 45 calendar days is ~31.
DEADLINE_DAYS = {"edgar": 2, "house": 31, "senate": 31}

# Spec: CEO and CFO above other officers, officers above directors, 10% owners lowest.
ROLE_WEIGHTS = {
    "CEO": 1.0,
    "CFO": 1.0,
    "officer": 0.75,
    "director": 0.5,
    "10% owner": 0.25,
}

CLUSTER_WINDOW_DAYS = 14        # calendar days either side of the trade date
ZSCORE_MIN_HISTORY = 3          # prior filings needed before a filer has a "normal" lag
ZSCORE_MIN_STD = 0.5            # business days; keeps always-on-time filers finite
SIZE_MIN_FILER_HISTORY = 5      # prior same-side trades before using the filer's own history
SIZE_MARKET_WINDOW_DAYS = 365   # otherwise rank against all same-side trades this past year
SIZE_MARKET_MIN = 100
PRICE_MAX_GAP_DAYS = 5          # calendar days a close may lag the date it stands in for


@dataclass
class TradeInput:
    trade_id: int
    filing_id: int
    filer_id: int
    source: str
    document_type: str
    amends_filing_id: int | None
    filed_at: datetime
    ticker: str | None
    side: str
    trade_date: date
    shares: Decimal | None = None
    price: Decimal | None = None
    amount_low: Decimal | None = None
    amount_high: Decimal | None = None
    role: str | None = None
    is_10b5_1: bool = False


@dataclass
class FeatureRow:
    trade_id: int
    filing_date: date
    lag_days: int | None = None
    lag_ratio: float | None = None
    drift_pct: float | None = None
    days_since_filing: int | None = None
    filer_lag_zscore: float | None = None
    cluster_count: int | None = None
    role_weight: float | None = None
    size_score: float | None = None
    size_basis: str | None = None
    trade_value: float | None = None
    committee_match: bool | None = None
    duplicate_of_trade_id: int | None = None
    flags: list[str] = field(default_factory=list)


class PriceSeries:
    """Daily closes for one ticker, looked up as of a date."""

    def __init__(self, bars: Iterable[tuple[date, float | Decimal | None]]):
        clean = sorted((d, float(c)) for d, c in bars if c is not None and float(c) > 0)
        self.dates = [d for d, _ in clean]
        self.closes = [c for _, c in clean]

    def close_as_of(self, d: date) -> tuple[date, float] | None:
        """The last close on or before d, if it is recent enough to stand in for d."""
        i = bisect.bisect_right(self.dates, d) - 1
        if i < 0 or (d - self.dates[i]).days > PRICE_MAX_GAP_DAYS:
            return None
        return self.dates[i], self.closes[i]


def filing_date(filed_at: datetime) -> date:
    if filed_at.tzinfo is None:
        return filed_at.date()
    return filed_at.astimezone(EASTERN).date()


def trade_value(t: TradeInput) -> float | None:
    """Dollar size: exact for Form 4, the range midpoint for PTRs."""
    if t.shares is not None and t.price is not None and t.shares > 0 and t.price > 0:
        return float(t.shares * t.price)
    if t.amount_low is not None and t.amount_high is not None and t.amount_high > 0:
        return float(t.amount_low + t.amount_high) / 2
    return None


def drift_pct(side: str, trade_close: float, filing_close: float) -> float:
    """Return from trade-date close to filing-date close, in percent, signed so a
    positive number always means the move went the insider's way before disclosure.
    """
    move = (filing_close / trade_close - 1) * 100
    return -move if side == "sell" else move


def percentile(sorted_values: list[float], x: float) -> float:
    """Mid-rank percentile of x within a sorted sample, 0 to 1."""
    lo = bisect.bisect_left(sorted_values, x)
    hi = bisect.bisect_right(sorted_values, x)
    return (lo + 0.5 * (hi - lo)) / len(sorted_values)


class _LogHistogram:
    """Counts of positive values in log-spaced bins, with add, remove and rank.

    Ranks against a rolling year of trades would be O(n) per insert with a sorted
    list; a Fenwick tree over 0.005-decade bins keeps it O(log n) with ranks exact
    to about 1% in dollar terms.
    """

    STEP = 0.005
    LO, HI = 0.0, 12.0  # $1 to $1T

    def __init__(self):
        self.size = int((self.HI - self.LO) / self.STEP) + 1
        self.tree = [0] * (self.size + 1)
        self.counts = [0] * self.size
        self.total = 0

    def _bin(self, v: float) -> int:
        b = int((math.log10(max(v, 1.0)) - self.LO) / self.STEP)
        return min(max(b, 0), self.size - 1)

    def _update(self, b: int, delta: int) -> None:
        self.counts[b] += delta
        self.total += delta
        i = b + 1
        while i <= self.size:
            self.tree[i] += delta
            i += i & -i

    def _below(self, b: int) -> int:
        i, s = b, 0
        while i > 0:
            s += self.tree[i]
            i -= i & -i
        return s

    def add(self, v: float) -> None:
        self._update(self._bin(v), 1)

    def remove(self, v: float) -> None:
        self._update(self._bin(v), -1)

    def rank(self, v: float) -> float:
        b = self._bin(v)
        return (self._below(b) + 0.5 * self.counts[b]) / self.total


def _duplicates(trades: list[TradeInput]) -> dict[int, tuple[TradeInput, bool]]:
    """Map amendment trades that restate a trade of the original filing to that trade.

    An exact repeat (same side, date, shares and price) matches first; otherwise a line
    with the same side and date is taken to correct it. Returns {trade_id: (original,
    exact)}; each original line is matched at most once.
    """
    by_filing: dict[int, list[TradeInput]] = defaultdict(list)
    for t in trades:
        by_filing[t.filing_id].append(t)
    out: dict[int, tuple[TradeInput, bool]] = {}
    amendments: dict[int, list[TradeInput]] = defaultdict(list)
    for t in trades:
        if t.amends_filing_id is not None:
            amendments[t.filing_id].append(t)
    for lines in amendments.values():
        originals = by_filing.get(lines[0].amends_filing_id, [])
        used: set[int] = set()
        for exact in (True, False):
            for t in lines:
                if t.trade_id in out:
                    continue
                for o in originals:
                    if o.trade_id in used or (o.side, o.trade_date) != (t.side, t.trade_date):
                        continue
                    if exact and (o.shares, o.price) != (t.shares, t.price):
                        continue
                    out[t.trade_id] = (o, exact)
                    used.add(o.trade_id)
                    break
    return out


def compute(
    trades: list[TradeInput],
    prices: Callable[[str], PriceSeries | None],
    as_of: date,
) -> list[FeatureRow]:
    dups = _duplicates(trades)
    filed = {t.trade_id: filing_date(t.filed_at) for t in trades}
    for tid, (original, _) in dups.items():
        filed[tid] = filing_date(original.filed_at)

    rows = {t.trade_id: FeatureRow(trade_id=t.trade_id, filing_date=filed[t.trade_id]) for t in trades}

    # Pass 1, per trade and in filing order: lag, staleness, role, size, filer z-score.
    order = sorted(trades, key=lambda t: (filed[t.trade_id], t.filing_id, t.trade_id))
    filer_lags: dict[int, list[int]] = defaultdict(list)          # one lag per prior filing
    filer_values: dict[tuple[int, str], list[float]] = defaultdict(list)  # kept sorted
    market = {"buy": _LogHistogram(), "sell": _LogHistogram()}
    market_window: deque[tuple[date, str, float]] = deque()
    pending: list[TradeInput] = []  # today's trades, added to history once the day is done

    def commit(day_trades: list[TradeInput]) -> None:
        filing_lags: dict[tuple[int, int], int] = {}
        for t in day_trades:
            if t.trade_id in dups:
                continue
            r = rows[t.trade_id]
            if r.lag_days is not None and not t.document_type.endswith("/A"):
                key = (t.filer_id, t.filing_id)
                filing_lags[key] = max(filing_lags.get(key, r.lag_days), r.lag_days)
            if r.trade_value is not None:
                bisect.insort(filer_values[(t.filer_id, t.side)], r.trade_value)
                market[t.side].add(r.trade_value)
                market_window.append((r.filing_date, t.side, r.trade_value))
        for (filer_id, _), lag in filing_lags.items():
            filer_lags[filer_id].append(lag)

    current_day = None
    for t in order:
        r = rows[t.trade_id]
        day = r.filing_date
        if day != current_day:
            commit(pending)
            pending = []
            current_day = day
            while market_window and (day - market_window[0][0]).days > SIZE_MARKET_WINDOW_DAYS:
                _, side, v = market_window.popleft()
                market[side].remove(v)
        pending.append(t)

        if t.trade_id in dups:
            original, exact = dups[t.trade_id]
            r.duplicate_of_trade_id = original.trade_id
            r.flags.append("amendment_duplicate" if exact else "amendment_correction")
        elif t.amends_filing_id is not None or t.document_type.endswith("/A"):
            r.flags.append("amendment")
        if t.is_10b5_1:
            r.flags.append("10b5_1")

        lag = SEC.business_days_between(t.trade_date, day)
        if lag < 0:
            r.flags.append("trade_after_filing")
        else:
            r.lag_days = lag
            deadline = DEADLINE_DAYS.get(t.source)
            if deadline:
                r.lag_ratio = lag / deadline
        r.days_since_filing = max(0, NYSE.business_days_between(day, as_of))
        r.role_weight = ROLE_WEIGHTS.get(t.role) if t.role else None

        history = filer_lags.get(t.filer_id, [])
        if r.lag_days is not None and len(history) >= ZSCORE_MIN_HISTORY:
            mean = sum(history) / len(history)
            std = math.sqrt(sum((x - mean) ** 2 for x in history) / len(history))
            r.filer_lag_zscore = (r.lag_days - mean) / max(std, ZSCORE_MIN_STD)

        r.trade_value = trade_value(t)
        if r.trade_value is None:
            r.flags.append("no_value")
        else:
            own = filer_values.get((t.filer_id, t.side), [])
            if len(own) >= SIZE_MIN_FILER_HISTORY:
                r.size_score, r.size_basis = percentile(own, r.trade_value), "filer"
            elif market[t.side].total >= SIZE_MARKET_MIN:
                r.size_score, r.size_basis = market[t.side].rank(r.trade_value), "market"
    commit(pending)

    # Pass 2, per ticker and side: clusters of distinct filers trading together.
    by_ticker: dict[tuple[str, str], list[TradeInput]] = defaultdict(list)
    for t in trades:
        if t.ticker and t.trade_id not in dups:
            by_ticker[(t.ticker, t.side)].append(t)
    window = timedelta(days=CLUSTER_WINDOW_DAYS)
    for group in by_ticker.values():
        group.sort(key=lambda t: t.trade_date)
        dates = [t.trade_date for t in group]
        for t in group:
            lo = bisect.bisect_left(dates, t.trade_date - window)
            hi = bisect.bisect_right(dates, t.trade_date + window)
            day = rows[t.trade_id].filing_date
            rows[t.trade_id].cluster_count = len(
                {o.filer_id for o in group[lo:hi] if rows[o.trade_id].filing_date <= day}
            )
    for tid, (original, _) in dups.items():
        rows[tid].cluster_count = rows[original.trade_id].cluster_count

    # Pass 3, per ticker: drift from the trade-date close to the filing-date close.
    tickers: dict[str, list[TradeInput]] = defaultdict(list)
    for t in trades:
        if t.ticker:
            tickers[t.ticker].append(t)
        else:
            rows[t.trade_id].flags.append("no_ticker")
    for ticker, group in tickers.items():
        series = prices(ticker)
        for t in group:
            r = rows[t.trade_id]
            if "trade_after_filing" in r.flags:
                continue
            if series is None or not series.dates:
                r.flags.append("no_prices")
                continue
            start = series.close_as_of(t.trade_date)
            end = series.close_as_of(r.filing_date)
            if start is None:
                r.flags.append("no_trade_price")
            if end is None:
                r.flags.append("no_filing_price")
            if start and end:
                r.drift_pct = drift_pct(t.side, start[1], end[1])

    return [rows[t.trade_id] for t in trades]
