"""Insider skill: have this insider's earlier buys beaten SPY? Point in time.

avik, 2026-11-09: "hone in on executives who consistently improve their portfolio
outpacing the s&p 500". For every buy we look up the insider's record as it stood the
day before the filing:

- **Episodes.** Every earlier open-market buy by the same person (reporting owner CIK),
  at any company, not under a 10b5-1 plan. Buys of one stock filed in the same calendar
  month are one episode (the first filing's outcome), so a run of daily purchases counts
  once instead of ten times.
- **Outcome.** The episode's h-session excess return over SPY, gross of costs (entry at
  the open after filing, exit at the close h sessions later, as in the backtest), clipped
  to [-100%, +100%] so one penny stock can't make a record.
- **Known.** An episode counts only once its exit date is *before* the new filing date.
  A 60-day record therefore lags a 20-day one by about two months.
- **Shrinkage.** raw = the mean of the n known outcomes. The record is
  prior + n / (n + k) * (raw - prior), where prior is the mean of every insider's
  episodes known by then (all insiders, point in time) and k is estimated once from
  the variance of insiders' records known before the first test year (`estimate_k`):
  k = within-insider variance / between-insider variance. With little true skill, k is
  large and records stay near the prior until there are many episodes.
- **Minimum.** Fewer than `MIN_EPISODES` known episodes is "no record".

Sources: the backtest's own buy results (2015 on, every price check applied), plus the
SEC data sets' 2014 buys (`tt insider-history`), priced from the same bars, for insiders
active before the database starts. Yahoo bars begin in January 2014, so 2012-2013 buys
can't be priced.
"""

from __future__ import annotations

import bisect
import csv
import gzip
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

HORIZONS = (20, 60)
MIN_EPISODES = 3
CLIP = (-1.0, 1.0)
K_BOUNDS = (1.0, 100.0)
DEFAULT_K = 20.0
MIN_PRIOR_EPISODES = 100     # fewer known episodes overall and the prior is 0
MAX_ENTRY_GAP_DAYS = 7       # calendar days from filing to the first session after it


@dataclass(frozen=True)
class Episode:
    owner: str
    issuer: str
    filed: date
    known: date      # exit day: the outcome is public from the day after
    excess: float    # clipped


@dataclass(frozen=True)
class Record:
    n: int
    raw: float | None        # mean known outcome
    hit: float | None        # share of known outcomes above 0
    prior: float
    shrunk: float | None     # None when n < MIN_EPISODES

    @property
    def has_record(self) -> bool:
        return self.shrunk is not None


def clip(v: float) -> float:
    return min(max(v, CLIP[0]), CLIP[1])


def owner_key(cik: str | int | None) -> str | None:
    if cik is None:
        return None
    s = str(cik).strip().lstrip("0")
    return s or None


def collapse(rows) -> list[Episode]:
    """One episode per (owner, issuer, filing month): the earliest filing's outcome.

    `rows` are (owner, issuer, filed, known, excess) with excess unclipped."""
    best: dict[tuple, tuple] = {}
    for owner, issuer, filed, known, excess in rows:
        if owner is None or excess is None:
            continue
        k = (owner, issuer, filed.year, filed.month)
        cur = best.get(k)
        if cur is None or (filed, known) < (cur[2], cur[3]):
            best[k] = (owner, issuer, filed, known, excess)
    return sorted((Episode(o, i, f, kn, clip(x)) for o, i, f, kn, x in best.values()),
                  key=lambda e: (e.known, e.owner, e.issuer, e.filed))


def estimate_k(episodes: list[Episode], before: date) -> float:
    """Within-insider variance over between-insider variance of true means, from the
    episodes known before `before` (method of moments, insiders with 2+ episodes)."""
    by_owner: dict[str, list[float]] = defaultdict(list)
    for e in episodes:
        if e.known < before:
            by_owner[e.owner].append(e.excess)
    groups = [v for v in by_owner.values() if len(v) >= 2]
    if len(groups) < 30:
        return DEFAULT_K
    ss_within = sum(sum((x - sum(v) / len(v)) ** 2 for x in v) for v in groups)
    df_within = sum(len(v) - 1 for v in groups)
    s2 = ss_within / df_within
    means = [sum(v) / len(v) for v in groups]
    grand = sum(means) / len(means)
    var_means = sum((m - grand) ** 2 for m in means) / (len(means) - 1)
    tau2 = var_means - sum(s2 / len(v) for v in groups) / len(groups)
    if tau2 <= 0:
        return K_BOUNDS[1]
    return min(max(s2 / tau2, K_BOUNDS[0]), K_BOUNDS[1])


class SkillBook:
    """Point-in-time records for one horizon. `record(owner, d)` uses only episodes
    whose exit is strictly before d."""

    def __init__(self, episodes: list[Episode], k: float, min_episodes: int = MIN_EPISODES):
        self.k = k
        self.min_episodes = min_episodes
        self.n_episodes = len(episodes)
        eps = sorted(episodes, key=lambda e: e.known)
        self.episodes = eps
        self._all_dates = [e.known for e in eps]
        self._all_sum = [0.0]
        for e in eps:
            self._all_sum.append(self._all_sum[-1] + e.excess)
        by_owner: dict[str, list[Episode]] = defaultdict(list)
        for e in eps:
            by_owner[e.owner].append(e)
        self._dates: dict[str, list[date]] = {}
        self._sum: dict[str, list[float]] = {}
        self._hits: dict[str, list[int]] = {}
        for o, v in by_owner.items():
            self._dates[o] = [e.known for e in v]
            s, h = [0.0], [0]
            for e in v:
                s.append(s[-1] + e.excess)
                h.append(h[-1] + (e.excess > 0))
            self._sum[o], self._hits[o] = s, h

    def prior(self, d: date) -> float:
        i = bisect.bisect_left(self._all_dates, d)
        return self._all_sum[i] / i if i >= MIN_PRIOR_EPISODES else 0.0

    def record(self, owner: str | None, d: date) -> Record:
        prior = self.prior(d)
        ds = self._dates.get(owner) if owner else None
        n = bisect.bisect_left(ds, d) if ds else 0
        if n == 0:
            return Record(0, None, None, prior, None)
        raw = self._sum[owner][n] / n
        hit = self._hits[owner][n] / n
        shrunk = prior + n / (n + self.k) * (raw - prior) if n >= self.min_episodes else None
        return Record(n, raw, hit, prior, shrunk)


# ---------------------------------------------------------------- sources

def rows_from_results(results, h: int, owner_keys: dict[int, str] | None = None):
    """(owner, issuer, filed, known, excess) for every priced, non-plan buy result."""
    for x in results:
        e = x.event
        if e.side != "buy" or e.is_10b5_1 or not x.ok:
            continue
        leg = x.outcome.legs.get(h)
        r = x.r(h, "spy", net=False)
        if leg is None or r is None:
            continue
        owner = owner_key(owner_keys.get(e.filer_id)) if owner_keys else f"filer:{e.filer_id}"
        yield owner, owner_key(e.issuer_cik) or e.ticker, e.filing_date, leg["exit"], r


def load_history_buys(path: Path, before: date) -> list[tuple[str, str, date]]:
    """(owner, issuer, filed) for open-market buys filed before `before`, one per
    filing day, from an insider-history CSV."""
    opener = gzip.open if str(path).endswith(".gz") else open
    seen = set()
    with opener(path, "rt") as fh:
        for r in csv.DictReader(fh):
            if r["side"] != "buy":
                continue
            filed = date.fromisoformat(r["filed"])
            if filed >= before:
                continue
            k = (owner_key(r["owner_cik"]), owner_key(r["issuer_cik"]), filed)
            if k[0] and k[1]:
                seen.add(k)
    return sorted(seen, key=lambda k: (k[2], k[0], k[1]))


def price_outcome(prices, ticker: str, filed: date, h: int) -> tuple[date, float] | None:
    """(exit day, excess over SPY) for a buy filed on `filed`: bought at the open of the
    first session after the filing day, sold at the close of the h-th session."""
    bars, spy = prices.get(ticker), prices.get("SPY")
    if bars is None or spy is None:
        return None
    i = bars.index_on_or_after(filed + timedelta(days=1))
    if i is None or i + h - 1 >= len(bars):
        return None
    a, b = bars.bars[i], bars.bars[i + h - 1]
    if (a.date - filed).days > MAX_ENTRY_GAP_DAYS or not a.adj_open:
        return None
    sa, sb = spy.at(a.date), spy.at(b.date)
    if sa is None or sb is None or not sa.adj_open:
        return None
    return b.date, (b.adj_close / a.adj_open - 1) - (sb.adj_close / sa.adj_open - 1)


def rows_from_history(buys, tickers: dict[str, str], prices, h: int):
    for owner, issuer, filed in buys:
        t = tickers.get(issuer)
        if not t:
            continue
        out = price_outcome(prices, t, filed, h)
        if out is not None:
            yield owner, issuer, filed, out[0], out[1]


def build_books(results, prices, owner_keys: dict[int, str] | None = None,
                history: Path | None = None, tickers: dict[str, str] | None = None,
                k_before: date = date(2017, 1, 1), db_start: date = date(2015, 1, 1),
                horizons=HORIZONS) -> dict[int, SkillBook]:
    """One SkillBook per horizon. History buys filed before `db_start` (when the
    database's own events begin) are added, priced from `prices`."""
    hist = load_history_buys(history, db_start) if history else []
    books = {}
    for h in horizons:
        rows = list(rows_from_results(results, h, owner_keys))
        if hist and tickers:
            rows += list(rows_from_history(hist, tickers, prices, h))
        eps = collapse(rows)
        books[h] = SkillBook(eps, estimate_k(eps, k_before))
    return books
