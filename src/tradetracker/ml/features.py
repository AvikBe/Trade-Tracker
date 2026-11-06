"""One point-in-time feature row per shown buy, for the gradient-boosted models.

Everything here was public before the position is opened at the entry-day open:

- **The score's signals** (`scoring.signals`): role, trade value, stake change, cluster
  size, repeat buys, 52-week drawdown, track record, dollar volume, drift (the run-up
  between the trade and entry), market cap, beta, volatility and insider type.
- **The filing**: disclosure lag, number of lines, the estimated round-trip cost, and
  the trade's size against the company's market cap and daily dollar volume.
- **The stock's own prices**, from bars strictly before the entry day: returns over 5 to
  250 sessions, 12-1 momentum, distance from the 52-week low, sessions since the high,
  20-day volatility and its ratio to the 250-day one, the largest daily gain in 20
  sessions, the trend in dollar volume, and the share price.
- **The market**, from SPY and IWM bars before entry: SPY's 20- and 250-session returns,
  its drawdown from its 52-week high and its 20-day volatility, and small caps (IWM)
  against SPY over 60 sessions.
- **Other insiders' trades**, from events filed on earlier days only (a same-day
  filing may come after this one): distinct buyers and sellers of the company in the
  previous 90 days, the company's net purchase ratio over 180 days (Lakonishok and Lee
  2001), this insider's sales of the stock over the year, their earlier buys of any
  stock, the days since anyone last bought the stock, and market-wide insider buying
  in the previous 30 days against its own previous year.

A missing value stays missing (NaN): LightGBM learns where to send it.
"""

from __future__ import annotations

import bisect
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta

from ..backtest.events import Event
from ..backtest.study import Result
from ..scoring.signals import Signals

NAN = float("nan")

# Numeric features, in the order the models see them.
NUMERIC = [
    # the score's signals
    "log_value", "stake_change", "cluster_count", "repeat_buys", "drawdown", "track_record",
    "track_n", "log_dollar_volume", "drift", "log_market_cap", "beta", "vol",
    # the filing
    "lag_days", "n_lines", "cost", "value_to_cap", "value_to_dollar_volume", "month",
    # the stock's prices
    "ret_5", "ret_20", "ret_60", "ret_120", "ret_250", "mom_12_1", "low_distance",
    "sessions_since_high", "vol_20", "vol_ratio", "max_ret_20", "dollar_volume_trend",
    "log_price",
    # the market
    "spy_ret_20", "spy_ret_250", "spy_drawdown", "spy_vol_20", "iwm_minus_spy_60",
    # other insiders
    "firm_buyers_90", "firm_sellers_90", "firm_net_purchase_180", "insider_sells_365",
    "insider_prior_buys", "days_since_firm_buy", "market_buying_30",
]
CATEGORICAL = ["role", "sector", "insider_type"]
FEATURES = NUMERIC + CATEGORICAL
# Which group each feature belongs to, for the report's group importance.
GROUPS = {
    **{f: "score signals" for f in NUMERIC[:12]},
    **{f: "filing" for f in NUMERIC[12:18]},
    **{f: "stock prices" for f in NUMERIC[18:31]},
    **{f: "market" for f in NUMERIC[31:36]},
    **{f: "other insiders" for f in NUMERIC[36:]},
    "role": "score signals", "insider_type": "score signals", "sector": "filing",
}
TWO_INPUT_FEATURES = ["drawdown", "drift"]

YEAR_SESSIONS = 250
MIN_SESSIONS = 60
DAYS_SINCE_CAP = 730


def _log(v: float | None) -> float:
    return math.log(v) if v is not None and v > 0 else NAN


def _f(v) -> float:
    return NAN if v is None else float(v)


def _ret(closes: list[float], i: int, k: int) -> float:
    return closes[i] / closes[i - k] - 1 if i - k >= 0 else NAN


def _ann_vol(closes: list[float], i: int, k: int) -> float:
    if i - k < 0:
        return NAN
    r = [closes[j] / closes[j - 1] - 1 for j in range(i - k + 1, i + 1)]
    return statistics.stdev(r) * math.sqrt(252) if len(r) > 1 else NAN


def price_features(bars, entry: date) -> dict:
    """The stock's own price features, from bars before `entry` (never on it)."""
    out = {k: NAN for k in ("ret_5", "ret_20", "ret_60", "ret_120", "ret_250", "mom_12_1",
                            "low_distance", "sessions_since_high", "vol_20", "vol_ratio",
                            "max_ret_20", "dollar_volume_trend", "log_price")}
    if bars is None:
        return out
    i = bars.index_before(entry)
    if i is None:
        return out
    s0 = max(0, i - YEAR_SESSIONS - 1)      # only the last year matters
    closes = [b.adj_close for b in bars.bars[s0: i + 1]]
    last = bars.bars[i]
    n, i = i + 1, i - s0                       # sessions of history; local index
    out["log_price"] = _log(last.close)
    for k in (5, 20, 60, 120, 250):
        out[f"ret_{k}"] = _ret(closes, i, k)
    if n > YEAR_SESSIONS:
        out["mom_12_1"] = closes[i - 20] / closes[i - YEAR_SESSIONS] - 1
    if n >= MIN_SESSIONS:
        lo = max(0, i - YEAR_SESSIONS + 1)
        window = closes[lo: i + 1]
        out["low_distance"] = closes[i] / min(window) - 1
        peak = max(range(len(window)), key=lambda j: (window[j], j))
        out["sessions_since_high"] = len(window) - 1 - peak
        out["vol_20"] = _ann_vol(closes, i, 20)
        long_vol = _ann_vol(closes, i, min(i, YEAR_SESSIONS))
        if long_vol and long_vol == long_vol and out["vol_20"] == out["vol_20"]:
            out["vol_ratio"] = out["vol_20"] / long_vol
    if i >= 20:
        out["max_ret_20"] = max(closes[j] / closes[j - 1] - 1 for j in range(i - 19, i + 1))
    dv = [(b.close or 0) * (b.volume or 0) for b in bars.bars[max(0, n - 120): n]]
    if len(dv) >= MIN_SESSIONS:
        recent, longer = statistics.median(dv[-20:]), statistics.median(dv)
        out["dollar_volume_trend"] = math.log(recent / longer) if recent > 0 and longer > 0 else NAN
    return out


class Market:
    """SPY and IWM features by entry day (memoized: many events share a day)."""

    def __init__(self, spy, iwm=None):
        self.spy, self.iwm = spy, iwm
        self._memo: dict[date, dict] = {}

    def features(self, entry: date) -> dict:
        if entry in self._memo:
            return self._memo[entry]
        out = {"spy_ret_20": NAN, "spy_ret_250": NAN, "spy_drawdown": NAN,
               "spy_vol_20": NAN, "iwm_minus_spy_60": NAN}
        i = self.spy.index_before(entry) if self.spy is not None else None
        if i is not None:
            s0 = max(0, i - YEAR_SESSIONS - 1)
            closes = [b.adj_close for b in self.spy.bars[s0: i + 1]]
            full, i = i, i - s0
            out["spy_ret_20"] = _ret(closes, i, 20)
            out["spy_ret_250"] = _ret(closes, i, YEAR_SESSIONS)
            out["spy_drawdown"] = closes[i] / max(closes[max(0, i - YEAR_SESSIONS + 1):]) - 1
            out["spy_vol_20"] = _ann_vol(closes, i, 20)
            j = self.iwm.index_before(entry) if self.iwm is not None else None
            if j is not None and j >= 60 and full >= 60:
                iw = self.iwm.bars
                out["iwm_minus_spy_60"] = (iw[j].adj_close / iw[j - 60].adj_close
                                           - closes[i] / closes[i - 60])
        self._memo[entry] = out
        return out


class Activity:
    """Who else traded, as known before a filing date (events filed on earlier days).

    Built from every event, buys and sells, whatever their liquidity or plan status:
    a 10b5-1 sale is still a sale an investor can see.
    """

    def __init__(self, events: list[Event]):
        self.firm: dict[str, list[tuple[date, str, int]]] = defaultdict(list)
        self.firm_dates: dict[str, list[date]] = {}
        self.sells: dict[tuple, list[date]] = defaultdict(list)     # (filer, firm)
        self.filer_buys: dict[int, list[date]] = defaultdict(list)
        self.market_buys: list[date] = []
        for e in events:
            k = self.key(e)
            self.firm[k].append((e.filing_date, e.side, e.filer_id))
            if e.side == "sell":
                self.sells[(e.filer_id, k)].append(e.filing_date)
            else:
                self.filer_buys[e.filer_id].append(e.filing_date)
                self.market_buys.append(e.filing_date)
        for k, v in self.firm.items():
            v.sort()
            self.firm_dates[k] = [d for d, _, _ in v]
        for v in (*self.sells.values(), *self.filer_buys.values()):
            v.sort()
        self.market_buys.sort()
        self._mkt: dict[date, float] = {}

    @staticmethod
    def key(e: Event) -> str:
        return (e.issuer_cik or "").lstrip("0") or f"ticker:{e.ticker}"

    @staticmethod
    def _count(ds: list[date], lo: date, hi: date) -> int:
        """Dates with lo <= d < hi."""
        return bisect.bisect_left(ds, hi) - bisect.bisect_left(ds, lo)

    def _window(self, k: str, lo: date, hi: date) -> list[tuple[date, str, int]]:
        ds = self.firm_dates.get(k, [])
        return self.firm.get(k, [])[bisect.bisect_left(ds, lo): bisect.bisect_left(ds, hi)]

    def market_buying(self, d: date) -> float:
        """Buy events filed in the 30 days before d, over the average 30 days of the
        year before that (1 is a normal month)."""
        if d not in self._mkt:
            recent = self._count(self.market_buys, d - timedelta(days=30), d)
            year = self._count(self.market_buys, d - timedelta(days=395), d - timedelta(days=30))
            self._mkt[d] = recent / (year * 30 / 365) if year else NAN
        return self._mkt[d]

    def features(self, e: Event) -> dict:
        k, d = self.key(e), e.filing_date
        w90 = self._window(k, d - timedelta(days=90), d)
        w180 = self._window(k, d - timedelta(days=180), d)
        nb = sum(1 for _, s, _ in w180 if s == "buy")
        ns = len(w180) - nb
        ds = self.firm_dates.get(k, [])
        last_buy = None
        for fd, side, _ in reversed(self.firm.get(k, [])[: bisect.bisect_left(ds, d)]):
            if side == "buy":
                last_buy = fd
                break
        return {
            "firm_buyers_90": len({f for _, s, f in w90 if s == "buy"}),
            "firm_sellers_90": len({f for _, s, f in w90 if s == "sell"}),
            "firm_net_purchase_180": (nb - ns) / (nb + ns) if nb + ns else NAN,
            "insider_sells_365": self._count(self.sells.get((e.filer_id, k), []),
                                             d - timedelta(days=365), d),
            "insider_prior_buys": self._count(self.filer_buys.get(e.filer_id, []), date.min, d),
            "days_since_firm_buy": (min((d - last_buy).days, DAYS_SINCE_CAP)
                                    if last_buy else DAYS_SINCE_CAP),
            "market_buying_30": self.market_buying(d),
        }


@dataclass
class Row:
    """One shown buy: its result and signals, its features, and its year."""
    result: Result
    signals: Signals
    features: dict
    year: int


def event_features(x: Result, s: Signals) -> dict:
    e, o = x.event, x.outcome
    cap = s.market_cap
    return {
        "log_value": _log(s.value),
        "stake_change": _f(s.stake_change),
        "cluster_count": _f(s.cluster_count),
        "repeat_buys": _f(s.repeat_buys),
        "drawdown": _f(s.drawdown),
        "track_record": _f(s.track_record),
        "track_n": _f(s.track_n),
        "log_dollar_volume": _log(s.dollar_volume),
        "drift": _f(s.drift),
        "log_market_cap": _log(cap),
        "beta": _f(s.beta),
        "vol": _f(s.vol),
        "lag_days": _f(e.lag_days),
        "n_lines": _f(e.n_lines),
        "cost": _f(o.cost),
        "value_to_cap": s.value / cap if cap and s.value else NAN,
        "value_to_dollar_volume": (s.value / s.dollar_volume
                                   if s.dollar_volume and s.value else NAN),
        "month": float(e.filing_date.month),
        "role": s.role or "unknown",
        "sector": x.sector or "SPY",
        "insider_type": s.insider_type or "unclassified",
    }


def build_rows(pool: list[tuple[Result, Signals]], prices, activity: Activity) -> list[Row]:
    """Feature rows for the shown buys in `pool` (each must have an entry day)."""
    market = Market(prices.get("SPY"), prices.get("IWM"))
    rows = []
    for x, s in pool:
        entry = x.outcome.entry
        f = event_features(x, s)
        f.update(price_features(prices.get(x.event.ticker), entry))
        f.update(market.features(entry))
        f.update(activity.features(x.event))
        rows.append(Row(x, s, f, entry.year))
    return rows
