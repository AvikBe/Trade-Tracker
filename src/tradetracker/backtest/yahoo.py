"""Yahoo Finance daily bars, written in the Tiingo cache layout.

Research use only: Yahoo is free and needs no key, so it gives the backtest breadth
while the Tiingo backfill takes months. Two known limits, both reported by the
backtest rather than hidden:

- Delisted symbols are gone ("No data found, symbol may be delisted"), so a Yahoo-only
  study has survivorship bias.
- A reused symbol returns the current company's history (BBBY today is not Bed Bath &
  Beyond). The backtest checks every event's Form 4 price against the vendor's raw
  close on the trade date and drops events that don't match.

The chart API gives split-adjusted open/close/volume plus a split and dividend
adjusted close. Raw prices are rebuilt by undoing the splits after each date.
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx

from .prices import Bar, safe_name, write_csv

log = logging.getLogger(__name__)

CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
HISTORY_START = date(2014, 1, 1)
EXTRA_SYMBOLS = ["IWM"]  # Russell 2000, a small-cap benchmark next to SPY


def yahoo_symbol(ticker: str) -> str:
    """Yahoo writes share classes with a dash: BRK.B -> BRK-B."""
    return ticker.replace(".", "-").replace("/", "-")


def parse_chart(payload: dict) -> list[Bar]:
    """Bars from a chart API response, with raw open/close and volume rebuilt."""
    result = (payload.get("chart") or {}).get("result") or []
    if not result:
        return []
    r = result[0]
    stamps = r.get("timestamp") or []
    if not stamps:
        return []
    tz = ZoneInfo(r.get("meta", {}).get("exchangeTimezoneName") or "America/New_York")
    quote = (r.get("indicators", {}).get("quote") or [{}])[0]
    adj = ((r.get("indicators", {}).get("adjclose") or [{}])[0]).get("adjclose") or [None] * len(stamps)
    splits = sorted(
        (datetime.fromtimestamp(s["date"], timezone.utc).astimezone(tz).date(),
         s["numerator"] / s["denominator"])
        for s in (r.get("events", {}).get("splits") or {}).values()
        if s.get("numerator") and s.get("denominator")
    )

    bars = []
    for i, ts in enumerate(stamps):
        d = datetime.fromtimestamp(ts, timezone.utc).astimezone(tz).date()
        o, c, v = (quote.get(k, [None] * len(stamps))[i] for k in ("open", "close", "volume"))
        a = adj[i]
        if c is None or a is None or c <= 0:
            continue
        # Splits on a later date than d: Yahoo's prices for d are divided by them.
        factor = 1.0
        for sd, ratio in splits:
            if sd > d:
                factor *= ratio
        bars.append(
            Bar(
                date=d,
                open=None if o is None else o * factor,
                close=c * factor,
                adj_open=None if o is None else o * a / c,
                adj_close=a,
                volume=None if v is None else v / factor,
            )
        )
    # The API sometimes repeats the latest day; keep the last row per date.
    return list({b.date: b for b in bars}.values())


class YahooClient:
    def __init__(self, timeout: float = 30.0):
        self.http = httpx.Client(
            timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0 (research backtest)"},
        )

    def close(self) -> None:
        self.http.close()

    def daily(self, ticker: str, start: date = HISTORY_START, end: date | None = None):
        """Returns (status, bars): status is 'ok', 'not_found' or 'error:<detail>'."""
        p1 = int(datetime(start.year, start.month, start.day, tzinfo=timezone.utc).timestamp())
        end = end or date.today()
        p2 = int(datetime(end.year, end.month, end.day, 23, 59, tzinfo=timezone.utc).timestamp())
        params = {"period1": p1, "period2": p2, "interval": "1d", "events": "div,split"}
        for attempt in range(5):
            try:
                resp = self.http.get(CHART_URL.format(symbol=yahoo_symbol(ticker)), params=params)
            except httpx.HTTPError as e:
                if attempt == 4:
                    return f"error:{type(e).__name__}", []
                time.sleep(2 ** attempt)
                continue
            if resp.status_code == 429 or resp.status_code >= 500:
                time.sleep(5 * 2 ** attempt)
                continue
            if resp.status_code == 404:
                return "not_found", []
            try:
                payload = resp.json()
            except json.JSONDecodeError:
                return f"error:http {resp.status_code}", []
            err = (payload.get("chart") or {}).get("error")
            if err:
                return ("not_found" if err.get("code") == "Not Found" else f"error:{err.get('code')}"), []
            bars = parse_chart(payload)
            return ("ok" if bars else "empty"), bars
        return "error:rate_limited", []


def fetch_all(
    tickers: list[str], root: Path, *, workers: int = 4, refresh: bool = False,
    client_factory=YahooClient,
) -> dict[str, int]:
    """Fetch every ticker's history into `<root>/bars`, recording `<root>/status.json`."""
    bars_dir = root / "bars"
    bars_dir.mkdir(parents=True, exist_ok=True)
    status_path = root / "status.json"
    status: dict[str, dict] = json.loads(status_path.read_text()) if status_path.exists() else {}
    todo = [
        t for t in dict.fromkeys(tickers)
        if refresh or status.get(t, {}).get("status") not in {"ok", "empty", "not_found"}
    ]

    client = client_factory()  # httpx clients are safe to share across threads

    def one(ticker: str) -> tuple[str, dict]:
        st, bars = client.daily(ticker)
        rec = {"status": st, "rows": len(bars), "fetched": date.today().isoformat()}
        if bars:
            write_csv(bars_dir / f"{safe_name(ticker)}.csv.gz", bars)
            rec.update(first=bars[0].date.isoformat(), last=bars[-1].date.isoformat())
        return ticker, rec

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for ticker, rec in pool.map(one, todo):
            status[ticker] = rec
            done += 1
            if done % 250 == 0:
                log.info("yahoo: %d/%d fetched", done, len(todo))
                status_path.write_text(json.dumps(status, indent=0, sort_keys=True))
    client.close()
    status_path.write_text(json.dumps(status, indent=0, sort_keys=True))
    counts: dict[str, int] = {"requested": len(todo)}
    for rec in status.values():
        key = rec["status"].split(":")[0]
        counts[key] = counts.get(key, 0) + 1
    return counts
