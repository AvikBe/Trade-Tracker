"""Daily prices from Tiingo's free tier.

Free plan limits: 50 requests/hour, 1,000/day, 1 GB/month. One request returns a
ticker's whole history, so a backfill is one call per ticker and the quota decides
how many tickers load per run. Delisted tickers are served by the same endpoint.
"""

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

import httpx

BASE = "https://api.tiingo.com/tiingo/daily"
HOURLY_LIMIT = 50
DAILY_LIMIT = 1000


@dataclass
class PriceBar:
    ticker: str
    date: date
    open: Decimal | None
    high: Decimal | None
    low: Decimal | None
    close: Decimal | None
    adj_open: Decimal | None
    adj_close: Decimal | None
    volume: int | None


class TickerNotFound(LookupError):
    pass


def _d(v) -> Decimal | None:
    return None if v is None else Decimal(str(v))


def parse_prices(ticker: str, rows: list[dict]) -> list[PriceBar]:
    return [
        PriceBar(
            ticker=ticker,
            date=date.fromisoformat(r["date"][:10]),
            open=_d(r.get("open")),
            high=_d(r.get("high")),
            low=_d(r.get("low")),
            close=_d(r.get("close")),
            adj_open=_d(r.get("adjOpen")),
            adj_close=_d(r.get("adjClose")),
            volume=int(r["volume"]) if r.get("volume") is not None else None,
        )
        for r in rows
    ]


class TiingoClient:
    def __init__(self, api_key: str, *, transport: httpx.BaseTransport | None = None):
        self._http = httpx.Client(
            headers={"Authorization": f"Token {api_key}", "Content-Type": "application/json"},
            timeout=60.0,
            transport=transport,
        )

    def daily(self, ticker: str, start: date, end: date | None = None) -> list[PriceBar]:
        # Tiingo spells share classes with a dash (BRK-B), SEC with a dot (BRK.B).
        symbol = ticker.replace(".", "-").lower()
        params = {"startDate": start.isoformat(), "format": "json"}
        if end:
            params["endDate"] = end.isoformat()
        resp = self._http.get(f"{BASE}/{symbol}/prices", params=params)
        if resp.status_code == 404:
            raise TickerNotFound(ticker)
        resp.raise_for_status()
        return parse_prices(ticker, resp.json())

    def close(self) -> None:
        self._http.close()
