"""Daily prices from Tiingo's free tier.

Free plan limits: 50 requests/hour, 1,000/day, 500 unique symbols/month, 1 GB/month.
One request returns a ticker's whole history, so a backfill is one call per ticker, and
the symbol limit, not the request limits, decides how many tickers load per month.
Delisted tickers are served by the same endpoint.
"""

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

import httpx

BASE = "https://api.tiingo.com/tiingo/daily"
HOURLY_LIMIT = 50
DAILY_LIMIT = 1000
MONTHLY_SYMBOL_LIMIT = 500
CSV_COLUMNS = "date,open,high,low,close,volume,adjOpen,adjClose"


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


class RateLimited(RuntimeError):
    """Tiingo refused the call because a quota is spent. kind: hourly | daily | symbols."""

    def __init__(self, kind: str, detail: str):
        super().__init__(detail)
        self.kind = kind


class AuthError(RuntimeError):
    pass


def _error_text(resp: httpx.Response) -> str | None:
    """Tiingo's error message, or None for a data response.

    JSON errors are {"detail": "Error: ..."}; CSV requests get a bare "Error: ..." body,
    often with status 200.
    """
    text = resp.text.strip()
    if text.startswith("Error:"):
        return text
    if text.startswith("{"):
        try:
            detail = resp.json().get("detail")
        except ValueError:
            return None
        return str(detail) if detail else None
    return None


def _raise_for_error(ticker: str, resp: httpx.Response) -> None:
    msg = _error_text(resp)
    low = (msg or "").lower()
    if resp.status_code in (401, 403) or "invalid token" in low or "authenticat" in low:
        raise AuthError(msg or f"HTTP {resp.status_code}")
    if resp.status_code == 429 or any(w in low for w in ("run over", "allocation", "rate limit")):
        if "symbol" in low:
            kind = "symbols"
        elif "day" in low or "daily" in low:
            kind = "daily"
        else:
            kind = "hourly"
        raise RateLimited(kind, msg or f"HTTP {resp.status_code}")
    if resp.status_code == 404 or "not found" in low:
        raise TickerNotFound(ticker)
    if msg:
        raise httpx.HTTPStatusError(msg, request=resp.request, response=resp)
    resp.raise_for_status()


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
            volume=int(Decimal(str(r["volume"]))) if r.get("volume") is not None else None,
        )
        for r in rows
    ]


def symbol(ticker: str) -> str:
    # Tiingo spells share classes with a dash (BRK-B), SEC with a dot (BRK.B).
    return ticker.replace(".", "-").lower()


def parse_csv(ticker: str, text: str) -> list[PriceBar]:
    lines = text.strip().splitlines()
    if not lines:
        return []
    cols = lines[0].split(",")
    return parse_prices(ticker, [dict(zip(cols, (v or None for v in ln.split(",")))) for ln in lines[1:]])


class TiingoClient:
    def __init__(self, api_key: str, *, transport: httpx.BaseTransport | None = None):
        self._http = httpx.Client(
            headers={"Authorization": f"Token {api_key}", "Content-Type": "application/json"},
            timeout=60.0,
            transport=transport,
        )

    def daily(self, ticker: str, start: date, end: date | None = None) -> list[PriceBar]:
        params = {"startDate": start.isoformat(), "format": "json"}
        if end:
            params["endDate"] = end.isoformat()
        resp = self._http.get(f"{BASE}/{symbol(ticker)}/prices", params=params)
        _raise_for_error(ticker, resp)
        return parse_prices(ticker, resp.json())

    def daily_csv(self, ticker: str, start: date) -> str:
        """Raw CSV history (CSV_COLUMNS), about a third of the JSON's bandwidth."""
        params = {"startDate": start.isoformat(), "format": "csv", "columns": CSV_COLUMNS}
        resp = self._http.get(f"{BASE}/{symbol(ticker)}/prices", params=params)
        _raise_for_error(ticker, resp)
        text = resp.text
        if not text.startswith("date,"):
            raise httpx.HTTPStatusError(
                f"unexpected body: {text[:200]!r}", request=resp.request, response=resp
            )
        return text

    def close(self) -> None:
        self._http.close()
