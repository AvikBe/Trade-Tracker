"""Where orders go: Alpaca's paper account, or a dry run that only logs them.

Both take the same requests and report fills the same way, so the paper book doesn't
care which one it talks to.

- `AlpacaPaper` talks to https://paper-api.alpaca.markets and nothing else: it refuses
  any other base URL, so this code can never reach a live (real money) account.
- `DryRun` places nothing. It logs each order and fills it from daily bars once they
  exist: an opening order at that session's open, a closing order at its close. That
  matches the backtest's entry (the open) and exit (the close), with no spread.

Orders carry a `client_order_id` made from the session, ticker and side, so a rerun
of the same morning can't place the same order twice.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import date, datetime

import httpx

log = logging.getLogger(__name__)

PAPER_URL = "https://paper-api.alpaca.markets"
FINAL = {"filled", "canceled", "expired", "rejected", "done_for_day_expired"}
OPEN_TIF, CLOSE_TIF, DAY_TIF = "opg", "cls", "day"


@dataclass
class Fill:
    status: str                       # Alpaca's order status, or "pending" in a dry run
    filled_qty: float = 0.0
    filled_price: float | None = None
    filled_at: str | None = None
    broker_id: str | None = None
    tif: str | None = None            # the time in force actually used
    note: str = ""


class BrokerError(RuntimeError):
    pass


class AlpacaPaper:
    kind = "alpaca"

    def __init__(self, key: str, secret: str, base_url: str = PAPER_URL,
                 transport: httpx.BaseTransport | None = None):
        if base_url.rstrip("/") != PAPER_URL:
            raise BrokerError(f"refusing {base_url}: only Alpaca's paper API ({PAPER_URL}) is allowed")
        if not key or not secret:
            raise BrokerError("Alpaca paper keys are missing (ALPACA_API_KEY, ALPACA_SECRET_KEY)")
        self.http = httpx.Client(base_url=PAPER_URL, timeout=30.0, transport=transport,
                                 headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret})

    @classmethod
    def from_env(cls) -> AlpacaPaper:
        base = os.environ.get("APCA_API_BASE_URL", PAPER_URL)
        key = os.environ.get("ALPACA_API_KEY") or os.environ.get("APCA_API_KEY_ID", "")
        secret = os.environ.get("ALPACA_SECRET_KEY") or os.environ.get("APCA_API_SECRET_KEY", "")
        return cls(key, secret, base)

    def _req(self, method: str, path: str, **kw) -> httpx.Response:
        for attempt in range(4):
            try:
                r = self.http.request(method, path, **kw)
            except httpx.TransportError as exc:
                if attempt == 3:
                    raise BrokerError(f"{method} {path}: {exc}") from exc
                continue
            if r.status_code == 429 or r.status_code >= 500:
                if attempt == 3:
                    break
                time.sleep(2 ** attempt)
                continue
            return r
        raise BrokerError(f"{method} {path}: HTTP {r.status_code} {r.text[:200]}")

    def account(self) -> dict:
        r = self._req("GET", "/v2/account")
        if r.status_code != 200:
            raise BrokerError(f"account: HTTP {r.status_code} {r.text[:200]}")
        return r.json()

    def equity(self) -> float:
        return float(self.account()["equity"])

    def position_qty(self, symbol: str) -> float | None:
        """Shares held now (after any split), or 0 when there's no position."""
        r = self._req("GET", f"/v2/positions/{symbol}")
        if r.status_code == 404:
            return 0.0
        if r.status_code != 200:
            raise BrokerError(f"position {symbol}: HTTP {r.status_code} {r.text[:200]}")
        return float(r.json()["qty"])

    def tradable(self, symbol: str) -> bool:
        r = self._req("GET", f"/v2/assets/{symbol}")
        if r.status_code == 404:
            return False
        if r.status_code != 200:
            raise BrokerError(f"asset {symbol}: HTTP {r.status_code} {r.text[:200]}")
        a = r.json()
        return bool(a.get("tradable")) and a.get("status") == "active"

    @staticmethod
    def _fill(o: dict, note: str = "") -> Fill:
        return Fill(status=o.get("status", "unknown"), filled_qty=float(o.get("filled_qty") or 0),
                    filled_price=float(o["filled_avg_price"]) if o.get("filled_avg_price") else None,
                    filled_at=o.get("filled_at"), broker_id=o.get("id"),
                    tif=o.get("time_in_force"), note=note)

    def by_client_id(self, client_id: str) -> Fill | None:
        r = self._req("GET", "/v2/orders:by_client_order_id", params={"client_order_id": client_id})
        if r.status_code == 404:
            return None
        if r.status_code != 200:
            raise BrokerError(f"order {client_id}: HTTP {r.status_code} {r.text[:200]}")
        return self._fill(r.json())

    def submit(self, symbol: str, qty: int, side: str, tif: str, client_id: str,
               session: date) -> Fill:
        """A whole-share market order. An auction order (opg or cls) that Alpaca won't
        take, e.g. after its cutoff, is sent again as a plain day order."""
        existing = self.by_client_id(client_id)
        if existing is not None:
            return existing
        body = {"symbol": symbol, "qty": str(int(qty)), "side": side, "type": "market",
                "time_in_force": tif, "client_order_id": client_id}
        r = self._req("POST", "/v2/orders", json=body)
        if r.status_code in (200, 201):
            return self._fill(r.json())
        note = f"{tif} refused: HTTP {r.status_code} {r.text[:200]}"
        if r.status_code in (403, 422) and tif in (OPEN_TIF, CLOSE_TIF):
            body.update(time_in_force=DAY_TIF, client_order_id=client_id + "-day")
            r2 = self._req("POST", "/v2/orders", json=body)
            if r2.status_code in (200, 201):
                return self._fill(r2.json(), note=note + "; sent as a day order")
            note += f"; day order refused: HTTP {r2.status_code} {r2.text[:200]}"
        return Fill(status="rejected", note=note)

    def refresh(self, client_id: str, broker_id: str | None = None, **_) -> Fill:
        f = self.by_client_id(client_id) or self.by_client_id(client_id + "-day")
        return f or Fill(status="unknown", note="order not found at Alpaca")


class DryRun:
    """Places nothing. Fills come from daily bars (`bars_for(ticker)`), once they exist."""
    kind = "dry-run"

    def __init__(self, bars_for, max_wait_sessions: int = 5, calendar=None):
        from ..features.calendar import NYSE
        self.bars_for = bars_for
        self.cal = calendar or NYSE
        self.max_wait = max_wait_sessions

    def tradable(self, symbol: str) -> bool:
        return True

    def submit(self, symbol: str, qty: int, side: str, tif: str, client_id: str,
               session: date) -> Fill:
        log.info("dry run: would %s %d %s (%s) for %s", side, qty, symbol, tif, session)
        return Fill(status="pending", tif=tif, note="dry run: logged, not sent")

    def position_qty(self, symbol: str) -> float | None:
        return None                       # the book's own quantity is the position

    def refresh(self, client_id: str, broker_id: str | None = None, *, symbol: str,
                session: date, tif: str, today: date, qty: float = 0, **_) -> Fill:
        bars = self.bars_for(symbol)
        bar = bars.at(session) if bars is not None else None
        if bar is None:
            if today > self.cal.add(session, self.max_wait):
                return Fill(status="expired", tif=tif, note="no bar for the order's session")
            return Fill(status="pending", tif=tif)
        if tif == CLOSE_TIF:
            price = bar.close
        else:
            price = bar.open
        if not price:
            return Fill(status="expired", tif=tif, note="no price in the session's bar")
        return Fill(status="filled", filled_qty=qty, filled_price=price,
                    filled_at=session.isoformat(), tif=tif,
                    note="dry run fill at the session's " + ("close" if tif == CLOSE_TIF else "open"))


def now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
