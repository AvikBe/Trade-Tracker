"""The paper book: one run per trading morning, before the open.

Each run, in order:

1. Reconcile: ask the broker what happened to earlier orders. A filled buy opens a
   position, a filled sell closes it, and an order that died (canceled, expired,
   rejected) frees its slot or, for a sell, leaves the position to be sold again.
2. Ingest every EDGAR daily index published since the last run (normally yesterday's).
3. Score each new buy as the backtest did at entry (`paper.signals`) and keep the ones
   at or above the model's cutoff, one per stock.
4. Exits: a position bought at session s is sold at the close of session s + h - 1
   (held h sessions, the backtest's exit). A missed exit is sold at the next open.
5. Entries: the backtest's top-20 portfolio. Up to `slots` positions, each 1/slots of
   equity, no stock twice, best score first, bought at today's open. A position that
   sells today still holds its slot today, as in `scoring.study.simulate`.
6. Save state and logs, and write `report.md`.

Everything lives in one directory (`state.json`, `decisions.csv`, `orders.csv`,
`runs.log`, `report.md`), so any session can pick the book up. A rerun on the same
morning is safe: client order ids and the held-ticker check stop duplicate orders, and
nothing is saved until the run ends.
"""

from __future__ import annotations

import csv
import fcntl
import json
import logging
import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from ..backtest.prices import Bars
from ..edgar.common import EASTERN
from ..features.calendar import NYSE, SEC
from ..scoring import model as M
from . import ingest
from .broker import CLOSE_TIF, DAY_TIF, FINAL, OPEN_TIF, Fill
from .signals import Decision, best_per_ticker, evaluate

log = logging.getLogger(__name__)

STATE_VERSION = 1
SLOTS = 20
HORIZON = 60
MIN_CAP = 50e6
MAX_LATE_SESSIONS = 5        # an entry may come at most this many sessions after plan
START_CASH = 100_000.0
MAX_INGEST_DAYS = 31         # calendar days of indexes one run will catch up on (older buys are too late anyway)

DECISION_COLUMNS = ["run_session", "filing_date", "planned_entry", "entry", "ticker", "filer",
                    "filer_name", "role", "value", "status", "score", "drawdown", "drift",
                    "market_cap", "dollar_volume", "last_close", "last_close_date", "cost",
                    "is_10b5_1", "accessions", "action"]
ORDER_COLUMNS = ["logged_at", "session", "client_id", "ticker", "side", "qty", "tif",
                 "status", "filled_qty", "filled_price", "filled_at", "broker_id", "note"]


@dataclass
class Config:
    state_dir: Path
    model: M.Model
    slots: int = SLOTS
    horizon: int = HORIZON
    min_cap: float | None = MIN_CAP
    start_cash: float = START_CASH
    max_late: int = MAX_LATE_SESSIONS


@dataclass
class RunResult:
    session: date | None
    ingested: list[str] = field(default_factory=list)
    waiting_for: str | None = None
    decisions: list[Decision] = field(default_factory=list)
    candidates: list[Decision] = field(default_factory=list)
    orders: list[dict] = field(default_factory=list)       # every order change, for orders.csv
    submitted: list[dict] = field(default_factory=list)    # orders placed this run
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------- state

def new_state(broker_kind: str, start_cash: float, first_index_day: date) -> dict:
    return {"version": STATE_VERSION, "broker": broker_kind, "start_cash": start_cash,
            "created": _now(), "last_index_day": first_index_day.isoformat(),
            "positions": [], "runs": []}


def load_state(path: Path) -> dict | None:
    if not path.exists():
        return None
    s = json.loads(path.read_text())
    if s.get("version") != STATE_VERSION:
        raise ValueError(f"{path}: state version {s.get('version')} != {STATE_VERSION}")
    return s


def save_state(path: Path, state: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1, sort_keys=True, default=str))
    tmp.replace(path)


@contextmanager
def locked(state_dir: Path):
    state_dir.mkdir(parents=True, exist_ok=True)
    with (state_dir / ".lock").open("w") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"another paper-trade run holds {state_dir}/.lock") from None
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _append_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerows(rows)


def live_positions(state: dict) -> list[dict]:
    """Positions holding a slot: waiting to be bought, held, or waiting to be sold."""
    return [p for p in state["positions"] if p["status"] in ("pending_entry", "open", "pending_exit")]


def exit_due(entry_session: date, horizon: int) -> date:
    return NYSE.add(entry_session, horizon - 1)


# --------------------------------------------------------------------- the run

class Book:
    def __init__(self, cfg: Config, broker, prices, edgar=None, shares=None,
                 tickers: dict[str, str] | None = None, filings_cache: Path | None = None):
        """`prices.get(ticker) -> Bars | None` (Yahoo in production); `edgar` an
        `EdgarClient` for the daily indexes (cached per day in `filings_cache`)."""
        self.cfg, self.broker, self.prices = cfg, broker, prices
        self.edgar, self.shares, self.tickers = edgar, shares, tickers or {}
        self.filings_cache = filings_cache
        self.state_path = cfg.state_dir / "state.json"

    # ----------------------------------------------------------- reconcile

    def _refresh(self, p: dict, leg: str, today: date) -> Fill:
        o = p[leg]
        return self.broker.refresh(o["client_id"], o.get("broker_id"), symbol=p["ticker"],
                                   session=date.fromisoformat(o["session"]), tif=o["tif"],
                                   today=today, qty=o["qty"])

    def reconcile(self, state: dict, today: date, res: RunResult) -> None:
        for p in state["positions"]:
            if p["status"] == "pending_entry":
                f = self._refresh(p, "entry_order", today)
                self._record(p, p["entry_order"], f, res)
                if f.status == "filled" or (f.status in FINAL and f.filled_qty > 0):
                    s = date.fromisoformat((f.filled_at or p["entry_order"]["session"])[:10])
                    s = NYSE.on_or_before(s)
                    p.update(status="open", entry_session=s.isoformat(), entry_qty=f.filled_qty,
                             entry_price=f.filled_price,
                             exit_due=exit_due(s, self.cfg.horizon).isoformat())
                    bars = self.prices.get(p["ticker"])
                    b = bars.at(s) if bars is not None else None
                    p["entry_adj_open"] = b.adj_open if b else None
                    if f.status != "filled":
                        res.notes.append(f"{p['ticker']}: buy only partly filled "
                                         f"({f.filled_qty:g} of {p['entry_order']['qty']})")
                elif f.status in FINAL or f.status == "unknown":
                    p.update(status="not_filled", close_reason=f"buy {f.status}: {f.note}".strip())
                    res.notes.append(f"{p['ticker']}: buy {f.status}, slot freed")
            elif p["status"] == "pending_exit":
                f = self._refresh(p, "exit_order", today)
                self._record(p, p["exit_order"], f, res)
                if f.status == "filled":
                    s = NYSE.on_or_before(date.fromisoformat((f.filled_at or p["exit_order"]["session"])[:10]))
                    p.update(status="closed", exit_session=s.isoformat(), exit_qty=f.filled_qty,
                             exit_price=f.filled_price)
                    self._settle(p)
                elif f.status in FINAL or f.status == "unknown":
                    # Sold nothing (or part): keep the position and sell the rest next run.
                    if f.filled_qty:
                        p["entry_qty_sold"] = p.get("entry_qty_sold", 0) + f.filled_qty
                    p["status"] = "open"
                    p.setdefault("exit_attempts", []).append(p.pop("exit_order"))
                    res.notes.append(f"{p['ticker']}: sell {f.status} ({f.note}); will retry")

    def _record(self, p: dict, order: dict, f: Fill, res: RunResult) -> None:
        changed = f.status != order.get("status")
        order.update(status=f.status, filled_qty=f.filled_qty, filled_price=f.filled_price,
                     filled_at=f.filled_at, broker_id=f.broker_id or order.get("broker_id"))
        if f.tif:
            order["tif"] = f.tif
        if f.note:
            order["note"] = f.note
        if changed:
            res.orders.append({**order, "ticker": p["ticker"], "logged_at": _now()})

    def _settle(self, p: dict) -> None:
        """Returns of a closed position: gross, SPY over the same sessions, and excess."""
        ret = None
        if self.broker.kind == "dry-run":
            bars = self.prices.get(p["ticker"])
            b = bars.at(date.fromisoformat(p["exit_session"])) if bars is not None else None
            if b and p.get("entry_adj_open"):
                ret = b.adj_close / p["entry_adj_open"] - 1
        elif p.get("entry_price") and p.get("exit_price") and p.get("entry_qty"):
            ret = (p["exit_price"] * p["exit_qty"]) / (p["entry_price"] * p["entry_qty"]) - 1
        p["ret"] = ret
        p["spy_ret"] = spy_return(self.prices, date.fromisoformat(p["entry_session"]),
                                  date.fromisoformat(p["exit_session"]))
        p["excess"] = None if ret is None or p["spy_ret"] is None else ret - p["spy_ret"]
        # A dry run fills at the open and close with no spread, so charge the backtest's
        # cost estimate. Alpaca's simulated fills are at the quote, so they already pay it.
        cost = (p.get("cost") or 0.0) if self.broker.kind == "dry-run" else 0.0
        p["excess_net"] = None if p["excess"] is None else p["excess"] - cost

    # ----------------------------------------------------------- ingest

    def ingest(self, state: dict, today: date, res: RunResult) -> list[ingest.LiveEvent]:
        """Every daily index after `last_index_day` and before today that is out."""
        d = date.fromisoformat(state["last_index_day"]) + timedelta(days=1)
        stop = min(today, d + timedelta(days=MAX_INGEST_DAYS))
        events: list[ingest.LiveEvent] = []
        while d < stop:
            if not SEC.is_business_day(d):
                state["last_index_day"] = d.isoformat()
                d += timedelta(days=1)
                continue
            data = ingest.fetch_day(self.edgar, d, self.filings_cache)
            if data is None:
                res.waiting_for = d.isoformat()
                res.notes.append(f"EDGAR's index for {d} isn't out yet; will retry next run")
                break
            ex: dict = {}
            evs = ingest.build_events(data["filings"], self.tickers, ex)
            res.ingested.append(f"{d}: {data['counts']['index_rows']} Form 4s, "
                                f"{len(evs)} buy events" + (f", {data['counts']['errors']} unreadable"
                                                            if data["counts"]["errors"] else ""))
            events += evs
            state["last_index_day"] = d.isoformat()
            d += timedelta(days=1)
        return events

    # ----------------------------------------------------------- decide and trade

    def decide(self, events: list[ingest.LiveEvent], session: date, res: RunResult) -> list[Decision]:
        out = []
        for le in events:
            planned = NYSE.add(le.event.filing_date, 1)
            late = NYSE.business_days_between(planned, session)
            if late > self.cfg.max_late:
                d = Decision(le, session, "too_late")
            else:
                d = evaluate(le, self.prices.get(le.event.ticker), session, self.cfg.model,
                             self.shares, self.cfg.min_cap)
                if late > 0:
                    d.flags.append(f"late_entry_{late}")
            out.append(d)
        res.decisions = out
        return out

    def equity(self, state: dict, session: date) -> float:
        if self.broker.kind != "dry-run":
            return self.broker.equity()
        eq = state["start_cash"]
        for p in state["positions"]:
            if p["status"] == "closed" and p.get("ret") is not None:
                eq += p["entry_price"] * p["entry_qty"] * p["ret"]
            elif p["status"] in ("open", "pending_exit"):
                eq += p["entry_price"] * p["entry_qty"] * (mark(self.prices, p, session) or 0.0)
        return eq

    def _order(self, p: dict, leg: str, side: str, qty: int, tif: str, session: date,
               res: RunResult) -> None:
        cid = f"tt-{session.isoformat()}-{p['ticker']}-{side}"
        f = self.broker.submit(p["ticker"], qty, side, tif, cid, session)
        p[leg] = {"client_id": cid, "session": session.isoformat(), "side": side, "qty": qty,
                  "tif": f.tif or tif, "submitted_at": _now(), "status": f.status,
                  "broker_id": f.broker_id, "note": f.note, "filled_qty": f.filled_qty,
                  "filled_price": f.filled_price, "filled_at": f.filled_at}
        res.orders.append({**p[leg], "ticker": p["ticker"], "logged_at": _now()})
        res.submitted.append({**p[leg], "ticker": p["ticker"]})

    def exits(self, state: dict, session: date, res: RunResult) -> None:
        for p in state["positions"]:
            if p["status"] != "open" or date.fromisoformat(p["exit_due"]) > session:
                continue
            on_time = date.fromisoformat(p["exit_due"]) == session
            qty = self.broker.position_qty(p["ticker"])
            if qty is None:
                qty = p["entry_qty"] - p.get("entry_qty_sold", 0)
            if qty <= 0:
                p.update(status="closed", exit_session=session.isoformat(), exit_qty=0,
                         exit_price=None, close_reason="no shares left at the broker")
                self._settle(p)
                continue
            self._order(p, "exit_order", "sell", int(qty), CLOSE_TIF if on_time else DAY_TIF,
                        session, res)
            if not on_time:
                res.notes.append(f"{p['ticker']}: exit was due {p['exit_due']}, selling at today's open")
            p["status"] = "pending_exit" if p["exit_order"]["status"] != "rejected" else "open"

    def entries(self, state: dict, session: date, cands: list[Decision], res: RunResult) -> None:
        held = live_positions(state)
        free = self.cfg.slots - len(held)
        tickers = {p["ticker"] for p in held}
        actions: dict[int, str] = {}
        if not cands:
            return
        size = self.equity(state, session) / self.cfg.slots
        for d in cands:
            if d.ticker in tickers:
                actions[id(d)] = "already held"
                continue
            if free <= 0:
                actions[id(d)] = "no free slot"
                continue
            qty = math.floor(size / d.last_close) if d.last_close else 0
            if qty < 1:
                actions[id(d)] = "price above one slot"
                continue
            if not self.broker.tradable(d.ticker):
                actions[id(d)] = "not tradable at the broker"
                continue
            e, le, s = d.live.event, d.live, d.signals
            p = {"id": f"{session.isoformat()}-{d.ticker}", "ticker": d.ticker,
                 "issuer_cik": e.issuer_cik, "issuer_name": le.issuer_name,
                 "filer": le.filer_cik, "filer_name": le.filer_name, "role": e.role,
                 "value": e.value, "filing_date": e.filing_date.isoformat(),
                 "accessions": le.accessions, "score": d.score, "drawdown": s.drawdown,
                 "drift": s.drift, "market_cap": s.market_cap, "dollar_volume": s.dollar_volume,
                 "cost": d.cost, "signal_close": d.last_close, "flags": d.flags,
                 "planned_entry": NYSE.add(e.filing_date, 1).isoformat(), "status": "pending_entry"}
            self._order(p, "entry_order", "buy", qty, OPEN_TIF, session, res)
            if p["entry_order"]["status"] == "rejected":
                actions[id(d)] = f"order rejected: {p['entry_order']['note']}"
                p.update(status="not_filled", close_reason="buy rejected")
            else:
                actions[id(d)] = f"buy {qty}"
                free -= 1
                tickers.add(d.ticker)
            state["positions"].append(p)
        self._actions = actions

    def run(self, today: date, state: dict | None = None) -> RunResult:
        with locked(self.cfg.state_dir):
            if state is None:
                state = load_state(self.state_path) or new_state(
                    self.broker.kind, self.cfg.start_cash, SEC_prev(today))
            session = today if NYSE.is_business_day(today) else None
            res = RunResult(session)
            self._actions = {}
            self.reconcile(state, today, res)
            if session is not None:
                events = self.ingest(state, today, res)
                decisions = self.decide(events, session, res)
                res.candidates = best_per_ticker(decisions)
                self.exits(state, session, res)
                self.entries(state, session, res.candidates, res)
            else:
                res.notes.append("not a trading day: new filings wait for the next session")
            self._log(state, today, res)
            save_state(self.state_path, state)
            return res

    # ----------------------------------------------------------- logs

    def _log(self, state: dict, today: date, res: RunResult) -> None:
        sd = self.cfg.state_dir
        rows = []
        cand_ids = {id(d) for d in res.candidates}
        for d in res.decisions:
            e, s = d.live.event, d.signals
            action = self._actions.get(id(d), "")
            if not action and d.status == "candidate" and id(d) not in cand_ids:
                action = "another insider's buy of this stock scored higher"
            rows.append({"run_session": res.session, "filing_date": e.filing_date,
                         "planned_entry": NYSE.add(e.filing_date, 1), "entry": d.entry,
                         "ticker": e.ticker, "filer": d.live.filer_cik,
                         "filer_name": d.live.filer_name, "role": e.role or "",
                         "value": round(e.value, 2), "status": d.status,
                         "score": _fmt(d.score), "drawdown": _fmt(s.drawdown if s else None),
                         "drift": _fmt(s.drift if s else None),
                         "market_cap": _fmt(s.market_cap if s else None),
                         "dollar_volume": _fmt(s.dollar_volume if s else None),
                         "last_close": _fmt(d.last_close), "last_close_date": d.last_close_date or "",
                         "cost": _fmt(d.cost), "is_10b5_1": int(e.is_10b5_1),
                         "accessions": " ".join(d.live.accessions), "action": action})
        if rows:
            _append_csv(sd / "decisions.csv", DECISION_COLUMNS, rows)
        if res.orders:
            _append_csv(sd / "orders.csv", ORDER_COLUMNS, res.orders)
        statuses: dict[str, int] = {}
        for d in res.decisions:
            statuses[d.status] = statuses.get(d.status, 0) + 1
        state["runs"] = (state["runs"] + [{
            "at": _now(), "today": today.isoformat(),
            "session": res.session.isoformat() if res.session else None,
            "ingested": res.ingested, "waiting_for": res.waiting_for,
            "decisions": statuses, "candidates": [d.ticker for d in res.candidates],
            "orders": [f"{o['side']} {o['qty']} {o['ticker']} {o['tif']}: {o['status']}"
                       for o in res.submitted],
            "notes": res.notes}])[-200:]
        with (sd / "runs.log").open("a") as f:
            f.write(f"=== {_now()} today={today} session={res.session} broker={self.broker.kind}\n")
            for line in res.ingested + res.notes:
                f.write(f"  {line}\n")
            f.write(f"  decisions: {statuses}; candidates: {[d.ticker for d in res.candidates]}\n")
            for o in res.orders:
                f.write(f"  order {o.get('side')} {o.get('qty')} {o.get('ticker', o.get('client_id'))} "
                        f"{o.get('tif')} -> {o.get('status')} {o.get('note') or ''}\n")


def SEC_prev(today: date) -> date:
    """The day before the first index a new book reads: so it starts with the latest
    business day before today."""
    d = today - timedelta(days=1)
    while not SEC.is_business_day(d):
        d -= timedelta(days=1)
    return d - timedelta(days=1)


def _fmt(v) -> str:
    return "" if v is None else f"{v:.6g}"


# --------------------------------------------------------------------- marks

def spy_return(prices, start: date, end: date) -> float | None:
    """SPY from the open of `start` to the close of `end` (or the last close before)."""
    spy = prices.get("SPY")
    if spy is None:
        return None
    a = spy.at(start)
    j = spy.index_on_or_before(end)
    if a is None or not a.adj_open or j is None or spy.dates[j] < start:
        return None
    return spy.bars[j].adj_close / a.adj_open - 1


def mark(prices, p: dict, as_of: date) -> float | None:
    """An open position's return from its entry open to the last close before `as_of`."""
    bars: Bars | None = prices.get(p["ticker"])
    if bars is None or not p.get("entry_adj_open"):
        if bars is not None and p.get("entry_price"):
            i = bars.index_before(as_of)
            if i is not None and bars.bars[i].close:
                return bars.bars[i].close / p["entry_price"] - 1
        return None
    i = bars.index_before(as_of)
    if i is None or bars.dates[i] < date.fromisoformat(p["entry_session"]):
        return None
    return bars.bars[i].adj_close / p["entry_adj_open"] - 1


def eastern_today() -> date:
    return datetime.now(EASTERN).date()
