"""`tt paper-trade`: one morning's run of the paper book, and the inputs it needs."""

from __future__ import annotations

import csv
import json
import logging
import shutil
import time
from datetime import date, timedelta
from pathlib import Path

from .. import fundamentals
from ..edgar.client import EdgarClient
from ..edgar.tickers import fetch_company_tickers
from ..scoring import model as M
from . import engine, report
from .broker import AlpacaPaper, DryRun
from .prices import YahooPrices

log = logging.getLogger(__name__)

SHARES_MAX_AGE = timedelta(days=7)
RECENT_FRAMES = 3            # quarters of share counts re-fetched weekly (late filers fill in)


def refresh_shares(client, path: Path, seed: Path | None, today: date) -> fundamentals.SharesTable:
    """The share-count table, kept current: seeded from `seed` (the backtest's file),
    then the latest quarters' frames re-fetched once a week."""
    if not path.exists() and seed and seed.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(seed, path)          # keeps its date, so a stale seed is refreshed
    stale = not path.exists() or time.time() - path.stat().st_mtime > SHARES_MAX_AGE.total_seconds()
    if stale and client is not None:
        start = today - timedelta(days=92 * (RECENT_FRAMES - 1) + 1)
        frames = fundamentals.frame_names(start, today)
        names = {f"CY{y}Q{q}I" for y, q in frames}
        kept = []
        if path.exists():
            with path.open() as f:
                kept = [r for r in csv.DictReader(f) if r["frame"] not in names]
        tmp = path.with_suffix(".tmp")
        with tmp.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fundamentals.COLUMNS)
            w.writeheader()
            w.writerows(kept)
        n = fundamentals.fetch(client, tmp, start, today)
        tmp.replace(path)
        log.info("share counts: re-fetched %s", n)
    return fundamentals.SharesTable.load(path)


def load_tickers(client, path: Path, today: date) -> dict[str, str]:
    """SEC's CIK -> ticker map, fetched once a day: fills tickers missing on filings."""
    if path.exists() and date.fromtimestamp(path.stat().st_mtime) == today:
        return json.loads(path.read_text())
    try:
        t = fetch_company_tickers(client)
    except Exception as exc:  # a missing map only costs the few filings without a symbol
        log.warning("company_tickers.json: %s", exc)
        return json.loads(path.read_text()) if path.exists() else {}
    path.write_text(json.dumps(t))
    return t


def make_broker(kind: str, prices):
    if kind == "alpaca":
        return AlpacaPaper.from_env()
    return DryRun(prices.get)


def run(state_dir: Path, model_path: Path, broker_kind: str, user_agent: str,
        today: date | None = None, shares_seed: Path | None = None,
        filings_cache: Path | None = None) -> engine.RunResult:
    today = today or engine.eastern_today()
    m = M.Model.from_json(model_path.read_text())
    edgar = EdgarClient(user_agent)
    prices = YahooPrices(start=today - timedelta(days=600), end=today)
    broker = make_broker(broker_kind, prices)
    state_dir.mkdir(parents=True, exist_ok=True)
    shares = refresh_shares(edgar, state_dir / "shares-outstanding.csv", shares_seed, today)
    tickers = load_tickers(edgar, state_dir / "company-tickers.json", today)
    cfg = engine.Config(state_dir=state_dir, model=m)
    book = engine.Book(cfg, broker, prices, edgar=edgar, shares=shares, tickers=tickers,
                       filings_cache=filings_cache or state_dir / "filings")
    res = book.run(today)
    state = engine.load_state(book.state_path)
    try:
        equity = book.equity(state, today)
    except Exception as exc:
        log.warning("equity: %s", exc)
        equity = None
    (state_dir / "report.md").write_text(report.render(state, prices, today, equity))
    return res
