"""A file-based Tiingo price cache that an hourly job fills across many sessions.

Cloud sessions are ephemeral and have no shared Postgres, so the backfill state lives in
a directory (the project's shared folder in practice), and `tt import-prices` copies it
into `daily_prices` wherever a database is. Layout:

    universe.csv        quarter,ticker,buys,trades  (from `tt price-universe`)
    status.json         per ticker: status, attempts, rows, first/last bar date
    calls.csv           one line per Tiingo request: time, ticker, outcome
    state.json          cooldown after Tiingo refused a call
    bars/<T>.csv.gz     Tiingo CSV, 2014-01-01 onward
    coverage.txt        summary written after every run

Each ticker is fetched once for its full history, so one call covers every quarter the
ticker appears in. The job spends at most what is left of the hourly, daily and monthly
unique-symbol quotas, counted from calls.csv with a safety margin, and stops at the
first refusal. Statuses: ok, empty (Tiingo knows the symbol but has no bars since
2014), not_found, error (retried up to MAX_ATTEMPTS times, an hour apart), failed.
"""

from __future__ import annotations

import csv
import gzip
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import httpx

from .loader import BENCHMARKS, HISTORY_START
from .tiingo import AuthError, RateLimited, TickerNotFound

# Headroom under the free plan (50/hour, 1,000/day, 500 symbols/month) for manual calls.
HOURLY_BUDGET = 45
DAILY_BUDGET = 950
MONTHLY_SYMBOL_BUDGET = 480
MAX_ATTEMPTS = 3
RETRY_AFTER = timedelta(hours=1)
LOCK_STALE = timedelta(minutes=50)
COOLDOWN = {
    "hourly": timedelta(minutes=55),
    "daily": timedelta(hours=6),
    "symbols": timedelta(hours=24),
}
DONE = {"ok", "empty", "not_found", "failed"}
BENCHMARK_QUARTER = "benchmark"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _safe(ticker: str) -> str:
    return re.sub(r"[^A-Za-z0-9.\-]", "_", ticker)


def _write_json(path: Path, data) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, sort_keys=True))
    os.replace(tmp, path)


@dataclass
class Call:
    at: datetime
    ticker: str
    outcome: str


@dataclass
class Budget:
    hourly: int
    daily: int
    symbols: int  # new symbols still allowed this calendar month (UTC)
    used_symbols: set[str] = field(default_factory=set)

    def allows(self, ticker: str) -> bool:
        if self.hourly <= 0 or self.daily <= 0:
            return False
        return ticker in self.used_symbols or self.symbols > 0

    def spend(self, ticker: str) -> None:
        self.hourly -= 1
        self.daily -= 1
        if ticker not in self.used_symbols:
            self.used_symbols.add(ticker)
            self.symbols -= 1


class Busy(RuntimeError):
    pass


class PriceCache:
    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.bars_dir = self.root / "bars"

    # ---- files -------------------------------------------------------------------
    def ensure(self) -> None:
        self.bars_dir.mkdir(parents=True, exist_ok=True)

    @property
    def universe_path(self) -> Path:
        return self.root / "universe.csv"

    def universe(self) -> list[dict]:
        if not self.universe_path.exists():
            return []
        with self.universe_path.open(newline="") as f:
            return [
                {**r, "buys": int(r["buys"]), "trades": int(r["trades"])} for r in csv.DictReader(f)
            ]

    def write_universe(self, rows: list[dict]) -> None:
        self.ensure()
        tmp = self.universe_path.with_suffix(".tmp")
        with tmp.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["quarter", "ticker", "buys", "trades"])
            w.writeheader()
            w.writerows(rows)
        os.replace(tmp, self.universe_path)

    def status(self) -> dict[str, dict]:
        p = self.root / "status.json"
        return json.loads(p.read_text()) if p.exists() else {}

    def save_status(self, status: dict[str, dict]) -> None:
        _write_json(self.root / "status.json", status)

    def state(self) -> dict:
        p = self.root / "state.json"
        return json.loads(p.read_text()) if p.exists() else {}

    def save_state(self, state: dict) -> None:
        _write_json(self.root / "state.json", state)

    def calls(self) -> list[Call]:
        p = self.root / "calls.csv"
        if not p.exists():
            return []
        out = []
        for line in p.read_text().splitlines():
            parts = line.split(",")
            if len(parts) == 3:
                out.append(Call(datetime.fromisoformat(parts[0]), parts[1], parts[2]))
        return out

    def log_call(self, at: datetime, ticker: str, outcome: str) -> None:
        with (self.root / "calls.csv").open("a") as f:
            f.write(f"{at.isoformat()},{ticker},{outcome}\n")
            f.flush()
            os.fsync(f.fileno())

    def bars_path(self, ticker: str) -> Path:
        return self.bars_dir / f"{_safe(ticker)}.csv.gz"

    def write_bars(self, ticker: str, text: str) -> None:
        path = self.bars_path(ticker)
        tmp = path.with_suffix(".tmp")
        with gzip.open(tmp, "wt") as f:
            f.write(text)
        os.replace(tmp, path)

    def read_bars(self, ticker: str) -> str:
        with gzip.open(self.bars_path(ticker), "rt") as f:
            return f.read()

    # ---- lock --------------------------------------------------------------------
    def acquire(self, now: datetime) -> None:
        lock = self.root / "LOCK"
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                held_since = datetime.fromisoformat(lock.read_text().strip())
            except (OSError, ValueError):
                held_since = now - LOCK_STALE
            if now - held_since < LOCK_STALE:
                raise Busy(f"another run holds {lock} since {held_since.isoformat()}")
            lock.unlink(missing_ok=True)
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, now.isoformat().encode())
        os.close(fd)

    def release(self) -> None:
        (self.root / "LOCK").unlink(missing_ok=True)


# ---- planning ----------------------------------------------------------------------
def budget(calls: list[Call], now: datetime) -> Budget:
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    used = {c.ticker for c in calls if c.at >= month_start}
    return Budget(
        hourly=HOURLY_BUDGET - sum(c.at > now - timedelta(hours=1) for c in calls),
        daily=DAILY_BUDGET - sum(c.at > now - timedelta(days=1) for c in calls),
        symbols=MONTHLY_SYMBOL_BUDGET - len(used),
        used_symbols=used,
    )


def queue(universe: list[dict], status: dict[str, dict], order: list[str], now: datetime) -> list[str]:
    """Tickers still to fetch, in priority order.

    Benchmarks first, then quarters in `order`, and within a quarter by buy count, then
    trade count, since buys are what the tracker ranks. Each ticker appears once, under
    its highest-priority quarter. Errors wait RETRY_AFTER before another attempt.
    """
    rank = {q: i for i, q in enumerate(order)}
    rows = [r for r in universe if r["quarter"] in rank]
    rows.sort(key=lambda r: (rank[r["quarter"]], -r["buys"], -r["trades"], r["ticker"]))
    out, seen = [], set()
    for t in [*BENCHMARKS, *(r["ticker"] for r in rows)]:
        if t in seen:
            continue
        seen.add(t)
        s = status.get(t)
        if s and s["status"] in DONE:
            continue
        if s and s["status"] == "error":
            if now - datetime.fromisoformat(s["last_attempt"]) < RETRY_AFTER:
                continue
        out.append(t)
    return out


# ---- the job -----------------------------------------------------------------------
def _bar_range(text: str) -> tuple[int, str | None, str | None]:
    lines = text.strip().splitlines()[1:]
    if not lines:
        return 0, None, None
    return len(lines), lines[0].split(",", 1)[0], lines[-1].split(",", 1)[0]


def run(cache: PriceCache, client, order: list[str], *, now=utcnow, limit: int | None = None) -> dict:
    """Fetch as many queued tickers as the quotas allow. Safe to run any time."""
    cache.ensure()
    started = now()
    stats = Counter()
    stats["stopped"] = ""
    stats["requested"] = 0
    state = cache.state()
    until = state.get("cooldown_until")
    if until and datetime.fromisoformat(until) > started:
        stats["stopped"] = f"cooling down until {until} ({state.get('cooldown_reason')})"
        return dict(stats)
    try:
        cache.acquire(started)
    except Busy as e:
        stats["stopped"] = str(e)
        return dict(stats)
    try:
        status = cache.status()
        b = budget(cache.calls(), started)
        todo = queue(cache.universe(), status, order, started)
        for ticker in todo:
            if limit is not None and stats["requested"] >= limit:
                stats["stopped"] = "limit"
                break
            if not b.allows(ticker):
                stats["stopped"] = (
                    "monthly symbol budget spent" if b.hourly > 0 and b.daily > 0 else "hourly/daily budget spent"
                )
                break
            at = now()
            prev = status.get(ticker, {})
            entry = {"attempts": prev.get("attempts", 0) + 1, "last_attempt": at.isoformat()}
            try:
                text = client.daily_csv(ticker, HISTORY_START)
            except RateLimited as e:
                cache.log_call(at, ticker, f"limited_{e.kind}")
                state = {
                    "cooldown_until": (at + COOLDOWN[e.kind]).isoformat(),
                    "cooldown_reason": str(e)[:300],
                }
                cache.save_state(state)
                stats["stopped"] = f"Tiingo {e.kind} limit: {str(e)[:200]}"
                break
            except AuthError as e:
                cache.log_call(at, ticker, "auth")
                stats["stopped"] = f"Tiingo rejected the API key: {e}"
                break
            except TickerNotFound:
                entry["status"] = "not_found"
            except (httpx.HTTPError, OSError) as e:
                entry["status"] = "error" if entry["attempts"] < MAX_ATTEMPTS else "failed"
                entry["detail"] = str(e)[:300]
            else:
                rows, first, last = _bar_range(text)
                if rows:
                    cache.write_bars(ticker, text)
                    entry.update(status="ok", rows=rows, first=first, last=last)
                else:
                    entry["status"] = "empty"
            b.spend(ticker)
            stats["requested"] += 1
            stats[entry["status"]] += 1
            cache.log_call(at, ticker, entry["status"])
            status[ticker] = entry
            cache.save_status(status)
        else:
            stats["stopped"] = stats["stopped"] or "queue empty"
        stats["queued"] = len(queue(cache.universe(), status, order, now()))
        stats["symbols_left_this_month"] = max(0, b.symbols)
    finally:
        cache.release()
    (cache.root / "coverage.txt").write_text(coverage_report(cache, order, now()))
    return dict(stats)


# ---- reporting ---------------------------------------------------------------------
def _quarter_bounds(q: str) -> tuple[str, str]:
    y, n = int(q[:4]), int(q[-1])
    start = date(y, 3 * n - 2, 1)
    end = date(y + (n == 4), 1 if n == 4 else 3 * n + 1, 1) - timedelta(days=1)
    return start.isoformat(), end.isoformat()


def coverage(cache: PriceCache, order: list[str]) -> list[dict]:
    """Per quarter: tickers and trades whose ticker has bars spanning that quarter."""
    status = cache.status()
    by_q: dict[str, list[dict]] = defaultdict(list)
    for r in cache.universe():
        by_q[r["quarter"]].append(r)
    out = []
    for q in order:
        rows = by_q.get(q, [])
        if not rows:
            continue
        qs, qe = _quarter_bounds(q)
        c = Counter()
        for r in rows:
            s = status.get(r["ticker"], {})
            st = s.get("status", "pending")
            c[st] += 1
            if st == "ok" and s["first"] <= qe and s["last"] >= qs:
                c["covered"] += 1
                c["buys_covered"] += r["buys"]
                c["trades_covered"] += r["trades"]
            elif st == "ok":
                c["out_of_range"] += 1
        out.append(
            {
                "quarter": q,
                "tickers": len(rows),
                "covered": c["covered"],
                "buys": sum(r["buys"] for r in rows),
                "buys_covered": c["buys_covered"],
                "trades": sum(r["trades"] for r in rows),
                "trades_covered": c["trades_covered"],
                "out_of_range": c["out_of_range"],
                "not_found": c["not_found"] + c["empty"],
                "failed": c["failed"] + c["error"],
                "pending": c["pending"],
            }
        )
    return out


def coverage_report(cache: PriceCache, order: list[str], now: datetime) -> str:
    status = cache.status()
    counts = Counter(s["status"] for s in status.values())
    universe = {r["ticker"] for r in cache.universe()} | set(BENCHMARKS)
    b = budget(cache.calls(), now)
    state = cache.state()
    lines = [
        f"Price cache coverage at {now.isoformat(timespec='minutes')}",
        f"distinct tickers: {len(universe)}; fetched: {sum(counts.values())} "
        + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())),
        f"budget left: {max(0, b.hourly)} this hour, {max(0, b.daily)} today, "
        f"{max(0, b.symbols)} new symbols this month",
    ]
    if state.get("cooldown_until") and datetime.fromisoformat(state["cooldown_until"]) > now:
        lines.append(f"cooling down until {state['cooldown_until']}: {state.get('cooldown_reason')}")
    lines += [
        "",
        "Covered = the ticker has Tiingo bars spanning the quarter.",
        f"{'quarter':>8} {'tickers':>8} {'covered':>8} {'buys':>7} {'buys cov':>9} "
        f"{'trades':>7} {'trades cov':>11} {'no data':>8} {'range':>6} {'failed':>7}",
    ]
    for c in coverage(cache, order):
        lines.append(
            f"{c['quarter']:>8} {c['tickers']:>8} {c['covered']:>8} {c['buys']:>7} "
            f"{c['buys_covered']:>9} {c['trades']:>7} {c['trades_covered']:>11} "
            f"{c['not_found']:>8} {c['out_of_range']:>6} {c['failed']:>7}"
        )
    return "\n".join(lines) + "\n"


# ---- database bridges --------------------------------------------------------------
PLAUSIBLE_TICKER = r"^[A-Z][A-Z0-9]{0,5}([.-][A-Z0-9]{1,2})?$"


def build_universe(conn) -> list[dict]:
    """(filing quarter, ticker) pairs with buy and trade counts, from the trades table.

    Amendment repeats are left in: they are few, and they never add a ticker.
    """
    rows = conn.execute(
        """
        SELECT to_char(f.filed_at AT TIME ZONE 'America/New_York', 'YYYY"q"Q') AS quarter,
               t.ticker,
               count(*) FILTER (WHERE t.side = 'buy') AS buys,
               count(*) AS trades
        FROM trades t JOIN filings f USING (filing_id)
        WHERE t.ticker ~ %s
        GROUP BY 1, 2
        ORDER BY 1, 2
        """,
        (PLAUSIBLE_TICKER,),
    ).fetchall()
    return [{"quarter": q, "ticker": t, "buys": b, "trades": n} for q, t, b, n in rows]


def import_to_db(cache: PriceCache, conn) -> dict:
    """Upsert every cached ticker into daily_prices and price_status. Idempotent."""
    status = cache.status()
    stats = Counter()
    with conn.cursor() as cur:
        cur.execute(
            "CREATE TEMP TABLE IF NOT EXISTS price_import (LIKE daily_prices INCLUDING DEFAULTS) "
            "ON COMMIT DELETE ROWS"
        )
        for ticker, s in sorted(status.items()):
            if s["status"] == "ok":
                text = cache.read_bars(ticker)
                lines = text.strip().splitlines()
                cols = lines[0].split(",")
                idx = {c: i for i, c in enumerate(cols)}
                with cur.copy(
                    "COPY price_import (ticker, date, open, high, low, close, adj_open, "
                    "adj_close, volume) FROM STDIN"
                ) as cp:
                    for ln in lines[1:]:
                        v = ln.split(",")

                        def g(c):
                            x = v[idx[c]] if c in idx and idx[c] < len(v) else ""
                            return x or None

                        vol = g("volume")
                        cp.write_row(
                            (ticker, g("date"), g("open"), g("high"), g("low"), g("close"),
                             g("adjOpen"), g("adjClose"),
                             None if vol is None else int(float(vol)))
                        )
                cur.execute(
                    """
                    INSERT INTO daily_prices (ticker, date, open, high, low, close,
                                              adj_open, adj_close, volume, source)
                    SELECT ticker, date, open, high, low, close, adj_open, adj_close, volume,
                           'tiingo' FROM price_import
                    ON CONFLICT (ticker, date) DO UPDATE SET
                      open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low,
                      close = EXCLUDED.close, adj_open = EXCLUDED.adj_open,
                      adj_close = EXCLUDED.adj_close, volume = EXCLUDED.volume
                    """
                )
                stats["bars"] += cur.rowcount
                stats["tickers"] += 1
                cur.execute("TRUNCATE price_import")
            if s["status"] in DONE | {"error"}:
                db_status = {"empty": "not_found", "failed": "error"}.get(s["status"], s["status"])
                cur.execute(
                    """
                    INSERT INTO price_status (ticker, status, detail, last_attempt)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (ticker) DO UPDATE SET status = EXCLUDED.status,
                      detail = EXCLUDED.detail, last_attempt = EXCLUDED.last_attempt
                    """,
                    (ticker, db_status, s.get("detail"), s["last_attempt"]),
                )
                stats[db_status] += 1
    conn.commit()
    return dict(stats)
