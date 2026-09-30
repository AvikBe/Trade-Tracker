"""Daily bars for the backtest, from a pluggable source.

The backtest needs more than the features do: opens (entry is at an open), raw closes
(to check a vendor's ticker is the company on the filing), volume (for the spread
estimate) and adjusted prices (for returns). Two sources:

- `DbPrices`: the `daily_prices` table, optionally limited to one `source`.
- `CachePrices`: a directory of `bars/<TICKER>.csv.gz` files in Tiingo's CSV layout
  (date,open,high,low,close,volume,adjOpen,adjClose, with raw open/close). The hourly
  Tiingo routine writes this layout, and `tt fetch-yahoo` writes the same one.

Both give a `Bars` object per ticker, or None when the source has no data for it.
"""

from __future__ import annotations

import bisect
import csv
import gzip
import io
import re
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path

import psycopg


@dataclass(frozen=True)
class Bar:
    date: date
    open: float | None       # raw, as traded that day
    close: float | None      # raw
    adj_open: float | None   # split and dividend adjusted
    adj_close: float | None
    volume: float | None     # raw shares


class Bars:
    """One ticker's bars, sorted by date, with lookups by date."""

    def __init__(self, bars: list[Bar]):
        self.bars = sorted(
            (b for b in bars if b.adj_close is not None and b.adj_close > 0),
            key=lambda b: b.date,
        )
        self.dates = [b.date for b in self.bars]

    def __len__(self) -> int:
        return len(self.bars)

    def index_on_or_after(self, d: date) -> int | None:
        i = bisect.bisect_left(self.dates, d)
        return i if i < len(self.dates) else None

    def index_on_or_before(self, d: date) -> int | None:
        i = bisect.bisect_right(self.dates, d) - 1
        return i if i >= 0 else None

    def index_before(self, d: date) -> int | None:
        i = bisect.bisect_left(self.dates, d) - 1
        return i if i >= 0 else None

    def at(self, d: date) -> Bar | None:
        i = bisect.bisect_left(self.dates, d)
        return self.bars[i] if i < len(self.dates) and self.dates[i] == d else None


def safe_name(ticker: str) -> str:
    """File name for a ticker; matches the Tiingo cache (`prices/cache.py`)."""
    return re.sub(r"[^A-Za-z0-9.\-]", "_", ticker)


def _f(v: str | None) -> float | None:
    if v is None or v == "" or v.lower() in {"nan", "null", "none"}:
        return None
    x = float(v)
    return x if x == x else None


def parse_csv(text: str) -> list[Bar]:
    out = []
    for row in csv.DictReader(io.StringIO(text)):
        adj_close = _f(row.get("adjClose"))
        close = _f(row.get("close"))
        open_ = _f(row.get("open"))
        adj_open = _f(row.get("adjOpen"))
        if adj_open is None and open_ is not None and close and adj_close:
            adj_open = open_ * adj_close / close
        out.append(
            Bar(
                date=date.fromisoformat(row["date"][:10]),
                open=open_,
                close=close,
                adj_open=adj_open,
                adj_close=adj_close,
                volume=_f(row.get("volume")),
            )
        )
    return out


def write_csv(path: Path, bars: list[Bar]) -> None:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["date", "open", "high", "low", "close", "volume", "adjOpen", "adjClose"])
    for b in bars:
        w.writerow([b.date.isoformat(), _s(b.open), "", "", _s(b.close), _s(b.volume),
                    _s(b.adj_open), _s(b.adj_close)])
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(gzip.compress(buf.getvalue().encode()))
    tmp.replace(path)


def _s(v: float | None) -> str:
    return "" if v is None else repr(round(v, 8))


class CachePrices:
    """Bars from a Tiingo-layout cache directory (`<root>/bars/<T>.csv.gz`)."""

    def __init__(self, root: Path | str, name: str = "cache"):
        self.root = Path(root)
        self.name = name
        self.bars_dir = self.root / "bars" if (self.root / "bars").is_dir() else self.root
        self.get = lru_cache(maxsize=None)(self._load)

    def _load(self, ticker: str) -> Bars | None:
        path = self.bars_dir / f"{safe_name(ticker)}.csv.gz"
        if not path.exists():
            return None
        bars = Bars(parse_csv(gzip.decompress(path.read_bytes()).decode()))
        return bars if len(bars) else None

    def tickers(self) -> set[str]:
        return {p.name[: -len(".csv.gz")] for p in self.bars_dir.glob("*.csv.gz")}


class DbPrices:
    """Bars from `daily_prices`, optionally only rows of one `source`."""

    def __init__(self, conn: psycopg.Connection, source: str | None = None):
        self.conn = conn
        self.source = source
        self.name = f"db:{source}" if source else "db"
        self.get = lru_cache(maxsize=None)(self._load)

    def _load(self, ticker: str) -> Bars | None:
        rows = self.conn.execute(
            """
            SELECT date, open, close, adj_open, adj_close, volume FROM daily_prices
            WHERE ticker = %s AND (%s::text IS NULL OR source = %s)
            """,
            (ticker, self.source, self.source),
        ).fetchall()
        bars = Bars(
            [
                Bar(d, _n(o), _n(c), _n(ao), _n(ac if ac is not None else c), _n(v))
                for d, o, c, ao, ac, v in rows
            ]
        )
        return bars if len(bars) else None


def _n(v) -> float | None:
    return None if v is None else float(v)


def open_source(spec: str, conn: psycopg.Connection | None = None):
    """'db', 'db:yahoo', 'dir:/path' (or a bare path) -> a price source."""
    if spec == "db" or spec.startswith("db:"):
        if conn is None:
            raise ValueError("a database price source needs a connection")
        return DbPrices(conn, spec[3:] or None)
    path = spec[4:] if spec.startswith("dir:") else spec
    if not Path(path).is_dir():
        raise ValueError(f"price cache directory not found: {path}")
    return CachePrices(path, name=Path(path).name)
