"""Shares outstanding from SEC XBRL frames, for a point-in-time market cap.

Every 10-K and 10-Q cover page tags `dei:EntityCommonStockSharesOutstanding` with the
date of the count. The frames API returns that fact for every filer per calendar
quarter in one call (`CY2020Q3I`), so about 50 calls cover 2014 to today.

Point in time: the count is dated shortly before the filing it appears in, but the
frames don't give the filing date. A count dated E is used only from E + `LAG_DAYS`
on, which is later than any 10-K or 10-Q deadline, so the backtest never sees a count
before it was public. Share counts move slowly, so the staleness costs little.

Market cap = the count x the last raw close before entry, with the count scaled by any
split between its date and entry (from the raw/adjusted close ratio), so a reverse
split after the count doesn't make a microcap look large. This needs raw closes next to
the adjusted ones (Yahoo and Tiingo both give them).
"""

from __future__ import annotations

import bisect
import csv
import logging
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

log = logging.getLogger(__name__)

FRAME_URL = ("https://data.sec.gov/api/xbrl/frames/dei/EntityCommonStockSharesOutstanding/"
             "shares/CY{year}Q{q}I.json")
LAG_DAYS = 100
COLUMNS = ["cik", "end", "shares", "accn", "frame"]


def frame_names(start: date, end: date) -> list[tuple[int, int]]:
    out = []
    y, q = start.year, (start.month - 1) // 3 + 1
    while (y, q) <= (end.year, (end.month - 1) // 3 + 1):
        out.append((y, q))
        y, q = (y + 1, 1) if q == 4 else (y, q + 1)
    return out


def parse_frame(payload: dict, frame: str) -> list[dict]:
    rows = []
    for d in payload.get("data", []):
        try:
            shares = float(d["val"])
        except (KeyError, TypeError, ValueError):
            continue
        if shares <= 0:
            continue
        rows.append({"cik": str(d["cik"]).lstrip("0"), "end": d["end"], "shares": shares,
                     "accn": d.get("accn", ""), "frame": frame})
    return rows


def fetch(client, out: Path, start: date, end: date) -> dict:
    """Fetch every quarter's frame not yet in `out` (a CSV cache, appended to)."""
    done = set()
    if out.exists():
        with out.open() as f:
            done = {r["frame"] for r in csv.DictReader(f)}
    new = not out.exists()
    counts = {"frames": 0, "facts": 0, "skipped": len(done)}
    with out.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        for y, q in frame_names(start, end):
            frame = f"CY{y}Q{q}I"
            if frame in done:
                continue
            try:
                resp = client.get(FRAME_URL.format(year=y, q=q))
            except Exception as exc:  # a quarter not published yet answers 404
                log.warning("frame %s: %s", frame, exc)
                continue
            rows = parse_frame(resp.json(), frame)
            w.writerows(rows)
            counts["frames"] += 1
            counts["facts"] += len(rows)
    return counts


class SharesTable:
    """Share counts per issuer CIK, looked up as they were known on a date."""

    def __init__(self, rows: list[tuple[str, date, float]], lag_days: int = LAG_DAYS):
        by: dict[str, dict[date, float]] = defaultdict(dict)
        for cik, end, shares in rows:
            by[cik.lstrip("0")][end] = shares   # the same fact appears in two frames
        self.lag = timedelta(days=lag_days)
        self.dates = {k: sorted(v) for k, v in by.items()}
        self.values = {k: [v[d] for d in self.dates[k]] for k, v in by.items()}

    def __len__(self) -> int:
        return len(self.dates)

    def as_of(self, cik: str | None, d: date) -> tuple[date, float] | None:
        """The latest count public on d: dated at least `lag_days` before it."""
        if not cik:
            return None
        k = cik.lstrip("0")
        ds = self.dates.get(k)
        if not ds:
            return None
        i = bisect.bisect_right(ds, d - self.lag) - 1
        return (ds[i], self.values[k][i]) if i >= 0 else None

    @classmethod
    def load(cls, path: Path, lag_days: int = LAG_DAYS) -> SharesTable:
        rows = []
        with Path(path).open() as f:
            for r in csv.DictReader(f):
                rows.append((r["cik"], date.fromisoformat(r["end"]), float(r["shares"])))
        return cls(rows, lag_days)


def market_cap(table: SharesTable | None, cik: str | None, bars, entry: date) -> float | None:
    """Shares known at entry x the last raw close before entry, split-adjusted."""
    if table is None or bars is None:
        return None
    known = table.as_of(cik, entry)
    if known is None:
        return None
    count_day, shares = known
    i = bars.index_before(entry)
    if i is None:
        return None
    # A count dated before the first bar (a listing after the count) is taken at the
    # first bar: no split between the two can be seen.
    j = bars.index_on_or_before(count_day)
    j = 0 if j is None else j
    last, then = bars.bars[i], bars.bars[j]
    if not (last.close and last.adj_close and then.close and then.adj_close):
        return None
    # raw/adjusted falls by the split ratio across a forward split (and rises across a
    # reverse split); dividends move it by a few percent, which doesn't matter here.
    split = (then.close / then.adj_close) / (last.close / last.adj_close)
    return shares * split * last.close
