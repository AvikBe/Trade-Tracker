"""Routine and opportunistic insiders (Cohen, Malloy and Pomorski, "Decoding Inside
Information", Journal of Finance 2012).

At the start of each calendar year Y, an insider who traded (open-market buys or sales,
any issuer) in each of the three previous years is classified:

- **routine** if some calendar month had a trade in all three of those years (the same
  month, three years running: an annual bonus purchase, a yearly diversification sale);
- **opportunistic** otherwise.

Everyone else (fewer than three years of trading) is unclassified. The paper finds that
opportunistic trades predict returns and routine ones don't.

Point in time: only trades *filed* before 1 January of Y count, so a December trade
filed in January doesn't classify its insider until the year after.

Insiders are keyed by the reporting owner's CIK (the Form 4's primary owner, which is
the database's `filers.source_key`). The history comes from the SEC data sets
(`tt insider-history`), which go back further than the loaded database, so the 2015
trades can be classified from 2012-2014 trading.
"""

from __future__ import annotations

import csv
import gzip
from collections import defaultdict
from datetime import date
from pathlib import Path

ROUTINE = "routine"
OPPORTUNISTIC = "opportunistic"
LOOKBACK_YEARS = 3
COLUMNS = ["owner_cik", "issuer_cik", "trade_date", "filed", "side"]


def history_rows(filings) -> list[dict]:
    """One row per open-market trade line, from parsed filings (amendments included:
    a repeated month changes nothing)."""
    out = []
    for f in filings:
        for t in f.trades:
            out.append({"owner_cik": f.filer_key, "issuer_cik": f.issuer_cik or "",
                        "trade_date": t.trade_date.isoformat(),
                        "filed": f.filed_at.date().isoformat(), "side": t.side})
    return out


def write_history(rows, path: Path) -> None:
    with gzip.open(path, "wt", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)


class Classifier:
    """`kind(owner_cik, d)`: the owner's type for a trade on date d (its year's type)."""

    def __init__(self, rows):
        # owner -> list of (filed, trade year, trade month), deduplicated
        months: dict[str, set] = defaultdict(set)
        for owner, trade_date, filed in rows:
            months[owner.lstrip("0")].add((filed, trade_date.year, trade_date.month))
        self.months = months
        self.first_filed = min((f for v in months.values() for f, _, _ in v), default=None)
        self._cache: dict[tuple[str, int], str | None] = {}

    def __len__(self) -> int:
        return len(self.months)

    def kind_in_year(self, owner: str | None, year: int) -> str | None:
        if not owner:
            return None
        key = (owner.lstrip("0"), year)
        if key not in self._cache:
            self._cache[key] = self._classify(key[0], year)
        return self._cache[key]

    def kind(self, owner: str | None, d: date) -> str | None:
        return self.kind_in_year(owner, d.year)

    def _classify(self, owner: str, year: int) -> str | None:
        cutoff = date(year, 1, 1)
        years = range(year - LOOKBACK_YEARS, year)
        by_year: dict[int, set[int]] = defaultdict(set)
        for filed, y, m in self.months.get(owner, ()):
            if filed < cutoff and y in years:
                by_year[y].add(m)
        if any(not by_year[y] for y in years):
            return None
        common = set.intersection(*(by_year[y] for y in years))
        return ROUTINE if common else OPPORTUNISTIC

    def covers(self, year: int) -> bool:
        """True when the history starts early enough to classify trades in `year`."""
        return self.first_filed is not None and self.first_filed <= date(
            year - LOOKBACK_YEARS, 1, 31)

    @classmethod
    def load(cls, path: Path) -> Classifier:
        opener = gzip.open if str(path).endswith(".gz") else open
        rows = []
        with opener(path, "rt") as fh:
            for r in csv.DictReader(fh):
                rows.append((r["owner_cik"], date.fromisoformat(r["trade_date"]),
                             date.fromisoformat(r["filed"])))
        return cls(rows)
