"""The proven-insider gate for a shadow paper book (avik, 2026-11-09).

PR #12's study: of the buys the two-input model flags at 60 days, keep only those whose
insider has a positive record, meaning a shrunk 60-day excess over SPY above 0 on at
least `skill.MIN_EPISODES` earlier buy episodes, all known before the filing date.

Records come from two sources:

- **Study episodes** (`episodes.csv.gz` from `tt skill-study`, horizon 60): every
  episode from the SEC data sets, up to the last quarter loaded.
- **Live episodes**: buys from the book's cached daily filings that were filed after
  the study's last episode. Each is priced once its 60 sessions are over (open after
  the filing day to the 60th close, minus SPY, as `skill.price_outcome` does) and kept
  in `skill-live.csv` in the book's directory, so each day is priced only once.

k (the shrinkage) is the study's, fixed: re-estimating it live would let the gate drift.
"""

from __future__ import annotations

import csv
import gzip
import json
import logging
from datetime import date
from pathlib import Path

from .. import skill as SK
from ..features.calendar import NYSE
from . import ingest

log = logging.getLogger(__name__)

H = 60
DEFAULT_K = 2.896475     # PR #12, 60 days, episodes known before 2017
LIVE_FILE = "skill-live.csv"
LIVE_COLUMNS = ["owner", "issuer", "filed", "known", "excess"]
DONE_FILE = "skill-live-days.json"


def load_study_episodes(path: Path, h: int = H) -> list[SK.Episode]:
    opener = gzip.open if str(path).endswith(".gz") else open
    out = []
    with opener(path, "rt") as fh:
        for r in csv.DictReader(fh):
            if int(r["horizon"]) != h:
                continue
            out.append(SK.Episode(r["owner"], r["issuer"], date.fromisoformat(r["filed"]),
                                  date.fromisoformat(r["known"]), float(r["excess"])))
    return out


def matured(filed: date, today: date, h: int = H) -> bool:
    """True when a buy filed on `filed` has had its h-th session close before today."""
    return NYSE.add(NYSE.add(filed, 1), h - 1) < today


def live_rows(filings_dir: Path, after: date, today: date, prices, done: set[str],
              tickers: dict[str, str] | None = None, h: int = H):
    """(owner, issuer, filed, known, excess) for cached days after `after` whose buys
    have matured and that aren't in `done`; the days handled are added to `done`."""
    rows = []
    for p in sorted(filings_dir.glob("*.json")):
        try:
            d = date.fromisoformat(p.stem)
        except ValueError:
            continue
        if d <= after or p.stem in done or not matured(d, today, h):
            continue
        data = json.loads(p.read_text())
        for le in ingest.build_events(data["filings"], tickers):
            e = le.event
            if e.is_10b5_1:
                continue
            out = SK.price_outcome(prices, e.ticker, e.filing_date, h)
            if out is None:
                continue
            rows.append((SK.owner_key(le.filer_cik), SK.owner_key(e.issuer_cik) or e.ticker,
                         e.filing_date, out[0], out[1]))
        done.add(p.stem)
    return rows


class SkillGate:
    def __init__(self, book: SK.SkillBook, last_study_filed: date | None):
        self.book = book
        self.last_study_filed = last_study_filed

    @classmethod
    def load(cls, episodes: Path, state_dir: Path, filings_dir: Path | None, today: date,
             prices, tickers: dict[str, str] | None = None, k: float = DEFAULT_K) -> "SkillGate":
        study = load_study_episodes(episodes)
        last = max((e.filed for e in study), default=None)
        live_path, done_path = state_dir / LIVE_FILE, state_dir / DONE_FILE
        live = []
        if live_path.exists():
            with live_path.open() as fh:
                for r in csv.DictReader(fh):
                    live.append((r["owner"], r["issuer"], date.fromisoformat(r["filed"]),
                                 date.fromisoformat(r["known"]), float(r["excess"])))
        done = set(json.loads(done_path.read_text())) if done_path.exists() else set()
        if filings_dir is not None and filings_dir.is_dir() and last is not None:
            new = live_rows(filings_dir, last, today, prices, done, tickers)
            if new:
                live += new
                with live_path.open("w", newline="") as fh:
                    w = csv.writer(fh)
                    w.writerow(LIVE_COLUMNS)
                    w.writerows([(o, i, f.isoformat(), k_.isoformat(), repr(x))
                                 for o, i, f, k_, x in live])
            done_path.write_text(json.dumps(sorted(done)))
            log.info("skill gate: %s live episodes (%s new)", len(live), len(new))
        eps = SK.collapse([(e.owner, e.issuer, e.filed, e.known, e.excess) for e in study] + live)
        return cls(SK.SkillBook(eps, k), last)

    def record(self, owner_cik: str | None, filed: date) -> SK.Record:
        return self.book.record(SK.owner_key(owner_cik), filed)

    def passes(self, owner_cik: str | None, filed: date) -> tuple[bool, str]:
        r = self.record(owner_cik, filed)
        if r.shrunk is None:
            return False, f"no record (n={r.n})"
        note = f"record {100 * r.shrunk:+.1f}% (n={r.n})"
        return r.shrunk > 0, note


def stale_days(gate: SkillGate, today: date) -> int | None:
    return None if gate.last_study_filed is None else (today - gate.last_study_filed).days

