"""New Form 4 buys from EDGAR's daily form index, as backtest events.

Each business day EDGAR publishes `form.YYYYMMDD.idx`, listing every filing made that
day (a filing appears once per party, so it is deduplicated by accession). For each
Form 4 we read the full submission text, which carries the acceptance time and the
ownership XML in one request, and parse it with the same parser as the live feed.

Events follow `backtest.events.load_events`: one insider, stock and side per filing
date; 4/A amendments, trades dated after the filing and lines without a ticker are
left out; sales are kept out too, since the tracker only trades buys. The filing date
is the index date, so entry is the next NYSE open after it, as in the backtest's bulk
history (where acceptance times are unknown).
"""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import httpx

from ..backtest.events import Event
from ..edgar.common import EASTERN
from ..edgar.form4 import parse_form4
from ..models import ParsedFiling

log = logging.getLogger(__name__)

INDEX_URL = "https://www.sec.gov/Archives/edgar/daily-index/{y}/QTR{q}/form.{d}.idx"
ARCHIVE_URL = "https://www.sec.gov/Archives/{path}"
FORMS = {"4", "4/A"}
_ACC = re.compile(r"(\d{10}-\d{2}-\d{6})\.txt$")
_XML = re.compile(rb"<XML>\s*(.*?)\s*</XML>", re.S | re.I)
_ACCEPTED = re.compile(rb"<ACCEPTANCE-DATETIME>(\d{14})")
CACHE_VERSION = 1
WORKERS = 4


@dataclass(frozen=True)
class IndexEntry:
    form: str
    cik: str
    filed: date
    path: str               # edgar/data/<cik>/<accession>.txt

    @property
    def accession(self) -> str:
        m = _ACC.search(self.path)
        return m.group(1) if m else self.path


def index_url(d: date) -> str:
    return INDEX_URL.format(y=d.year, q=(d.month - 1) // 3 + 1, d=d.strftime("%Y%m%d"))


def parse_index(text: str, forms=FORMS) -> list[IndexEntry]:
    """Form 4 rows of a daily form index, one per accession.

    Rows are fixed-width, but company names can contain anything, so each row is read
    from the right: file name, date, CIK, then the form type is the first column.
    """
    out: dict[str, IndexEntry] = {}
    started = False
    for line in text.splitlines():
        if line.startswith("-----"):
            started = True
            continue
        if not started or not line.strip():
            continue
        parts = line.split()
        if len(parts) < 4 or not parts[-1].startswith("edgar/"):
            continue
        path, filed, cik = parts[-1], parts[-2], parts[-3]
        form = line[:17].strip() or parts[0]
        if form not in forms or not filed.isdigit() or len(filed) != 8:
            continue
        e = IndexEntry(form, cik, date(int(filed[:4]), int(filed[4:6]), int(filed[6:])), path)
        out.setdefault(e.accession, e)
    return sorted(out.values(), key=lambda e: e.accession)


def parse_submission(raw: bytes, entry: IndexEntry) -> ParsedFiling:
    """A Form 4 from its full submission text (SGML header plus documents)."""
    accepted = None
    m = _ACCEPTED.search(raw[:20_000])
    if m:
        accepted = datetime.strptime(m.group(1).decode(), "%Y%m%d%H%M%S").replace(tzinfo=EASTERN)
    xml = next((x for x in _XML.findall(raw) if b"ownershipDocument" in x), None)
    if xml is None:
        raise ValueError(f"{entry.accession}: no ownership XML in the submission")
    return parse_form4(xml, accession=entry.accession,
                       filed_at=datetime.combine(entry.filed, datetime.min.time(), tzinfo=EASTERN),
                       accepted_at=accepted, source_url=ARCHIVE_URL.format(path=entry.path))


# ------------------------------------------------------------------- per-day cache
#
# A day's filings never change once its index is out, so each day is fetched once and
# kept as JSON: the buy lines only (sales and other codes are not needed to trade).

def _filing_json(f: ParsedFiling) -> dict:
    return {
        "accession": f.source_filing_id, "form": f.document_type, "filer": f.filer_key,
        "filer_name": f.filer_name, "role": f.role, "issuer_cik": f.issuer_cik,
        "issuer_name": f.issuer_name, "ticker": f.issuer_ticker,
        "filed": f.filed_at.astimezone(EASTERN).date().isoformat(),
        "accepted_at": f.accepted_at.isoformat() if f.accepted_at else None,
        "url": f.source_url,
        "trades": [{"line": t.line_no, "side": t.side, "code": t.txn_code,
                    "trade_date": t.trade_date.isoformat(),
                    "shares": None if t.shares is None else str(t.shares),
                    "price": None if t.price is None else str(t.price),
                    "owned_after": None if t.shares_owned_after is None else str(t.shares_owned_after),
                    "owner": t.owner, "10b5_1": t.is_10b5_1}
                   for t in f.trades],
    }


def fetch_day(client, d: date, cache: Path | None = None) -> dict | None:
    """The day's Form 4 filings with open-market lines, or None if its index isn't out.

    Returns {"date", "filings": [...], "counts": {...}}; cached in `cache/<date>.json`.
    """
    path = cache / f"{d.isoformat()}.json" if cache else None
    if path and path.exists():
        data = json.loads(path.read_text())
        if data.get("version") == CACHE_VERSION:
            return data
    try:
        text = client.get(index_url(d)).text
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code in (403, 404):
            return None        # not published yet (or no filings that day)
        raise
    entries = parse_index(text)
    counts = {"index_rows": len(entries), "parsed": 0, "errors": 0, "with_trades": 0}
    filings, errors = [], []

    def one(e: IndexEntry):
        try:
            return e, parse_submission(client.get(ARCHIVE_URL.format(path=e.path)).content, e), None
        except Exception as exc:  # one bad filing must not stop the day
            return e, None, exc

    # A few requests in flight hide network latency; the client keeps SEC's rate limit.
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for e, f, exc in pool.map(one, entries):
            if exc is not None:
                counts["errors"] += 1
                errors.append({"accession": e.accession, "error": f"{type(exc).__name__}: {exc}"[:300]})
                continue
            counts["parsed"] += 1
            if f.trades:
                counts["with_trades"] += 1
                filings.append(_filing_json(f))
    data = {"version": CACHE_VERSION, "date": d.isoformat(), "counts": counts,
            "filings": filings, "errors": errors}
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1))
        tmp.replace(path)
    return data


# ------------------------------------------------------------------- events

@dataclass
class LiveEvent:
    event: Event
    accessions: list[str]
    filer_cik: str
    filer_name: str
    issuer_name: str | None


def build_events(filings: list[dict], tickers: dict[str, str] | None = None,
                 exclusions: dict | None = None) -> list[LiveEvent]:
    """Buy events from a day's filings, grouped as the backtest groups them.

    `tickers` (issuer CIK -> today's ticker) fills a missing symbol, as `tt map-tickers`
    did for the backtest's history.
    """
    ex = exclusions if exclusions is not None else {}

    def skip(reason: str, n: int = 1) -> None:
        ex[reason] = ex.get(reason, 0) + n

    events: dict[tuple, LiveEvent] = {}
    for f in filings:
        buys = [t for t in f["trades"] if t["side"] == "buy"]
        if not buys:
            continue
        if f["form"].endswith("/A"):
            skip("amendment", len(buys))
            continue
        ticker = f["ticker"] or (tickers or {}).get((f["issuer_cik"] or "").lstrip("0"))
        if not ticker:
            skip("no_ticker", len(buys))
            continue
        filed = date.fromisoformat(f["filed"])
        for t in buys:
            td = date.fromisoformat(t["trade_date"])
            if td > filed:
                skip("trade_after_filing")
                continue
            key = (f["filer"], ticker, filed)
            le = events.get(key)
            if le is None:
                e = Event(filer_id=int(f["filer"] or 0), issuer_cik=f["issuer_cik"], ticker=ticker,
                          side="buy", filing_date=filed, accepted_at=None,
                          first_trade_date=td, last_trade_date=td, lag_days=0, lag_ratio=0.0,
                          role=f["role"])
                le = events[key] = LiveEvent(e, [], f["filer"], f["filer_name"], f["issuer_name"])
            e = le.event
            if f["accession"] not in le.accessions:
                le.accessions.append(f["accession"])
            e.first_trade_date = min(e.first_trade_date, td)
            e.last_trade_date = max(e.last_trade_date, td)
            shares = float(t["shares"]) if t["shares"] else None
            price = float(t["price"]) if t["price"] else None
            if shares and price and shares > 0 and price > 0:
                e.value += shares * price
                e.shares += shares
            e.is_10b5_1 = e.is_10b5_1 or bool(t["10b5_1"])
            if t["owned_after"] is None or shares is None:
                e.holdings_missing = True
            else:
                acct = t["owner"] or "self"
                before = float(t["owned_after"]) - shares
                e.held_before[acct] = min(e.held_before.get(acct, before), before)
            e.n_lines += 1
    return sorted(events.values(), key=lambda le: (le.event.filing_date, le.event.ticker, le.filer_cik))
