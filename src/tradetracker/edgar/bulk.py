"""Load the SEC's quarterly Insider Transactions Data Sets (flattened Forms 3/4/5).

Each quarter is a ZIP, usually at
https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/{YYYY}q{N}_form345.zip
but the newest quarter sometimes sits under a different folder, so we read the links
off the SEC's data set index page and only fall back to that pattern. Each ZIP holds tab-separated tables keyed by ACCESSION_NUMBER. We read SUBMISSION,
REPORTINGOWNER, NONDERIV_TRANS and FOOTNOTES, and emit one ParsedFiling per Form 4.

The data sets carry a filing date but no acceptance time, so accepted_at stays NULL
and the backtest enters on the second trading day after filing (see plans doc).
Column names checked against the live 2024q1 file.
"""

import csv
import io
import re
import zipfile
from collections import defaultdict
from collections.abc import Iterator
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

from ..models import ParsedFiling, ParsedTrade
from .common import (
    KEPT_CODES,
    classify_role,
    clean_ticker,
    eastern_midnight,
    mentions_10b5_1,
    owner_from_nature,
    primary_owner,
    truthy,
)

BASE_URL = "https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets"
INDEX_URL = (
    "https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets"
)
FORM4_TYPES = {"4", "4/A"}
_ZIP_LINK = re.compile(r'href="([^"]*?(\d{4})q([1-4])_form345\.zip)"', re.IGNORECASE)


def quarter_url(year: int, quarter: int) -> str:
    return f"{BASE_URL}/{year}q{quarter}_form345.zip"


def index_links(html: str) -> dict[tuple[int, int], str]:
    """Map (year, quarter) to the ZIP URL linked from the data set index page."""
    out = {}
    for href, y, q in _ZIP_LINK.findall(html):
        url = href if href.startswith("http") else "https://www.sec.gov" + href
        out.setdefault((int(y), int(q)), url)
    return out


def _rows(zf: zipfile.ZipFile, name: str) -> Iterator[dict[str, str]]:
    member = next(n for n in zf.namelist() if n.upper().endswith(name))
    with zf.open(member) as fh:
        text = io.TextIOWrapper(fh, encoding="utf-8", errors="replace", newline="")
        yield from csv.DictReader(text, delimiter="\t", quoting=csv.QUOTE_NONE)


def _sec_date(s: str | None) -> date | None:
    """Parse '02-JAN-2024' (data set format) or ISO dates."""
    if not s:
        return None
    s = s.strip()
    for fmt in ("%d-%b-%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(s[:11], fmt).date()
        except ValueError:
            continue
    return None


def _dec(s: str | None) -> Decimal | None:
    if not s:
        return None
    try:
        return Decimal(s)
    except InvalidOperation:
        return None


def _owner_role(owner: dict[str, str]) -> str | None:
    rel = (owner.get("RPTOWNER_RELATIONSHIP") or "").lower()
    return classify_role(
        is_director="director" in rel,
        is_officer="officer" in rel,
        is_ten_pct="tenpercent" in rel.replace(" ", "").replace("%", "percent"),
        officer_title=owner.get("RPTOWNER_TITLE"),
    )


def parse_quarter(path: Path | str) -> Iterator[ParsedFiling]:
    with zipfile.ZipFile(path) as zf:
        subs = {
            r["ACCESSION_NUMBER"]: r
            for r in _rows(zf, "SUBMISSION.TSV")
            if r.get("DOCUMENT_TYPE") in FORM4_TYPES
        }

        candidates: dict[str, list] = defaultdict(list)
        for r in _rows(zf, "REPORTINGOWNER.TSV"):
            if r["ACCESSION_NUMBER"] in subs:
                candidates[r["ACCESSION_NUMBER"]].append((_owner_role(r), r["RPTOWNERCIK"], r))
        owners = {acc: primary_owner(c) for acc, c in candidates.items()}

        footnotes: dict[tuple[str, str], str] = {}
        for r in _rows(zf, "FOOTNOTES.TSV"):
            if r["ACCESSION_NUMBER"] in subs:
                footnotes[(r["ACCESSION_NUMBER"], r["FOOTNOTE_ID"])] = r.get("FOOTNOTE_TXT", "")

        txns: dict[str, list[dict[str, str]]] = defaultdict(list)
        for r in _rows(zf, "NONDERIV_TRANS.TSV"):
            if r["ACCESSION_NUMBER"] in subs:
                txns[r["ACCESSION_NUMBER"]].append(r)

    for acc, sub in subs.items():
        filed = _sec_date(sub.get("FILING_DATE"))
        owner = owners.get(acc)
        if filed is None or owner is None:
            continue
        role = _owner_role(owner)
        doc_10b5_1 = truthy(sub.get("AFF10B5ONE")) or mentions_10b5_1(sub.get("REMARKS"))

        trades = []
        rows = sorted(txns.get(acc, []), key=lambda r: int(r.get("NONDERIV_TRANS_SK") or 0))
        for line_no, r in enumerate(rows):
            code = (r.get("TRANS_CODE") or "").strip()
            if code not in KEPT_CODES:
                continue
            trade_date = _sec_date(r.get("TRANS_DATE"))
            if trade_date is None:
                continue
            fn_ids = [v for k, v in r.items() if k.endswith("_FN") and v]
            fn_text = " ".join(
                footnotes.get((acc, fid.strip()), "")
                for ids in fn_ids
                for fid in ids.split(",")
            )
            trades.append(
                ParsedTrade(
                    line_no=line_no,
                    side=KEPT_CODES[code],
                    txn_code=code,
                    trade_date=trade_date,
                    shares=_dec(r.get("TRANS_SHARES")),
                    price=_dec(r.get("TRANS_PRICEPERSHARE")),
                    owner=owner_from_nature(
                        r.get("DIRECT_INDIRECT_OWNERSHIP"), r.get("NATURE_OF_OWNERSHIP")
                    ),
                    is_10b5_1=doc_10b5_1 or mentions_10b5_1(fn_text),
                )
            )

        cik = (sub.get("ISSUERCIK") or "").lstrip("0") or None
        yield ParsedFiling(
            source="edgar",
            source_filing_id=acc,
            document_type=sub["DOCUMENT_TYPE"],
            filer_key=(owner.get("RPTOWNERCIK") or "").lstrip("0"),
            filer_name=owner.get("RPTOWNERNAME") or "",
            filer_kind="insider",
            role=role,
            issuer_cik=cik,
            issuer_name=sub.get("ISSUERNAME"),
            issuer_ticker=clean_ticker(sub.get("ISSUERTRADINGSYMBOL")),
            period_of_report=_sec_date(sub.get("PERIOD_OF_REPORT")),
            original_filed_on=_sec_date(sub.get("DATE_OF_ORIG_SUB")),
            filed_at=eastern_midnight(filed),
            source_url=_filing_url(cik, acc),
            trades=trades,
        )


def _filing_url(cik: str | None, accession: str) -> str | None:
    if not cik:
        return None
    return (
        f"https://www.sec.gov/Archives/edgar/data/{cik}/"
        f"{accession.replace('-', '')}/{accession}-index.htm"
    )
