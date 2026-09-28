"""Live Form 4 ingest from EDGAR's latest-filings Atom feed.

The feed lists the newest filings of a form type; each entry's <updated> is the
acceptance time. For each new accession we read the filing's index.json to find
the ownership XML, then parse it.
"""

import re
from dataclasses import dataclass
from datetime import datetime

from lxml import etree

from .client import EdgarClient

FEED_URL = (
    "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent"
    "&type=4&company=&dateb=&owner=include&start={start}&count=100&output=atom"
)
ATOM = "{http://www.w3.org/2005/Atom}"
_ACC = re.compile(r"(\d{10}-\d{2}-\d{6})")
_CIK_PATH = re.compile(r"/edgar/data/(\d+)/")


@dataclass
class FeedEntry:
    accession: str
    form_type: str
    accepted_at: datetime
    index_url: str
    cik: str


def parse_feed(xml: bytes) -> list[FeedEntry]:
    root = etree.fromstring(xml)
    out = []
    for e in root.iter(f"{ATOM}entry"):
        link = e.find(f"{ATOM}link")
        href = link.get("href") if link is not None else ""
        acc = _ACC.search(e.findtext(f"{ATOM}id") or "") or _ACC.search(href)
        cik = _CIK_PATH.search(href)
        cat = e.find(f"{ATOM}category")
        if not acc or not cik:
            continue
        out.append(
            FeedEntry(
                accession=acc.group(1),
                form_type=cat.get("term") if cat is not None else "",
                accepted_at=datetime.fromisoformat(e.findtext(f"{ATOM}updated")),
                index_url=href,
                cik=cik.group(1),
            )
        )
    return out


def fetch_latest(client: EdgarClient, pages: int = 1) -> list[FeedEntry]:
    entries: list[FeedEntry] = []
    for page in range(pages):
        entries += parse_feed(client.get(FEED_URL.format(start=page * 100)).content)
    # The same filing appears once per party (issuer and reporting owner).
    seen: dict[str, FeedEntry] = {}
    for e in entries:
        if e.form_type in {"4", "4/A"}:
            seen.setdefault(e.accession, e)
    return list(seen.values())


def ownership_xml_url(client: EdgarClient, entry: FeedEntry) -> str:
    folder = f"https://www.sec.gov/Archives/edgar/data/{entry.cik}/{entry.accession.replace('-', '')}"
    listing = client.get(f"{folder}/index.json").json()
    names = [i["name"] for i in listing["directory"]["item"]]
    # The raw ownership document is the .xml that isn't an XSL-rendered copy.
    xmls = [n for n in names if n.lower().endswith(".xml") and not n.lower().startswith("xsl")]
    if not xmls:
        raise LookupError(f"{entry.accession}: no ownership XML in {folder}")
    return f"{folder}/{xmls[0]}"
