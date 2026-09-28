"""Helpers shared by the Form 4 XML parser and the bulk dataset loader."""

import re
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

EASTERN = ZoneInfo("America/New_York")

# Open-market purchases and sales. Everything else (A, M, G, F, ...) is not signal.
KEPT_CODES = {"P": "buy", "S": "sell"}

_10B5_1 = re.compile(r"10b5-?1", re.IGNORECASE)
_CEO = re.compile(r"\b(ceo|chief executive|principal executive)\b", re.IGNORECASE)
_CFO = re.compile(r"\b(cfo|chief financial|principal financial)\b", re.IGNORECASE)
_EXCHANGE = re.compile(r"^(NYSE|NASDAQ|AMEX|NYSEAMERICAN|NYSE AMERICAN|OTC[A-Z]*|CBOE)\s*[:/]\s*")
_SPACED_LETTERS = re.compile(r"^[A-Z](?: [A-Z])+$")
_LIST_SEP = re.compile(r"\s+AND\s+|[,;/&\s]+")
_TICKER = re.compile(r"^[A-Z0-9]{1,6}(?:[.-][A-Z0-9]{1,3})?$")


def clean_ticker(raw: str | None) -> str | None:
    """Normalize the free-text trading symbol filers type on Form 4.

    Real filings carry 'NONE', 'N/A', '(SIRI)', 'NYSE: SCS', 'Z AND ZG' or
    'GEF,GEF.B'. We keep the first symbol and return None for anything that still
    doesn't look like a ticker, so `tt map-tickers` fills it from the SEC mapping.
    """
    s = re.sub(r"[\[\]()]", " ", (raw or "").upper()).strip()
    s = _EXCHANGE.sub("", s)
    if _SPACED_LETTERS.match(s):
        s = s.replace(" ", "")
    first = next((t for t in _LIST_SEP.split(s) if t), "")
    if first in {"NONE", "NA", "N", "NIL"} or not _TICKER.match(first):
        return None
    return first


def mentions_10b5_1(text: str | None) -> bool:
    return bool(text and _10B5_1.search(text))


def classify_role(
    *,
    is_director: bool,
    is_officer: bool,
    is_ten_pct: bool,
    officer_title: str | None,
) -> str | None:
    """Collapse a reporting owner's relationship into the spec's role ladder.

    CEO and CFO outrank other officers, officers outrank directors, and 10% owners
    rank lowest, so the most senior applicable role wins.
    """
    title = officer_title or ""
    if _CEO.search(title):
        return "CEO"
    if _CFO.search(title):
        return "CFO"
    if is_officer or title:
        return "officer"
    if is_director:
        return "director"
    if is_ten_pct:
        return "10% owner"
    return None


def owner_from_nature(direct_indirect: str | None, nature: str | None) -> str:
    if (direct_indirect or "D").upper().startswith("D"):
        return "self"
    n = (nature or "").lower()
    if "spouse" in n or "wife" in n or "husband" in n:
        return "spouse"
    if "child" in n or "son" in n or "daughter" in n:
        return "child"
    if "trust" in n:
        return "trust"
    return "indirect"


def eastern_midnight(d: date) -> datetime:
    return datetime.combine(d, time(0, 0), tzinfo=EASTERN)


def truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "y", "yes"}
