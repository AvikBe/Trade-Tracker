"""Parse a single Form 4 (or 4/A) ownershipDocument XML into a ParsedFiling."""

from datetime import date, datetime
from decimal import Decimal, InvalidOperation

from lxml import etree

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


def _text(el, path: str) -> str | None:
    found = el.find(path)
    if found is None or found.text is None:
        return None
    return found.text.strip() or None


def _value(el, path: str) -> str | None:
    # Most Form 4 leaves wrap their data in <value>; some are bare.
    return _text(el, f"{path}/value") or _text(el, path)


def _footnote_ids(el) -> list[str]:
    return [f.get("id") for f in el.iter("footnoteId") if f.get("id")]


def _decimal(s: str | None) -> Decimal | None:
    if s is None:
        return None
    try:
        return Decimal(s)
    except InvalidOperation:
        return None


def _date(s: str | None) -> date | None:
    return date.fromisoformat(s[:10]) if s else None


def _owner_role(owner) -> str | None:
    rel = owner.find("reportingOwnerRelationship")
    if rel is None:
        return None
    return classify_role(
        is_director=truthy(_text(rel, "isDirector")),
        is_officer=truthy(_text(rel, "isOfficer")),
        is_ten_pct=truthy(_text(rel, "isTenPercentOwner")),
        officer_title=_text(rel, "officerTitle"),
    )


def parse_form4(
    xml: bytes,
    *,
    accession: str,
    filed_at: datetime | None = None,
    accepted_at: datetime | None = None,
    source_url: str | None = None,
) -> ParsedFiling:
    root = etree.fromstring(xml)
    if root.tag != "ownershipDocument":
        root = root.find(".//ownershipDocument")
        if root is None:
            raise ValueError(f"{accession}: no ownershipDocument element")

    footnotes = {
        fn.get("id"): "".join(fn.itertext()).strip()
        for fn in root.findall("footnotes/footnote")
    }
    remarks = _text(root, "remarks")
    # aff10b5One is the checkbox added to Form 4 in April 2023.
    doc_10b5_1 = truthy(_text(root, "aff10b5One"))

    owners = [
        (_owner_role(o), _text(o, "reportingOwnerId/rptOwnerCik"), o)
        for o in root.findall("reportingOwner")
    ]
    owner = primary_owner(owners)
    if owner is None:
        raise ValueError(f"{accession}: no reportingOwner")
    role = _owner_role(owner)

    period = _date(_text(root, "periodOfReport"))
    if filed_at is None:
        filed_at = accepted_at or eastern_midnight(period or date.today())

    trades: list[ParsedTrade] = []
    rows = root.findall("nonDerivativeTable/nonDerivativeTransaction")
    for line_no, txn in enumerate(rows):
        code = _text(txn, "transactionCoding/transactionCode")
        if code not in KEPT_CODES:
            continue
        trade_date = _date(_value(txn, "transactionDate"))
        if trade_date is None:
            continue
        fn_text = " ".join(footnotes.get(i, "") for i in _footnote_ids(txn))
        trades.append(
            ParsedTrade(
                line_no=line_no,
                side=KEPT_CODES[code],
                txn_code=code,
                trade_date=trade_date,
                shares=_decimal(_value(txn, "transactionAmounts/transactionShares")),
                price=_decimal(_value(txn, "transactionAmounts/transactionPricePerShare")),
                owner=owner_from_nature(
                    _value(txn, "ownershipNature/directOrIndirectOwnership"),
                    _value(txn, "ownershipNature/natureOfOwnership"),
                ),
                is_10b5_1=doc_10b5_1 or mentions_10b5_1(fn_text),
            )
        )

    # A plan disclosed only in remarks applies to the whole filing.
    if mentions_10b5_1(remarks):
        for t in trades:
            t.is_10b5_1 = True

    return ParsedFiling(
        source="edgar",
        source_filing_id=accession,
        document_type=_text(root, "documentType") or "4",
        filer_key=(_text(owner, "reportingOwnerId/rptOwnerCik") or "").lstrip("0"),
        filer_name=_text(owner, "reportingOwnerId/rptOwnerName") or "",
        filer_kind="insider",
        role=role,
        issuer_cik=(_text(root, "issuer/issuerCik") or "").lstrip("0") or None,
        issuer_name=_text(root, "issuer/issuerName"),
        issuer_ticker=clean_ticker(_text(root, "issuer/issuerTradingSymbol")),
        period_of_report=period,
        original_filed_on=_date(_text(root, "dateOfOriginalSubmission")),
        owner_ciks=sorted({(o[1] or "").lstrip("0") for o in owners if (o[1] or "").strip("0")}),
        filed_at=filed_at,
        accepted_at=accepted_at,
        source_url=source_url,
        trades=trades,
    )
