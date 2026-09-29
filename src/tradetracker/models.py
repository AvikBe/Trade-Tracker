"""Source-neutral records produced by parsers and consumed by the store."""

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal


@dataclass
class ParsedTrade:
    line_no: int
    side: str  # 'buy' | 'sell'
    txn_code: str
    trade_date: date
    shares: Decimal | None
    price: Decimal | None
    owner: str
    is_10b5_1: bool = False
    asset_type: str = "stock"
    amount_low: Decimal | None = None
    amount_high: Decimal | None = None


@dataclass
class ParsedFiling:
    source: str
    source_filing_id: str
    document_type: str
    filer_key: str
    filer_name: str
    filer_kind: str
    role: str | None
    issuer_cik: str | None
    issuer_name: str | None
    issuer_ticker: str | None
    period_of_report: date | None
    filed_at: datetime
    accepted_at: datetime | None = None
    source_url: str | None = None
    original_filed_on: date | None = None  # amendments: when the original was filed
    trades: list[ParsedTrade] = field(default_factory=list)
    raw: dict | None = None
