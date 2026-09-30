from datetime import date
from decimal import Decimal
from pathlib import Path

from tradetracker.edgar.form4 import parse_form4

FIX = Path(__file__).parent / "fixtures"


def test_open_market_buys_are_kept_and_other_codes_dropped():
    f = parse_form4((FIX / "form4_buy.xml").read_bytes(), accession="0001234567-24-000001")
    assert f.document_type == "4"
    assert f.issuer_cik == "1234567"
    assert f.issuer_ticker == "EXWD"
    assert f.filer_key == "7654321"
    assert f.role == "CFO"
    assert [t.txn_code for t in f.trades] == ["P", "P"]  # the M exercise is dropped
    first, second = f.trades
    assert first.line_no == 0 and second.line_no == 2
    assert first.side == "buy"
    assert first.shares == Decimal("10000") and first.price == Decimal("12.34")
    assert first.shares_owned_after == Decimal("50000")
    assert second.shares_owned_after == Decimal("2500")
    assert first.trade_date == date(2024, 3, 4)
    assert first.owner == "self"
    assert second.owner == "spouse"
    assert not any(t.is_10b5_1 for t in f.trades)


def test_amendment_with_10b5_1_footnote():
    f = parse_form4(
        (FIX / "form4a_10b5_1_sale.xml").read_bytes(), accession="0001234567-24-000002"
    )
    assert f.document_type == "4/A"
    assert f.role == "CEO"
    assert [t.side for t in f.trades] == ["sell", "sell"]
    # Only the row carrying the plan footnote is flagged.
    assert [t.is_10b5_1 for t in f.trades] == [True, False]


def test_aff10b5one_checkbox_flags_every_row():
    xml = (FIX / "form4_buy.xml").read_bytes().replace(
        b"<aff10b5One>0</aff10b5One>", b"<aff10b5One>1</aff10b5One>"
    )
    f = parse_form4(xml, accession="x")
    assert all(t.is_10b5_1 for t in f.trades)
