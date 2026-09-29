"""Parsers against real SEC filings (tests/fixtures/real).

The bulk data set and the filing XML describe the same filings, so the two parsers
must agree, apart from the data set rounding shares and prices to two decimals.
"""

from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pytest

from tradetracker.edgar.bulk import parse_quarter
from tradetracker.edgar.form4 import parse_form4

REAL = Path(__file__).parent / "fixtures" / "real"
BULK = {f.source_filing_id: f for f in parse_quarter(REAL / "2024q1_subset_form345.zip")}
XML_ACCESSIONS = sorted(p.stem for p in REAL.glob("*.xml"))


def _xml(acc):
    return parse_form4((REAL / f"{acc}.xml").read_bytes(), accession=acc)


def _r2(v):
    # The data set rounds half up (37.145 -> 37.15).
    return None if v is None else v.quantize(Decimal("0.01"), ROUND_HALF_UP)


@pytest.mark.parametrize("acc", XML_ACCESSIONS)
def test_bulk_and_xml_parsers_agree(acc):
    b, x = BULK[acc], _xml(acc)
    for attr in ("document_type", "filer_key", "role", "issuer_cik", "issuer_ticker",
                 "period_of_report"):
        assert getattr(b, attr) == getattr(x, attr), attr

    def key(t):
        return (t.side, t.trade_date, _r2(t.shares), _r2(t.price), t.owner, t.is_10b5_1)

    assert sorted(map(key, b.trades)) == sorted(map(key, x.trades))


def test_all_eight_filings_parse():
    assert len(BULK) == 8
    assert {f.document_type for f in BULK.values()} == {"4", "4/A"}


def test_joint_filing_picks_the_insider_not_the_fund():
    # Kaufman (director and 10% owner) filed jointly with three MAK Capital funds.
    f = BULK["0001019056-24-000015"]
    assert (f.filer_name, f.role) == ("Kaufman Michael A", "director")
    assert _xml("0001019056-24-000015").filer_key == f.filer_key


def test_placeholder_tickers_are_left_for_the_sec_mapping():
    assert BULK["0001415889-24-001467"].issuer_ticker is None  # filed as "NONE"
    assert BULK["0001209191-24-002460"].issuer_ticker is None


def test_10b5_1_from_checkbox_spelled_true_and_1():
    assert all(t.is_10b5_1 for t in BULK["0000950170-24-009502"].trades)  # AFF10B5ONE=true
    assert all(t.is_10b5_1 for t in BULK["0001214659-24-001532"].trades)  # AFF10B5ONE=1


def test_indirect_holders_and_fractional_shares():
    f = BULK["0001415889-24-001467"]
    assert [t.owner for t in f.trades] == ["self", "child", "child", "spouse", "indirect"]
    x = _xml("0001415889-24-001467")
    assert x.trades[0].shares == Decimal("97.4307")  # the XML keeps full precision


def test_ceo_amendment_with_open_market_buys():
    f = BULK["0001437749-24-007342"]
    assert f.document_type == "4/A" and f.role == "CEO" and f.issuer_ticker == "NBBK"
    assert [t.side for t in f.trades] == ["buy"] * 4
    assert f.filed_at.date() == date(2024, 3, 11)


def test_both_parsers_read_the_original_filing_date_of_an_amendment():
    assert BULK["0001493152-24-004448"].original_filed_on == date(2024, 1, 25)
    assert _xml("0001493152-24-004448").original_filed_on == date(2024, 1, 25)


def test_both_parsers_keep_every_joint_owner():
    want = ["1385702", "1426156", "1426157", "1760572"]  # Kaufman and three MAK funds
    assert BULK["0001019056-24-000015"].owner_ciks == want
    assert _xml("0001019056-24-000015").owner_ciks == want
