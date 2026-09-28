from datetime import date

from bulkfixture import write
from tradetracker.edgar.bulk import parse_quarter, quarter_url


def test_quarter_url():
    assert quarter_url(2015, 1).endswith("/2015q1_form345.zip")


def test_parse_quarter(tmp_path):
    filings = {f.source_filing_id: f for f in parse_quarter(write(tmp_path / "q.zip"))}
    assert set(filings) == {
        "0000000001-24-000001", "0000000001-24-000002", "0000000001-24-000003"
    }  # the Form 3 is skipped

    f = filings["0000000001-24-000001"]
    assert f.filed_at.date() == date(2024, 3, 6)
    assert f.accepted_at is None
    assert f.issuer_cik == "1234567" and f.issuer_ticker == "EXWD"
    assert f.role == "CFO"
    assert [t.txn_code for t in f.trades] == ["P"]  # F (tax withholding) dropped
    assert f.source_url.endswith("/1234567/000000000124000001/0000000001-24-000001-index.htm")

    other = filings["0000000001-24-000003"]
    assert other.role == "10% owner"
    assert other.trades[0].owner == "trust"
    assert other.trades[0].is_10b5_1  # from remarks
