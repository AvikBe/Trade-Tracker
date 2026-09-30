from datetime import date

from bulkfixture import write
from tradetracker.edgar.bulk import index_links, parse_quarter, quarter_url


def test_quarter_url():
    assert quarter_url(2015, 1).endswith("/insider-transactions-data-sets/2015q1_form345.zip")


def test_index_links_prefers_page_links():
    html = (
        '<a href="/files/datastandardsinnovation/data/insider-transactions-data-sets/'
        '2026q2_form345.zip">2026 Q2</a>'
        '<a href="/files/structureddata/data/insider-transactions-data-sets/'
        '2026q1_form345.zip">2026 Q1</a>'
    )
    links = index_links(html)
    assert links[(2026, 2)] == (
        "https://www.sec.gov/files/datastandardsinnovation/data/"
        "insider-transactions-data-sets/2026q2_form345.zip"
    )
    assert (2026, 1) in links


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
    assert f.trades[0].shares_owned_after is None   # column absent from this fixture
    assert f.source_url.endswith("/1234567/000000000124000001/0000000001-24-000001-index.htm")

    other = filings["0000000001-24-000003"]
    assert other.role == "10% owner"
    assert other.trades[0].owner == "trust"
    assert other.trades[0].is_10b5_1  # from remarks


def test_clean_ticker_handles_real_filer_input():
    from tradetracker.edgar.common import clean_ticker

    cases = {
        "aapl": "AAPL",
        "NONE": None,
        "[ NONE ]": None,
        "N/A": None,
        "-": None,
        "(SIRI)": "SIRI",
        "NYSE: SCS": "SCS",
        "NYSE/TRN": "TRN",
        "Z AND ZG": "Z",
        "GEF,GEF.B": "GEF",
        "HEI, HEI.A": "HEI",
        "GTII/GTBIF": "GTII",
        "N O G": "NOG",
        "BRK.B": "BRK.B",
        None: None,
    }
    for raw, want in cases.items():
        assert clean_ticker(raw) == want, raw


def test_primary_owner_prefers_seniority_then_lowest_cik():
    from tradetracker.edgar.common import primary_owner

    assert primary_owner([("10% owner", "0001385702", "fund"), ("director", "0001426157", "person")]) == "person"
    assert primary_owner([(None, "1603544", "a"), (None, "1377832", "b"), (None, "1618205", "c")]) == "b"
    assert primary_owner([("officer", "5", "o"), ("CEO", "9", "ceo")]) == "ceo"
    assert primary_owner([(None, "", "blank"), (None, "7", "seven")]) == "seven"
    assert primary_owner([]) is None


def test_clean_ticker_more_real_values():
    from tradetracker.edgar.common import clean_ticker

    assert clean_ticker("BCDA;BCDAW") == "BCDA"
    assert clean_ticker("BIO BIOB") == "BIO"
    assert clean_ticker("BBXIA/B") == "BBXIA"
    assert clean_ticker("NASDAQ:AAPL") == "AAPL"
    assert clean_ticker("  msft  ") == "MSFT"
    assert clean_ticker("[NONE") is None
    assert clean_ticker("") is None
