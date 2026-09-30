from datetime import date
from pathlib import Path

from bulkfixture import write
from tradetracker import report, store, validate
from tradetracker.edgar.bulk import parse_quarter
from tradetracker.edgar.form4 import parse_form4
from tradetracker.prices import loader
from tradetracker.prices.tiingo import PriceBar

FIX = Path(__file__).parent / "fixtures"


def _load(conn, tmp_path):
    for f in parse_quarter(write(tmp_path / "q.zip")):
        store.save_filing(conn, f)
    conn.commit()


def test_bulk_ingest_is_idempotent_and_links_amendments(conn, tmp_path):
    _load(conn, tmp_path)
    _load(conn, tmp_path)
    assert conn.execute("SELECT count(*) FROM filings").fetchone()[0] == 3
    assert conn.execute("SELECT count(*) FROM trades").fetchone()[0] == 3
    orig, amended = conn.execute(
        "SELECT a.amends_filing_id, o.filing_id FROM filings a JOIN filings o "
        "ON o.source_filing_id = '0000000001-24-000001' "
        "WHERE a.source_filing_id = '0000000001-24-000002'"
    ).fetchone()
    assert orig == amended
    assert conn.execute(
        "SELECT first_seen, last_seen FROM ticker_history WHERE cik = '1234567'"
    ).fetchone() == (date(2024, 3, 6), date(2024, 3, 20))


def test_live_feed_fills_accepted_at_without_touching_first_seen(conn, tmp_path):
    _load(conn, tmp_path)
    before = conn.execute(
        "SELECT first_seen_at FROM filings WHERE source_filing_id = '0000000001-24-000001'"
    ).fetchone()[0]
    from datetime import datetime
    from zoneinfo import ZoneInfo

    accepted = datetime(2024, 3, 6, 16, 31, 5, tzinfo=ZoneInfo("America/New_York"))
    live = parse_form4((FIX / "form4_buy.xml").read_bytes(), accession="0000000001-24-000001",
                       accepted_at=accepted, filed_at=accepted)
    _, created = store.save_filing(conn, live)
    assert not created
    row = conn.execute(
        "SELECT first_seen_at, accepted_at FROM filings "
        "WHERE source_filing_id = '0000000001-24-000001'"
    ).fetchone()
    assert row == (before, accepted)


def test_validate_and_report(conn, tmp_path):
    _load(conn, tmp_path)
    store.save_prices(conn, [
        PriceBar("EXWD", date(2024, 3, 4), 12, 12.6, 11.9, 12.4, 12, 12.4, 1000),
        # OTHR traded at $40 but the day's range was ~$20: should be rejected.
        PriceBar("OTHR", date(2024, 3, 5), 20, 21, 19, 20, 20, 20, 1000),
    ])
    conn.execute("UPDATE trades SET ticker = NULL WHERE txn_code = 'P' AND filing_id = "
                 "(SELECT filing_id FROM filings WHERE document_type = '4/A')")
    counts = validate.run(conn)
    assert counts == {"trade_after_filing": 0, "unknown_ticker": 1, "price_out_of_range": 1}
    assert validate.run(conn) == {k: 0 for k in counts}  # idempotent

    text = report.render(conn)
    assert "edgar   2024" in text


def test_price_queue_puts_backfills_first_and_respects_quota(conn, tmp_path):
    _load(conn, tmp_path)
    todo = dict(loader.tickers_to_load(conn))
    assert todo["EXWD"] == loader.HISTORY_START and "SPY" in todo
    for _ in range(49):
        store.log_api_call(conn, "tiingo", "x")
    assert loader.remaining_quota(conn) == 1


REAL_ZIP = Path(__file__).parent / "fixtures" / "real" / "2024q1_subset_form345.zip"


def _counts(conn):
    return conn.execute(
        "SELECT (SELECT count(*) FROM filings), (SELECT count(*) FROM trades), "
        "(SELECT count(*) FROM filers), (SELECT count(*) FROM ticker_history)"
    ).fetchone()


def test_real_quarter_reload_changes_nothing(conn):
    for _ in range(2):
        for f in parse_quarter(REAL_ZIP):
            store.save_filing(conn, f)
        conn.commit()
        if _ == 0:
            first = _counts(conn)
    assert first == _counts(conn)
    assert first[:2] == (8, 16)


def test_amendment_listed_before_its_original_is_linked_after_load(conn):
    # In the real 2024q1 data set, MINM's 4/A (filed Jan 31) sits on line 82 of
    # SUBMISSION.tsv and its original 4 (filed Jan 25) on line 11,588.
    filings = list(parse_quarter(REAL_ZIP))
    amendment = next(f for f in filings if f.source_filing_id == "0001493152-24-004448")
    original = next(f for f in filings if f.source_filing_id == "0001493152-24-003754")
    store.save_filing(conn, amendment)
    store.save_filing(conn, original)
    assert conn.execute("SELECT count(amends_filing_id) FROM filings").fetchone()[0] == 0

    assert store.link_amendments(conn) == 1
    assert store.link_amendments(conn) == 0  # idempotent
    linked_to = conn.execute(
        "SELECT o.source_filing_id FROM filings a JOIN filings o ON o.filing_id = a.amends_filing_id "
        "WHERE a.source_filing_id = '0001493152-24-004448'"
    ).fetchone()[0]
    assert linked_to == "0001493152-24-003754"


def test_amendment_without_loaded_original_stays_unlinked_and_is_reported(conn):
    for f in parse_quarter(REAL_ZIP):
        store.save_filing(conn, f)
    store.link_amendments(conn)
    # NBBK's 4/A amends a filing that is not in the fixture.
    assert conn.execute(
        "SELECT amends_filing_id FROM filings WHERE source_filing_id = '0001437749-24-007342'"
    ).fetchone()[0] is None
    assert "amendments: 2, linked to their original: 1 (50.0%)" in report.render(conn)


def test_trade_after_filing_is_rejected(conn, tmp_path):
    _load(conn, tmp_path)
    conn.execute("UPDATE trades SET trade_date = '2024-12-31' WHERE line_no = 0 AND filing_id = "
                 "(SELECT filing_id FROM filings WHERE source_filing_id = '0000000001-24-000001')")
    assert validate.run(conn)["trade_after_filing"] == 1
    detail = conn.execute("SELECT detail FROM rejects WHERE rule = 'trade_after_filing'").fetchone()
    assert detail[0] == "trade 2024-12-31 after filing 2024-03-06"


class _FakeTiingo:
    def __init__(self, missing=(), failing=()):
        self.calls, self.missing, self.failing = [], set(missing), set(failing)

    def daily(self, ticker, start):
        import httpx

        from tradetracker.prices.tiingo import TickerNotFound

        self.calls.append((ticker, start))
        if ticker in self.missing:
            raise TickerNotFound(ticker)
        if ticker in self.failing:
            raise httpx.HTTPStatusError("429", request=httpx.Request("GET", "x"),
                                        response=httpx.Response(429))
        return [PriceBar(ticker, date(2024, 3, 4), 1, 1, 1, 1, 1, 1, 10)]


def test_price_loader_records_not_found_errors_and_resumes(conn, tmp_path):
    _load(conn, tmp_path)
    fake = _FakeTiingo(missing={"OTHR"}, failing={"SPY"})
    stats = loader.run(conn, fake, limit=100)
    assert stats["not_found"] == 1 and stats["errors"] == 1
    statuses = dict(conn.execute("SELECT ticker, status FROM price_status").fetchall())
    assert statuses["OTHR"] == "not_found" and statuses["SPY"] == "error"
    assert statuses["EXWD"] == "ok"

    # Next run: the loaded ticker only updates from its last bar, and the missing one
    # is not retried for 30 days. The errored one is retried.
    todo = dict(loader.tickers_to_load(conn))
    assert "OTHR" not in todo and "SPY" in todo
    assert todo["EXWD"] == date(2024, 3, 5)


def test_price_loader_never_exceeds_the_limit(conn, tmp_path):
    _load(conn, tmp_path)
    fake = _FakeTiingo()
    loader.run(conn, fake, limit=3)
    assert len(fake.calls) == 3
    assert conn.execute("SELECT count(*) FROM api_calls").fetchone()[0] == 3


def test_amendment_that_corrects_the_period_links_by_original_filing_date(conn):
    filings = {f.source_filing_id: f for f in parse_quarter(REAL_ZIP)}
    original = filings["0001493152-24-003754"]
    amendment = filings["0001493152-24-004448"]
    assert amendment.original_filed_on == original.filed_at.date()
    amendment.period_of_report = date(2024, 1, 19)  # as if the 4/A fixed the date
    store.save_filing(conn, original)
    store.save_filing(conn, amendment)
    assert store.link_amendments(conn) == 1


PLCE_ZIP = Path(__file__).parent / "fixtures" / "real" / "2024q1_plce_joint_form345.zip"


def test_joint_amendments_that_add_an_owner_link_to_their_originals(conn):
    # Children's Place, Feb 2024: Mithaq's three 4/As list Mithaq Capital, which the
    # originals did not, so the primary owner differs between amendment and original.
    for f in parse_quarter(PLCE_ZIP):
        store.save_filing(conn, f)
    store.link_amendments(conn)
    links = dict(conn.execute(
        "SELECT a.source_filing_id, o.source_filing_id FROM filings a "
        "LEFT JOIN filings o ON o.filing_id = a.amends_filing_id WHERE a.document_type = '4/A'"
    ).fetchall())
    assert links == {
        "0001104659-24-025044": "0001104659-24-022552",
        "0001104659-24-025045": "0001104659-24-024181",
        "0001104659-24-025046": "0001104659-24-024492",
    }
    assert conn.execute(
        "SELECT count(*) FROM filing_owners fo JOIN filings f USING (filing_id) "
        "WHERE f.source_filing_id = '0001104659-24-025044'"
    ).fetchone()[0] == 6


def test_amendment_does_not_link_to_another_owners_filing(conn):
    # Same issuer and period, but no owner in common: not an original.
    filings = {f.source_filing_id: f for f in parse_quarter(PLCE_ZIP)}
    original = filings["0001104659-24-022552"]
    original.owner_ciks, original.filer_key = ["999"], "999"
    store.save_filing(conn, original)
    store.save_filing(conn, filings["0001104659-24-025044"])
    assert store.link_amendments(conn) == 0
