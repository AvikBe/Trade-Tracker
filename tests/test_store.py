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
