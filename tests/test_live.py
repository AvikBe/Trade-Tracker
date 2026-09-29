"""Smoke tests against the real SEC and Tiingo services.

Skipped unless TT_LIVE=1 and the keys are set: `TT_LIVE=1 pytest -m live`.
"""

import os
from datetime import date

import pytest

from tradetracker.edgar import bulk, feed
from tradetracker.edgar.client import EdgarClient
from tradetracker.edgar.form4 import parse_form4

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.environ.get("TT_LIVE") != "1", reason="set TT_LIVE=1 to hit live APIs"),
]


@pytest.fixture(scope="module")
def edgar():
    ua = os.environ.get("EDGAR_USER_AGENT")
    if not ua:
        pytest.skip("EDGAR_USER_AGENT not set")
    c = EdgarClient(ua)
    yield c
    c.close()


def test_index_page_links_every_quarter_since_2015(edgar):
    links = bulk.index_links(edgar.get(bulk.INDEX_URL).text)
    today = date.today()
    last_full = (today.year, (today.month - 1) // 3) if today.month > 3 else (today.year - 1, 4)
    wanted = {(y, q) for y in range(2015, today.year + 1) for q in range(1, 5) if (y, q) < last_full}
    assert wanted - set(links) == set()


def test_live_feed_filings_parse(edgar):
    entries = feed.fetch_latest(edgar)
    assert entries, "feed returned no Form 4s"
    for e in entries[:3]:
        f = parse_form4(edgar.get(feed.ownership_xml_url(edgar, e)).content,
                        accession=e.accession, accepted_at=e.accepted_at)
        assert f.filer_key and f.issuer_cik


def test_tiingo_serves_a_delisted_ticker():
    from tradetracker.prices.tiingo import TiingoClient

    key = os.environ.get("TIINGO_API_KEY")
    if not key:
        pytest.skip("TIINGO_API_KEY not set")
    c = TiingoClient(key)
    bars = c.daily("AAN", date(2024, 9, 1))  # Aaron's, taken private in Oct 2024
    assert bars and bars[-1].date < date(2024, 12, 31)
