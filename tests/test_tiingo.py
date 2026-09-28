from datetime import date
from decimal import Decimal

import httpx
import pytest

from tradetracker.prices.tiingo import TickerNotFound, TiingoClient

ROWS = [
    {"date": "2024-03-04T00:00:00.000Z", "open": 12.0, "high": 12.6, "low": 11.9, "close": 12.4,
     "volume": 120000, "adjOpen": 11.8, "adjHigh": 12.4, "adjLow": 11.7, "adjClose": 12.2,
     "adjVolume": 120000, "divCash": 0.0, "splitFactor": 1.0},
]


def test_daily_parses_and_uses_dash_share_class():
    seen = {}

    def handler(request: httpx.Request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        return httpx.Response(200, json=ROWS)

    c = TiingoClient("k", transport=httpx.MockTransport(handler))
    bars = c.daily("BRK.B", date(2024, 1, 1))
    assert "/daily/brk-b/prices" in seen["url"] and "startDate=2024-01-01" in seen["url"]
    assert seen["auth"] == "Token k"
    assert bars[0].date == date(2024, 3, 4)
    assert bars[0].adj_close == Decimal("12.2") and bars[0].volume == 120000


def test_daily_not_found():
    c = TiingoClient("k", transport=httpx.MockTransport(lambda r: httpx.Response(404, json={})))
    with pytest.raises(TickerNotFound):
        c.daily("GONE", date(2024, 1, 1))
