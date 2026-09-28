import httpx
import pytest

from tradetracker.edgar import client as client_mod
from tradetracker.edgar.client import EdgarClient


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(client_mod.time, "sleep", lambda s: None)


def _client(responses):
    seen = []

    def handler(request):
        seen.append(request)
        r = responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    return EdgarClient("Trade Tracker t@example.com", transport=httpx.MockTransport(handler)), seen


def test_sends_declared_user_agent():
    c, seen = _client([httpx.Response(200, text="ok")])
    c.get("https://www.sec.gov/x")
    assert seen[0].headers["user-agent"] == "Trade Tracker t@example.com"


def test_retries_rate_limits_and_server_errors():
    c, seen = _client([httpx.Response(429), httpx.Response(503), httpx.Response(200, text="ok")])
    assert c.get("https://www.sec.gov/x").text == "ok"
    assert len(seen) == 3


def test_retries_read_timeouts():
    # The live getcurrent feed took 13-60s to answer when tested.
    c, seen = _client([httpx.ReadTimeout("slow"), httpx.Response(200, text="ok")])
    assert c.get("https://www.sec.gov/x").text == "ok"


def test_gives_up_after_retries():
    c, _ = _client([httpx.ReadTimeout("slow")] * 5)
    with pytest.raises(httpx.ReadTimeout):
        c.get("https://www.sec.gov/x", retries=4)
    c, _ = _client([httpx.Response(503)] * 3)
    with pytest.raises(httpx.HTTPStatusError):
        c.get("https://www.sec.gov/x", retries=2)


def test_does_not_retry_not_found():
    c, seen = _client([httpx.Response(404)])
    with pytest.raises(httpx.HTTPStatusError):
        c.get("https://www.sec.gov/x")
    assert len(seen) == 1
