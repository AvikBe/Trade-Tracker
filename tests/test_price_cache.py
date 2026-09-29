import gzip
import json
from datetime import date, datetime, timedelta, timezone

import httpx
import pytest

from tradetracker.prices import cache as pc
from tradetracker.prices.loader import BENCHMARKS
from tradetracker.prices.tiingo import (
    AuthError,
    RateLimited,
    TickerNotFound,
    TiingoClient,
    parse_csv,
)

T0 = datetime(2026, 9, 29, 10, 0, tzinfo=timezone.utc)
CSV = (
    "date,open,high,low,close,volume,adjOpen,adjClose\n"
    "2023-01-03,10.0,10.5,9.8,10.2,1000,9.9,10.1\n"
    "2024-03-28,11.0,11.5,10.8,11.2,2000,10.9,11.1\n"
)
EMPTY = "date,open,high,low,close,volume,adjOpen,adjClose\n"
ORDER = ["2024q1", "2023q4", "2015q1"]


class Clock:
    def __init__(self, t=T0):
        self.t = t

    def __call__(self):
        self.t += timedelta(seconds=1)
        return self.t


class FakeClient:
    """Answers daily_csv from a script: ticker -> CSV text or an exception (or a list)."""

    def __init__(self, script=None, default=CSV):
        self.script = script or {}
        self.default = default
        self.calls = []

    def daily_csv(self, ticker, start):
        self.calls.append(ticker)
        r = self.script.get(ticker, self.default)
        if isinstance(r, list):
            r = r.pop(0) if len(r) > 1 else r[0]
        if isinstance(r, Exception):
            raise r
        return r


def universe(*rows):
    return [{"quarter": q, "ticker": t, "buys": b, "trades": n} for q, t, b, n in rows]


@pytest.fixture
def cache(tmp_path):
    c = pc.PriceCache(tmp_path / "prices")
    c.write_universe(
        universe(
            ("2023q4", "AAA", 1, 5),
            ("2024q1", "BBB", 2, 2),
            ("2024q1", "CCC", 9, 9),
            ("2024q1", "AAA", 0, 1),
            ("2015q1", "OLD", 3, 3),
            ("2019q2", "SKIP", 50, 50),  # quarter not in the order
        )
    )
    return c


def done_benchmarks(cache):
    cache.save_status({b: {"status": "ok", "first": "2014-01-02", "last": "2026-09-28",
                           "rows": 1, "attempts": 1, "last_attempt": T0.isoformat()}
                       for b in BENCHMARKS})


# ---- queue and budget ----------------------------------------------------------------
def test_queue_orders_benchmarks_quarters_then_buys(cache):
    q = pc.queue(cache.universe(), {}, ORDER, T0)
    assert q[: len(BENCHMARKS)] == BENCHMARKS
    assert q[len(BENCHMARKS):] == ["CCC", "BBB", "AAA", "OLD"]


def test_queue_groups_put_buy_tickers_before_sell_only_ones():
    u = universe(
        ("2024q1", "SELL1", 0, 40),
        ("2024q1", "BUY1", 1, 1),
        ("2023q4", "BUY2", 5, 5),
        ("2023q4", "SELL2", 0, 90),
        ("2023q4", "SELL1", 0, 1),
        ("2015q1", "OLDB", 7, 7),
        ("2015q1", "OLDS", 0, 70),
        ("2015q1", "SELL2", 2, 2),  # buys only in a later group: stays sell-only in group 1
    )
    order = pc.parse_order("2024q1,2023q4|2015q1")
    assert order == [["2024q1", "2023q4"], ["2015q1"]]
    q = pc.queue(u, {}, order, T0)[len(BENCHMARKS):]
    assert q == ["BUY1", "BUY2", "SELL1", "SELL2", "OLDB", "OLDS"]
    assert pc.flat(order) == ["2024q1", "2023q4", "2015q1"]


def test_queue_skips_done_and_waits_before_retrying_errors(cache):
    status = {
        "CCC": {"status": "ok"},
        "BBB": {"status": "not_found"},
        "AAA": {"status": "error", "last_attempt": (T0 - timedelta(minutes=10)).isoformat()},
        "OLD": {"status": "error", "last_attempt": (T0 - timedelta(hours=2)).isoformat()},
        **{b: {"status": "empty"} for b in BENCHMARKS},
    }
    assert pc.queue(cache.universe(), status, ORDER, T0) == ["OLD"]
    status["OLD"]["status"] = "failed"
    assert pc.queue(cache.universe(), status, ORDER, T0) == []


def test_budget_counts_hour_day_and_calendar_month_symbols():
    calls = [pc.Call(T0 - timedelta(minutes=m), f"T{m}", "ok") for m in range(0, 300, 10)]
    calls += [pc.Call(T0 - timedelta(days=40), "OLDMONTH", "ok")]  # previous month
    calls += [pc.Call(T0 - timedelta(minutes=5), "T0", "error")]  # repeat symbol
    b = pc.budget(calls, T0)
    assert b.hourly == pc.HOURLY_BUDGET - 7  # minutes 0..50 plus the repeat
    assert b.daily == pc.DAILY_BUDGET - 31
    assert b.symbols == pc.MONTHLY_SYMBOL_BUDGET - 30
    assert b.allows("T0")
    b.symbols = 0
    assert b.allows("T0") and not b.allows("NEW")


def test_month_boundary_resets_symbols():
    calls = [pc.Call(datetime(2026, 9, 30, 23, 0, tzinfo=timezone.utc), f"S{i}", "ok") for i in range(480)]
    assert pc.budget(calls, datetime(2026, 9, 30, 23, 30, tzinfo=timezone.utc)).symbols == 0
    b = pc.budget(calls, datetime(2026, 10, 1, 0, 5, tzinfo=timezone.utc))
    assert b.symbols == pc.MONTHLY_SYMBOL_BUDGET and b.daily == pc.DAILY_BUDGET - 480


# ---- the job -------------------------------------------------------------------------
def test_run_fetches_writes_bars_and_is_resumable(cache):
    done_benchmarks(cache)
    client = FakeClient()
    s1 = pc.run(cache, client, ORDER, now=Clock(), limit=2)
    assert client.calls == ["CCC", "BBB"] and s1["ok"] == 2 and s1["stopped"] == "limit"
    assert gzip.decompress(cache.bars_path("CCC").read_bytes()).decode() == CSV
    st = cache.status()["CCC"]
    assert (st["status"], st["rows"], st["first"], st["last"]) == ("ok", 2, "2023-01-03", "2024-03-28")
    s2 = pc.run(cache, client, ORDER, now=Clock())
    assert client.calls[2:] == ["AAA", "OLD"] and s2["stopped"] == "queue empty"
    assert s2["queued"] == 0
    assert len(cache.calls()) == 4
    assert not (cache.root / "LOCK").exists()
    assert "2024q1" in (cache.root / "coverage.txt").read_text()


def test_not_found_empty_and_errors(cache):
    done_benchmarks(cache)
    boom = httpx.ConnectError("reset")
    client = FakeClient({"CCC": TickerNotFound("CCC"), "BBB": EMPTY, "AAA": boom, "OLD": boom})
    clock = Clock()
    s = pc.run(cache, client, ORDER, now=clock)
    st = cache.status()
    assert st["CCC"]["status"] == "not_found" and st["BBB"]["status"] == "empty"
    assert st["AAA"]["status"] == "error" and "reset" in st["AAA"]["detail"]
    assert s["error"] == 2 and not cache.bars_path("BBB").exists()
    # errors wait an hour, then retry until MAX_ATTEMPTS
    assert pc.run(cache, client, ORDER, now=clock)["requested"] == 0
    for _ in range(pc.MAX_ATTEMPTS - 1):
        clock.t += pc.RETRY_AFTER + timedelta(minutes=1)
        pc.run(cache, client, ORDER, now=clock)
    assert cache.status()["AAA"]["status"] == "failed"
    assert cache.status()["AAA"]["attempts"] == pc.MAX_ATTEMPTS
    clock.t += timedelta(hours=5)
    assert pc.run(cache, client, ORDER, now=clock)["requested"] == 0


def test_error_then_success_on_retry(cache):
    done_benchmarks(cache)
    client = FakeClient({"CCC": [httpx.ReadTimeout("slow"), CSV]})
    clock = Clock()
    pc.run(cache, client, ORDER, now=clock)
    clock.t += timedelta(hours=2)
    pc.run(cache, client, ORDER, now=clock)
    st = cache.status()["CCC"]
    assert st["status"] == "ok" and st["attempts"] == 2


@pytest.mark.parametrize("kind", ["hourly", "daily", "symbols"])
def test_rate_limit_stops_and_cools_down(cache, kind):
    done_benchmarks(cache)
    client = FakeClient({"BBB": RateLimited(kind, f"Error: over your {kind} allocation")})
    clock = Clock()
    s = pc.run(cache, client, ORDER, now=clock)
    assert client.calls == ["CCC", "BBB"] and kind in s["stopped"]
    assert "BBB" not in cache.status()  # not marked; it is retried after the cooldown
    assert pc.run(cache, client, ORDER, now=clock)["stopped"].startswith("cooling down")
    assert len(client.calls) == 2
    clock.t += pc.COOLDOWN[kind] + timedelta(minutes=1)
    client.script = {}
    pc.run(cache, client, ORDER, now=clock)
    assert cache.status()["BBB"]["status"] == "ok"


def test_auth_error_stops_without_marking(cache):
    done_benchmarks(cache)
    s = pc.run(cache, FakeClient(default=AuthError("Invalid token")), ORDER, now=Clock())
    assert "API key" in s["stopped"] and cache.status().keys() == set(BENCHMARKS)


def test_budgets_stop_the_run(cache, monkeypatch):
    done_benchmarks(cache)
    monkeypatch.setattr(pc, "HOURLY_BUDGET", 3)
    client = FakeClient()
    s = pc.run(cache, client, ORDER, now=Clock())
    assert s["requested"] == 3 and s["stopped"] == "hourly/daily budget spent"
    monkeypatch.setattr(pc, "HOURLY_BUDGET", 45)
    monkeypatch.setattr(pc, "MONTHLY_SYMBOL_BUDGET", 3)
    s = pc.run(cache, client, ORDER, now=Clock(T0 + timedelta(hours=2)))
    assert s["requested"] == 0 and s["stopped"] == "monthly symbol budget spent"


def test_retry_of_a_symbol_used_this_month_needs_no_new_symbol(cache, monkeypatch):
    done_benchmarks(cache)
    client = FakeClient({"CCC": [httpx.ReadTimeout("slow"), CSV]})
    clock = Clock()
    monkeypatch.setattr(pc, "MONTHLY_SYMBOL_BUDGET", 1)
    pc.run(cache, client, ORDER, now=clock)
    clock.t += timedelta(hours=2)
    s = pc.run(cache, client, ORDER, now=clock)
    assert s["ok"] == 1 and client.calls == ["CCC", "CCC"]


def test_lock_blocks_overlap_but_expires(cache):
    cache.ensure()
    cache.acquire(T0)
    client = FakeClient()
    assert "another run" in pc.run(cache, client, ORDER, now=Clock())["stopped"]
    assert client.calls == []
    pc.run(cache, client, ORDER, now=Clock(T0 + pc.LOCK_STALE + timedelta(minutes=1)))
    assert client.calls


def test_status_survives_a_crash_mid_run(cache):
    done_benchmarks(cache)

    class Crash(Exception):
        pass

    client = FakeClient({"BBB": Crash()})
    with pytest.raises(Crash):
        pc.run(cache, client, ORDER, now=Clock())
    assert cache.status()["CCC"]["status"] == "ok"
    assert not (cache.root / "LOCK").exists()
    client.script = {}
    pc.run(cache, client, ORDER, now=Clock(T0 + timedelta(minutes=5)))
    assert client.calls == ["CCC", "BBB", "BBB", "AAA", "OLD"]


def test_state_files_are_valid_json(cache):
    pc.run(cache, FakeClient(), ORDER, now=Clock())
    assert json.loads((cache.root / "status.json").read_text())
    assert not list(cache.root.rglob("*.tmp"))


# ---- coverage ------------------------------------------------------------------------
def test_quarter_bounds():
    assert pc._quarter_bounds("2023q4") == ("2023-10-01", "2023-12-31")
    assert pc._quarter_bounds("2024q1") == ("2024-01-01", "2024-03-31")
    assert pc._quarter_bounds("2018q3") == ("2018-07-01", "2018-09-30")


def test_coverage_counts_out_of_range_bars(cache):
    status = {
        "CCC": {"status": "ok", "first": "2014-01-02", "last": "2026-09-28"},
        "BBB": {"status": "ok", "first": "2024-06-01", "last": "2026-09-28"},  # listed later
        "AAA": {"status": "not_found"},
        "OLD": {"status": "ok", "first": "2014-01-02", "last": "2014-12-31"},  # delisted early
    }
    cache.save_status(status)
    cov = {c["quarter"]: c for c in pc.coverage(cache, ORDER)}
    assert cov["2024q1"]["covered"] == 1 and cov["2024q1"]["out_of_range"] == 1
    assert cov["2024q1"]["buys_covered"] == 9 and cov["2024q1"]["buys"] == 11
    assert cov["2024q1"]["not_found"] == 1
    assert cov["2015q1"]["covered"] == 0 and cov["2015q1"]["out_of_range"] == 1
    assert "2019q2" not in cov


# ---- Tiingo client -------------------------------------------------------------------
def client_for(status, body, seen=None):
    def handler(request):
        if seen is not None:
            seen.append(request)
        return httpx.Response(status, text=body)

    return TiingoClient("k", transport=httpx.MockTransport(handler))


def test_daily_csv_requests_columns_and_share_class():
    seen = []
    assert client_for(200, CSV, seen).daily_csv("BRK.B", date(2014, 1, 1)) == CSV
    url = seen[0].url
    assert url.path.endswith("/daily/brk-b/prices")
    assert url.params["format"] == "csv" and url.params["startDate"] == "2014-01-01"
    assert url.params["columns"].startswith("date,") and "adjClose" in url.params["columns"]


@pytest.mark.parametrize(
    "status,body,exc,kind",
    [
        (200, "Error: Ticker 'ZZQXQ' not found", TickerNotFound, None),  # real CSV reply
        (404, '{"detail":"Error: Ticker \'ZZQXQ\' not found"}', TickerNotFound, None),
        (429, '{"detail":"Error: You have run over your hourly request allocation."}', RateLimited, "hourly"),
        (200, "Error: You have run over your daily request allocation.", RateLimited, "daily"),
        (429, "Error: You have run over your 500 symbol look up for this month.", RateLimited, "symbols"),
        (429, "", RateLimited, "hourly"),
        (403, "None", AuthError, None),  # real reply to a bad key
        (401, '{"detail":"Invalid token."}', AuthError, None),
        (500, "oops", httpx.HTTPStatusError, None),
        (200, "<html>maintenance</html>", httpx.HTTPStatusError, None),
    ],
)
def test_daily_csv_errors(status, body, exc, kind):
    with pytest.raises(exc) as e:
        client_for(status, body).daily_csv("ZZQXQ", date(2014, 1, 1))
    if kind:
        assert e.value.kind == kind


def test_daily_csv_empty_list_means_no_bars():
    # Tiingo's real reply for a known symbol with no bars since the start date (KFS).
    text = client_for(200, "[]").daily_csv("KFS", date(2014, 1, 1))
    assert text.startswith("date,") and parse_csv("KFS", text) == []


def test_json_daily_still_raises_not_found_and_limits():
    with pytest.raises(TickerNotFound):
        client_for(200, '{"detail":"Error: Ticker \'X\' not found"}').daily("X", date(2024, 1, 1))
    with pytest.raises(RateLimited):
        client_for(429, '{"detail":"Error: You have run over your hourly request allocation."}').daily(
            "X", date(2024, 1, 1)
        )


def test_parse_csv_matches_json_parse():
    bars = parse_csv("AAA", CSV)
    assert len(bars) == 2 and bars[0].date == date(2023, 1, 3)
    assert str(bars[1].adj_close) == "11.1" and bars[1].volume == 2000
    assert parse_csv("AAA", EMPTY) == []


# ---- database bridges ----------------------------------------------------------------
def test_build_universe_and_import(conn, tmp_path):
    from bulkfixture import write
    from tradetracker import store
    from tradetracker.edgar.bulk import parse_quarter

    for f in parse_quarter(write(tmp_path / "q.zip")):
        store.save_filing(conn, f)
    conn.execute("UPDATE trades SET ticker = 'NYSE: BAD' WHERE ticker = 'OTHR'")
    conn.commit()
    rows = pc.build_universe(conn)
    assert rows and all(r["quarter"] == "2024q1" for r in rows)
    assert "NYSE: BAD" not in {r["ticker"] for r in rows}
    ex = next(r for r in rows if r["ticker"] == "EXWD")
    assert ex["trades"] >= ex["buys"] >= 1

    cache = pc.PriceCache(tmp_path / "prices")
    cache.write_universe(rows)
    assert cache.universe() == rows
    cache.write_bars("EXWD", CSV)
    cache.save_status({
        "EXWD": {"status": "ok", "rows": 2, "first": "2023-01-03", "last": "2024-03-28",
                 "attempts": 1, "last_attempt": T0.isoformat()},
        "GONE": {"status": "empty", "attempts": 1, "last_attempt": T0.isoformat()},
        "BADX": {"status": "failed", "attempts": 3, "last_attempt": T0.isoformat(), "detail": "x"},
    })
    s = pc.import_to_db(cache, conn)
    assert s["tickers"] == 1 and s["bars"] == 2
    again = pc.import_to_db(cache, conn)  # idempotent
    assert again["bars"] == 2
    assert conn.execute("SELECT count(*) FROM daily_prices WHERE ticker = 'EXWD'").fetchone()[0] == 2
    row = conn.execute(
        "SELECT close, adj_close, volume FROM daily_prices WHERE ticker = 'EXWD' AND date = '2024-03-28'"
    ).fetchone()
    assert (str(row[0]), str(row[1]), row[2]) == ("11.2", "11.1", 2000)
    st = dict(conn.execute("SELECT ticker, status FROM price_status").fetchall())
    assert st == {"EXWD": "ok", "GONE": "not_found", "BADX": "error"}
