"""Paper trading: EDGAR daily-index ingest, live scoring, brokers, the book and its report."""

import gzip
import json
from datetime import date, timedelta
from pathlib import Path

import httpx
import pytest
from test_scoring import bars_from, days_from, ev

from tradetracker.backtest import prices as bp
from tradetracker.backtest.returns import Outcome, evaluate as bt_evaluate
from tradetracker.cli import main as cli_main
from tradetracker.features.calendar import NYSE
from tradetracker.fundamentals import SharesTable
from tradetracker.paper import broker as B
from tradetracker.paper import engine as E
from tradetracker.paper import ingest as I
from tradetracker.paper import report as R
from tradetracker.paper import run as RUN
from tradetracker.paper import signals as PS
from tradetracker.scoring import model as M
from tradetracker.scoring import signals as S

REPO = Path(__file__).resolve().parents[1]
FIX = Path(__file__).parent / "fixtures" / "real" / "paper"
MODEL = M.Model.from_json((REPO / "models" / "two-inputs-60-days-cap-floor.json").read_text())


# ---------------------------------------------------------------- ingest: real EDGAR files

def real_index() -> str:
    return gzip.open(FIX / "form.20261013.idx.gz", "rt").read()


def test_parse_real_daily_index():
    rows = I.parse_index(real_index())
    # The day's index lists 4s and 4/As once per party (issuer and owner); one row per filing.
    assert len(rows) == len({r.accession for r in rows}) == 522
    assert {r.form for r in rows} == {"4", "4/A"}
    assert all(r.filed == date(2026, 10, 13) for r in rows)
    imsr = next(r for r in rows if r.accession == "0001193125-26-420002")
    assert imsr.path.endswith("0001193125-26-420002.txt") and imsr.form == "4"
    # Other form types (3, 10-K) in the same file are skipped.
    assert not [r for r in rows if r.form not in ("4", "4/A")]


def test_parse_index_edge_rows():
    text = ("Form Type   Company Name   CIK   Date Filed  File Name\n" + "-" * 80 + "\n"
            "4                A  B  C  CORP  4 / A                        123     20261013    edgar/data/123/0000000123-26-000001.txt\n"
            "4/A              X                                           9       20261013    edgar/data/9/0000000009-26-000002.txt\n"
            "4                X                                           9       20261013    edgar/data/77/0000000009-26-000002.txt\n"
            "424B2            Not a Form 4                                5       20261013    edgar/data/5/0000000005-26-000003.txt\n"
            "4                broken row\n"
            "\n")
    rows = I.parse_index(text)
    # A filing listed twice (once per party) is kept once, as first listed.
    assert [(r.form, r.accession, r.cik) for r in rows] == [("4/A", "0000000009-26-000002", "9"),
                                                            ("4", "0000000123-26-000001", "123")]


def entry_for(acc: str, form: str = "4") -> I.IndexEntry:
    return I.IndexEntry(form, "1", date(2026, 10, 13), f"edgar/data/1/{acc}.txt")


def test_parse_real_submissions():
    imsr = I.parse_submission((FIX / "0001193125-26-420002.txt").read_bytes(),
                              entry_for("0001193125-26-420002"))
    assert imsr.issuer_ticker == "IMSR" and imsr.role == "director"
    assert imsr.accepted_at.isoformat() == "2026-10-13T11:06:12-04:00"
    assert imsr.filed_at.date() == date(2026, 10, 13)
    [t] = imsr.trades
    assert (t.side, t.txn_code, float(t.shares), float(t.price)) == ("buy", "P", 12000, 3.278)
    assert not t.is_10b5_1
    # A 10b5-1 buy plan, many lines.
    api = I.parse_submission((FIX / "0001193125-26-419691.txt").read_bytes(), entry_for("0001193125-26-419691"))
    assert api.issuer_ticker == "API" and len(api.trades) == 18 and all(t.is_10b5_1 for t in api.trades)
    # An amendment and a sale.
    a = I.parse_submission((FIX / "0001104659-26-115304.txt").read_bytes(), entry_for("0001104659-26-115304", "4/A"))
    assert a.document_type == "4/A"
    s = I.parse_submission((FIX / "0001437749-26-032571.txt").read_bytes(), entry_for("0001437749-26-032571", "4/A"))
    assert [t.side for t in s.trades] == ["sell"]


def test_parse_submission_without_xml():
    with pytest.raises(ValueError, match="no ownership XML"):
        I.parse_submission(b"<SEC-DOCUMENT>\n<TYPE>4\n<TEXT>nothing</TEXT>", entry_for("x"))


class FakeEdgar:
    """Serves the real fixtures by URL; anything else is a 404."""

    def __init__(self, pages: dict[str, bytes]):
        self.pages, self.calls = pages, []

    def get(self, url):
        self.calls.append(url)
        if url not in self.pages:
            req = httpx.Request("GET", url)
            raise httpx.HTTPStatusError("404", request=req, response=httpx.Response(404, request=req))
        return httpx.Response(200, content=self.pages[url], request=httpx.Request("GET", url))


def fake_day_edgar(broken: bool = False) -> FakeEdgar:
    d = date(2026, 10, 13)
    accs = ["0001193125-26-420002", "0001193125-26-419691", "0001104659-26-115304",
            "0001437749-26-032571", "0001829340-26-000011"]
    lines = [r for r in real_index().splitlines()]
    head = [line for line in lines if not line.startswith(("4 ", "4/A"))]
    rows = [line for line in lines if any(a in line for a in accs)]
    pages = {I.index_url(d): "\n".join(head + rows).encode()}
    for line in rows:
        path = line.split()[-1]
        acc = path.rsplit("/", 1)[1][:-4]
        pages[I.ARCHIVE_URL.format(path=path)] = (FIX / f"{acc}.txt").read_bytes()
    if broken:
        k = next(k for k in pages if "420002" in k)
        pages[k] = b"garbage"
    return FakeEdgar(pages)


def test_fetch_day_caches_and_counts(tmp_path):
    ed = fake_day_edgar()
    data = I.fetch_day(ed, date(2026, 10, 13), tmp_path)
    assert data["counts"] == {"index_rows": 5, "parsed": 5, "errors": 0, "with_trades": 5}
    calls = len(ed.calls)
    again = I.fetch_day(ed, date(2026, 10, 13), tmp_path)
    assert again == data and len(ed.calls) == calls          # served from the cache
    assert I.fetch_day(ed, date(2026, 10, 14), tmp_path) is None   # index not out yet
    assert not (tmp_path / "2026-10-14.json").exists()


def test_fetch_day_survives_a_bad_filing(tmp_path):
    data = I.fetch_day(fake_day_edgar(broken=True), date(2026, 10, 13), tmp_path)
    assert data["counts"]["errors"] == 1 and data["counts"]["parsed"] == 4
    assert data["errors"][0]["accession"] == "0001193125-26-420002"


def test_events_from_real_day(tmp_path):
    data = I.fetch_day(fake_day_edgar(), date(2026, 10, 13), tmp_path)
    ex = {}
    evs = I.build_events(data["filings"], exclusions=ex)
    # IMSR and FEAM buys, API's 10b5-1 buys (kept, hidden later), the 4/A buy left out,
    # and the 4/A sale ignored (sales are never traded).
    assert sorted(le.event.ticker for le in evs) == ["API", "FEAM", "IMSR"]
    assert ex == {"amendment": 1}
    imsr = next(le for le in evs if le.event.ticker == "IMSR").event
    assert imsr.filing_date == date(2026, 10, 13) and imsr.first_trade_date == date(2026, 10, 9)
    assert imsr.avg_price == pytest.approx(3.278) and imsr.shares_held_before == pytest.approx(18669 - 12000)
    assert next(le for le in evs if le.event.ticker == "API").event.is_10b5_1


def filing(acc, filer="7", ticker="ABC", form="4", filed="2026-10-13", trades=None, cik="100"):
    return {"accession": acc, "form": form, "filer": filer, "filer_name": f"F{filer}", "role": "CEO",
            "issuer_cik": cik, "issuer_name": "Abc Inc", "ticker": ticker, "filed": filed,
            "accepted_at": None, "url": "", "trades": trades or [trade()]}


def trade(side="buy", td="2026-10-09", shares="100", price="10", after="1100", owner="self", plan=False):
    return {"line": 0, "side": side, "code": "P" if side == "buy" else "S", "trade_date": td,
            "shares": shares, "price": price, "owned_after": after, "owner": owner, "10b5_1": plan}


def test_build_events_groups_like_the_backtest():
    fs = [
        filing("a1", trades=[trade(td="2026-10-08", shares="100", price="10", after="1100"),
                             trade(td="2026-10-09", shares="50", price="12", after="1150"),
                             trade(owner="trust", shares="10", after="20")]),
        filing("a2", trades=[trade(td="2026-10-07", shares="40", price="9", after="1190")]),  # same insider, same day
        filing("b1", filer="8"),                                   # another insider: own event
        filing("c1", ticker=None, cik="555"),                      # ticker from the SEC map
        filing("d1", ticker=None, cik="556"),                      # no ticker at all
        filing("e1", trades=[trade(td="2026-10-14")]),             # trade after filing
        filing("f1", trades=[trade(side="sell")]),                 # sale only
        filing("g1", filer="9", trades=[trade(plan=True), trade()]),  # one 10b5-1 line taints the event
        filing("h1", filer="10", trades=[trade(after=None)]),      # holdings not reported
    ]
    ex = {}
    evs = I.build_events(fs, tickers={"555": "MAPPED"}, exclusions=ex)
    assert ex == {"no_ticker": 1, "trade_after_filing": 1}
    by = {(le.filer_cik, le.event.ticker): le for le in evs}
    e = by[("7", "ABC")].event
    assert by[("7", "ABC")].accessions == ["a1", "a2"]
    assert e.first_trade_date == date(2026, 10, 7) and e.last_trade_date == date(2026, 10, 9)
    assert e.value == pytest.approx(100 * 10 + 50 * 12 + 10 * 10 + 40 * 9)
    assert e.shares == pytest.approx(200)
    # Per account, the holding before the first buy: self min(1000, 1100, 1150) and trust 10.
    assert e.held_before == {"self": 1000.0, "trust": 10.0}
    assert ("7", "MAPPED") in by and by[("9", "ABC")].event.is_10b5_1
    assert by[("10", "ABC")].event.shares_held_before is None
    assert ("8", "ABC") in by and len(evs) == 5


# ---------------------------------------------------------------- live scoring

ENTRY = date(2026, 6, 1)


def history(closes, end=ENTRY, volume=1e6):
    """Bars ending the session before `end`."""
    n = len(closes)
    start = end
    for _ in range(n):
        start = start - timedelta(days=1)
        while not NYSE.is_business_day(start):
            start -= timedelta(days=1)
    return bars_from(start, closes, volume=volume)


def shares_table(cik="1", n=1e8, day=date(2025, 1, 2)):
    return SharesTable([(cik, day, n)])


def live(**kw) -> I.LiveEvent:
    base = dict(filing_date=date(2026, 5, 29),
                first_trade_date=date(2026, 5, 27), last_trade_date=date(2026, 5, 27),
                value=10_000.0, shares=1000.0)
    base.update(kw)
    return I.LiveEvent(ev(**base), ["acc"], "1", "Doe", "X Inc")


def deep_fall(n=300, top=30.0, now=10.0):
    """A stock that falls from `top` to `now` over n sessions (down 67%)."""
    return [top - (top - now) * i / (n - 1) for i in range(n)]


def test_live_candidate_deep_drawdown_no_runup():
    closes = deep_fall()
    bars = history(closes)
    # The trade-date close equals today's (no rise since the trade), Form 4 price 10.
    d = PS.evaluate(live(), bars, ENTRY, MODEL, shares_table(), 50e6)
    assert d.status == "candidate", d.status
    assert d.signals.drawdown == pytest.approx(10 / 30 - 1 + 0, abs=0.08)
    assert d.signals.drift <= 0 and d.score >= MODEL.threshold
    assert d.signals.market_cap == pytest.approx(1e8 * 10.0)
    assert d.last_close == pytest.approx(10.0)


def test_live_ignores_bars_on_or_after_entry():
    closes = deep_fall()
    bars = history(closes)
    d1 = PS.evaluate(live(), bars, ENTRY, MODEL, shares_table(), 50e6)
    # Add an absurd bar on entry day and after: nothing may change.
    extra = bp.Bars(bars.bars + [bp.Bar(ENTRY, 99, 99, 99, 99, 1e9),
                                 bp.Bar(NYSE.add(ENTRY, 1), 1, 1, 1, 1, 1)])
    d2 = PS.evaluate(live(), extra, ENTRY, MODEL, shares_table(), 50e6)
    assert (d1.status, d1.score, d1.signals.drawdown, d1.signals.drift, d1.signals.dollar_volume,
            d1.signals.market_cap) == (d2.status, d2.score, d2.signals.drawdown, d2.signals.drift,
                                       d2.signals.dollar_volume, d2.signals.market_cap)


@pytest.mark.parametrize("case,status", [
    ("10b5-1", "10b5-1"), ("no_bars", "no_prices"), ("stale", "no_prices"),
    ("penny", "illiquid"), ("thin", "illiquid"), ("runup", "drift"), ("mismatch", "ticker_mismatch"),
    ("small", "small_cap"), ("no_cap", "no_market_cap"), ("near_high", "below_cutoff"),
    ("small_rise", "below_cutoff"),
])
def test_live_hide_rules(case, status):
    closes = deep_fall()
    kw, shares, bars = {}, shares_table(), None
    if case == "10b5-1":
        kw["is_10b5_1"] = True
    if case == "penny":
        closes = [c / 6 for c in closes]                 # last close $1.67
        kw["value"] = 10_000.0 / 6
    if case == "mismatch":
        kw["value"] = 30_000.0                            # Form 4 price $30 vs a $10 close
    if case == "small":
        shares = shares_table(n=1e6)                      # $10M cap
    if case == "no_cap":
        shares = shares_table(cik="999")
    if case == "near_high":
        closes = [10.0] * 300
    if case == "runup":
        closes = closes[:-3] + [10.0, 12.0, 12.5]         # trade at 10, now 12.5 (+25%)
        kw.update(first_trade_date=history(closes).dates[-3], last_trade_date=history(closes).dates[-3])
    if case == "small_rise":
        closes = closes[:-3] + [10.0, 10.1, 10.2]         # +2% since the trade: penalty bites
        kw.update(first_trade_date=history(closes).dates[-3], last_trade_date=history(closes).dates[-3])
    bars = history(closes, volume=1e3 if case == "thin" else 1e6)
    if case == "no_bars":
        bars = None
    if case == "stale":
        bars = history(closes, end=ENTRY - timedelta(days=20))
    d = PS.evaluate(live(**kw), bars, ENTRY, MODEL, shares, 50e6)
    assert d.status == status


def test_live_matches_backtest_evaluate():
    """The paper trader's pre-entry inputs are the backtest's, bar for bar."""
    closes = deep_fall(400, top=50, now=12) + [11.5, 11.0, 11.2]
    full = history(closes + [11.3] * 80, end=NYSE.add(ENTRY, 80))
    entry = full.dates[403]
    td = full.dates[400]
    e = ev(first_trade_date=td, last_trade_date=td, filing_date=full.dates[402], value=11_000.0,
           shares=1000.0)
    spy = history([100.0] * 600, end=NYSE.add(ENTRY, 120))
    o = bt_evaluate(e, full, spy, None, "SPY", None)
    assert o.status == "ok" and o.entry == entry
    d = PS.evaluate(I.LiveEvent(e, [], "1", "", None), full, entry, MODEL, shares_table(), 50e6)

    class P:
        def get(self, t):
            return full
    assert d.signals.drift == pytest.approx(o.drift_pct / 100, abs=1e-12)
    assert d.signals.dollar_volume == pytest.approx(o.dollar_volume)
    assert d.cost == pytest.approx(o.cost)
    assert d.signals.drawdown == pytest.approx(S.drawdown(P(), "X", entry))


def test_best_per_ticker():
    def dec(t, score, status="candidate"):
        d = PS.Decision(live(ticker=t), ENTRY, status)
        d.score = score
        return d
    out = PS.best_per_ticker([dec("A", 1.2), dec("A", 1.5), dec("B", 1.3), dec("C", 9, "drift")])
    assert [(d.ticker, d.score) for d in out] == [("A", 1.5), ("B", 1.3)]


def test_model_cutoff_is_the_deep_drawdown_bucket():
    """The committed model: weight on drawdown only, beta 8, cutoff 1.0, so a buy is a
    candidate exactly when the stock is 45%+ below its high and hasn't risen since."""
    s = lambda dd, drift: S.Signals(None, 0, None, 1, 0, dd, None, 0, 5e6, drift)  # noqa: E731
    assert MODEL.weights == {"drawdown": 1.0} and MODEL.beta == 8.0 and MODEL.threshold == 1.0
    assert MODEL.score(s(-0.5, 0.0)) >= 1.0
    assert MODEL.score(s(-0.5, 0.001)) < 1.0
    assert MODEL.score(s(-0.44, -0.2)) < 1.0
    assert MODEL.score(s(-0.9, -0.3)) == pytest.approx(MODEL.score(s(-0.9, -0.2)))  # drift floor


# ---------------------------------------------------------------- brokers

def alpaca(handler):
    return B.AlpacaPaper("k", "s", transport=httpx.MockTransport(handler))


def test_alpaca_refuses_live_urls_and_missing_keys(monkeypatch):
    with pytest.raises(B.BrokerError, match="only Alpaca's paper API"):
        B.AlpacaPaper("k", "s", base_url="https://api.alpaca.markets")
    with pytest.raises(B.BrokerError, match="incomplete"):
        B.AlpacaPaper("k", "")
    monkeypatch.setenv("APCA_API_BASE_URL", "https://api.alpaca.markets")
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
    with pytest.raises(B.BrokerError):
        B.AlpacaPaper.from_env()
    monkeypatch.setenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets/")
    assert B.AlpacaPaper.from_env().kind == "alpaca"


def test_alpaca_without_keys_leaves_auth_to_the_proxy(monkeypatch):
    # Network secrets: no keys in the session, the agent proxy adds the headers.
    for v in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY", "APCA_API_KEY_ID", "APCA_API_SECRET_KEY",
              "APCA_API_BASE_URL"):
        monkeypatch.delenv(v, raising=False)
    seen = []

    def handler(req):
        seen.append(req)
        return httpx.Response(200, json={"account_number": "PA1", "status": "ACTIVE", "equity": "100000"})
    a = B.AlpacaPaper.from_env()
    a.http = httpx.Client(base_url=B.PAPER_URL, transport=httpx.MockTransport(handler))
    assert a.proxy_auth
    ok, why = a.ready()
    assert ok and "PA1" in why
    assert "apca-api-key-id" not in seen[0].headers and "apca-api-secret-key" not in seen[0].headers
    keyed = alpaca(handler)
    keyed.ready()
    assert seen[1].headers["apca-api-key-id"] == "k" and seen[1].headers["apca-api-secret-key"] == "s"


def test_alpaca_ready_reports_missing_auth():
    a = alpaca(lambda req: httpx.Response(401, json={"message": "unauthorized."}))
    ok, why = a.ready()
    assert not ok and "401" in why
    with pytest.raises(B.BrokerError):
        alpaca(lambda req: httpx.Response(404, text="nope")).ready()


def test_cli_alpaca_check_exit_codes(monkeypatch, capsys):
    from tradetracker import cli
    monkeypatch.setattr(B.AlpacaPaper, "ready", lambda self: (False, "not authorized (HTTP 401)"))
    assert cli.main(["alpaca-check"]) == 3
    assert "not ready" in capsys.readouterr().out
    monkeypatch.setattr(B.AlpacaPaper, "ready", lambda self: (True, "account PA1 ACTIVE"))
    assert cli.main(["alpaca-check"]) == 0


def order_json(cid, status="accepted", tif="opg", **kw):
    return {"id": "oid-" + cid, "client_order_id": cid, "status": status, "time_in_force": tif,
            "filled_qty": kw.get("filled_qty", "0"), "filled_avg_price": kw.get("price"),
            "filled_at": kw.get("filled_at")}


def test_alpaca_submit_sends_a_paper_market_order():
    seen = []

    def handler(req):
        seen.append(req)
        assert req.url.host == "paper-api.alpaca.markets"
        assert req.headers["APCA-API-KEY-ID"] == "k" and req.headers["APCA-API-SECRET-KEY"] == "s"
        if req.method == "GET":
            return httpx.Response(404, json={"message": "not found"})
        body = json.loads(req.content)
        assert body == {"symbol": "IMSR", "qty": "1607", "side": "buy", "type": "market",
                        "time_in_force": "opg", "client_order_id": "tt-2026-10-14-IMSR-buy"}
        return httpx.Response(200, json=order_json(body["client_order_id"]))
    f = alpaca(handler).submit("IMSR", 1607, "buy", "opg", "tt-2026-10-14-IMSR-buy", date(2026, 10, 14))
    assert f.status == "accepted" and f.broker_id == "oid-tt-2026-10-14-IMSR-buy" and f.tif == "opg"
    assert [r.method for r in seen] == ["GET", "POST"]
    assert seen[0].url.path == "/v2/orders:by_client_order_id"


def test_alpaca_submit_is_idempotent():
    def handler(req):
        assert req.method == "GET", "must not post an order that already exists"
        return httpx.Response(200, json=order_json("cid", status="filled", filled_qty="10", price="3.3"))
    f = alpaca(handler).submit("X", 10, "buy", "opg", "cid", date(2026, 10, 14))
    assert f.status == "filled" and f.filled_qty == 10 and f.filled_price == 3.3


def test_alpaca_auction_refused_falls_back_to_day():
    posts = []

    def handler(req):
        if req.method == "GET":
            return httpx.Response(404, json={})
        body = json.loads(req.content)
        posts.append(body)
        if body["time_in_force"] == "opg":
            return httpx.Response(422, json={"message": "opg orders must be submitted before 9:28"})
        return httpx.Response(200, json=order_json(body["client_order_id"], tif="day"))
    f = alpaca(handler).submit("X", 5, "buy", "opg", "cid", date(2026, 10, 14))
    assert [p["time_in_force"] for p in posts] == ["opg", "day"]
    assert posts[1]["client_order_id"] == "cid-day"
    assert f.tif == "day" and "opg refused" in f.note and f.status == "accepted"


def test_alpaca_rejection_and_lookups():
    def handler(req):
        p = req.url.path
        if p == "/v2/orders:by_client_order_id":
            cid = req.url.params["client_order_id"]
            if cid == "cid-day":
                return httpx.Response(200, json=order_json(cid, status="filled", tif="day",
                                                           filled_qty="5", price="2.5",
                                                           filled_at="2026-10-14T13:30:02Z"))
            return httpx.Response(404, json={})
        if p == "/v2/orders":
            return httpx.Response(403, json={"message": "insufficient buying power"})
        if p == "/v2/positions/GONE":
            return httpx.Response(404, json={})
        if p == "/v2/positions/X":
            return httpx.Response(200, json={"qty": "15"})
        if p == "/v2/assets/X":
            return httpx.Response(200, json={"tradable": True, "status": "active"})
        if p == "/v2/assets/OTC":
            return httpx.Response(200, json={"tradable": False, "status": "active"})
        if p == "/v2/assets/NOPE":
            return httpx.Response(404, json={})
        if p == "/v2/account":
            return httpx.Response(200, json={"equity": "100123.45"})
        raise AssertionError(p)
    a = alpaca(handler)
    f = a.submit("X", 5, "buy", "day", "new", date(2026, 10, 14))
    assert f.status == "rejected" and "insufficient" in f.note
    # A refreshed order found under its day-order fallback id.
    r = a.refresh("cid")
    assert r.status == "filled" and r.filled_qty == 5 and r.filled_at.startswith("2026-10-14")
    assert a.refresh("nothing").status == "unknown"
    assert a.position_qty("GONE") == 0 and a.position_qty("X") == 15
    assert a.tradable("X") and not a.tradable("OTC") and not a.tradable("NOPE")
    assert a.equity() == pytest.approx(100123.45)


def test_alpaca_retries_rate_limits(monkeypatch):
    monkeypatch.setattr(B.time, "sleep", lambda s: None)
    n = {"calls": 0}

    def handler(req):
        n["calls"] += 1
        if n["calls"] < 3:
            return httpx.Response(429, json={})
        return httpx.Response(200, json={"equity": "5"})
    assert alpaca(handler).equity() == 5 and n["calls"] == 3


def test_dry_run_fills_from_bars():
    s = date(2026, 10, 14)
    bars = bp.Bars([bp.Bar(s, 3.0, 3.5, 3.0, 3.5, 1e6)])
    dr = B.DryRun(lambda t: bars if t == "X" else None)
    assert dr.submit("X", 10, "buy", "opg", "c", s).status == "pending"
    assert dr.refresh("c", symbol="X", session=s, tif="opg", today=s, qty=10).filled_price == 3.0
    assert dr.refresh("c", symbol="X", session=s, tif="cls", today=s, qty=10).filled_price == 3.5
    f = dr.refresh("c", symbol="X", session=NYSE.add(s, 1), tif="opg", today=NYSE.add(s, 1), qty=10)
    assert f.status == "pending"
    f = dr.refresh("c", symbol="Y", session=s, tif="opg", today=NYSE.add(s, 6), qty=10)
    assert f.status == "expired"


# ---------------------------------------------------------------- the book

class Prices:
    def __init__(self, bars):
        self.bars = bars

    def get(self, t):
        return self.bars.get(t)


def market(tickers, start=date(2025, 1, 2), sessions=520, fall=True, drift_up=()):
    """Bars for SPY and each ticker: each falls 67% over its first 300 sessions, then
    drifts up 0.1% a session (so it opens a hair above the prior close). SPY rises
    0.05% a session."""
    ds = days_from(start, sessions)
    out = {"SPY": bp.Bars([bp.Bar(d, 100 * 1.0005 ** i, 100 * 1.0005 ** i,
                                  100 * 1.0005 ** i, 100 * 1.0005 ** i, 1e8) for i, d in enumerate(ds)])}
    for t in tickers:
        bs = []
        for i, d in enumerate(ds):
            c = 30 - 20 * i / 299 if i < 300 else 10 * 1.001 ** (i - 299)
            if not fall:
                c = 10.0
            bs.append(bp.Bar(d, c, c, c, c, 1e6))
        out[t] = bp.Bars(bs)
    return out, ds


def write_day(cache: Path, d: date, filings: list[dict]):
    cache.mkdir(parents=True, exist_ok=True)
    (cache / f"{d.isoformat()}.json").write_text(json.dumps(
        {"version": I.CACHE_VERSION, "date": d.isoformat(), "filings": filings, "errors": [],
         "counts": {"index_rows": len(filings), "parsed": len(filings), "errors": 0,
                    "with_trades": len(filings)}}))


def buy_filing(ticker, filer, d, price=None, cik=None, plan=False):
    """A buy traded on the filing date at the day's close (no drift)."""
    return filing(f"{ticker}-{filer}-{d}", filer=str(filer), ticker=ticker, filed=d.isoformat(),
                  cik=cik or "1", trades=[trade(td=d.isoformat(), shares="100",
                                                price=str(price or 1), plan=plan)])


class NoEdgar:
    def get(self, url):
        req = httpx.Request("GET", url)
        raise httpx.HTTPStatusError("404", request=req, response=httpx.Response(404, request=req))


def book(tmp_path, bars, broker=None, slots=20, model=MODEL):
    prices = Prices(bars)
    cfg = E.Config(state_dir=tmp_path / "book", model=model, slots=slots)
    b = E.Book(cfg, broker or B.DryRun(prices.get), prices, edgar=NoEdgar(),
               shares=SharesTable([("1", date(2024, 1, 2), 1e8)]), filings_cache=tmp_path / "filings")
    return b


def price_on(bars, t, d):
    return bars[t].at(d).close


def start_state(b, first_index_day):
    s = E.new_state(b.broker.kind, 100_000.0, first_index_day)
    E.save_state(b.state_path, s) if b.state_path.parent.exists() else None
    return s


def test_book_full_cycle(tmp_path):
    bars, ds = flat_market(["AAA", "BBB"])
    filed = ds[320]                 # 67% below the high, flat since
    entry = ds[321]
    lvl = bars["AAA"].at(filed).close
    # AAA opens flat on entry day and closes 1% higher every day after.
    bars["AAA"] = bp.Bars([x if x.date <= entry else bp.Bar(x.date, lvl * 1.01, lvl * 1.01, lvl * 1.01,
                                                            lvl * 1.01, 1e6) for x in bars["AAA"].bars])
    # BBB rose 2% between its trade and entry, so the run-up penalty puts it below the cutoff.
    bars["BBB"] = bp.Bars([x if x.date != filed else bp.Bar(x.date, x.close * 1.02, x.close * 1.02,
                                                            x.close * 1.02, x.close * 1.02, 1e6)
                           for x in bars["BBB"].bars])
    b = book(tmp_path, bars)
    b.cfg.state_dir.mkdir(parents=True)
    E.save_state(b.state_path, E.new_state("dry-run", 100_000.0, filed - timedelta(days=1)))
    write_day(tmp_path / "filings", filed, [
        buy_filing("AAA", 1, filed, price=lvl),
        filing("x", filer="2", ticker="BBB", filed=filed.isoformat(), cik="1",
               trades=[trade(td=ds[319].isoformat(), price=str(price_on(bars, "BBB", ds[319])))]),
    ])
    r1 = b.run(entry)
    assert [d.ticker for d in r1.candidates] == ["AAA"]
    st = E.load_state(b.state_path)
    [p] = st["positions"]
    close = lvl
    assert p["status"] == "pending_entry" and p["entry_order"]["tif"] == "opg"
    assert p["entry_order"]["qty"] == int(100_000 / 20 // close)
    assert p["entry_order"]["client_id"] == f"tt-{entry}-AAA-buy"
    dec = (b.cfg.state_dir / "decisions.csv").read_text()
    assert "AAA" in dec and "BBB" in dec and "below_cutoff" in dec

    # Same morning again: no duplicate order, no re-ingest.
    r1b = b.run(entry)
    assert not r1b.submitted and len(E.load_state(b.state_path)["positions"]) == 1

    # Next morning: the dry run fills at the entry session's open.
    r2 = b.run(NYSE.add(entry, 1))
    p = E.load_state(b.state_path)["positions"][0]
    assert p["status"] == "open" and p["entry_session"] == entry.isoformat()
    assert p["entry_price"] == pytest.approx(close)
    due = NYSE.add(entry, 59)
    assert p["exit_due"] == due.isoformat()
    assert "index for" in " ".join(r2.notes)     # nothing published for those days (404)

    # The exit morning: a closing-auction sell; filled at that session's close.
    st = E.load_state(b.state_path)
    st["last_index_day"] = (due - timedelta(days=1)).isoformat()   # skip 60 empty days of indexes
    E.save_state(b.state_path, st)
    b.run(NYSE.on_or_before(due - timedelta(days=1)))
    assert E.load_state(b.state_path)["positions"][0]["status"] == "open"
    r = b.run(due)
    p = E.load_state(b.state_path)["positions"][0]
    assert p["status"] == "pending_exit" and p["exit_order"]["tif"] == "cls"
    assert [o["side"] for o in r.submitted] == ["sell"]
    b.run(NYSE.add(due, 1))
    p = E.load_state(b.state_path)["positions"][0]
    assert p["status"] == "closed" and p["exit_session"] == due.isoformat()
    assert p["ret"] == pytest.approx(0.01)
    spy = bars["SPY"]
    assert p["spy_ret"] == pytest.approx(spy.at(due).adj_close / spy.at(entry).adj_open - 1)
    assert p["excess"] == pytest.approx(p["ret"] - p["spy_ret"])
    assert p["excess_net"] == pytest.approx(p["excess"] - p["cost"])
    # Held exactly 60 sessions.
    assert NYSE.business_days_between(entry, due) + 1 == 60
    # Equity counts the closed gain.
    assert b.equity(E.load_state(b.state_path), NYSE.add(due, 1)) == pytest.approx(
        100_000 + p["entry_qty"] * p["entry_price"] * 0.01)
    text = R.render(E.load_state(b.state_path), b.prices, NYSE.add(due, 1), 100_500)
    assert "## Closed" in text and "AAA" in text and "Beat SPY" in text


def flat_market(tickers, start=date(2025, 1, 2), sessions=520):
    """Each ticker falls 67% over 300 sessions, then is flat: a candidate any day after."""
    ds = days_from(start, sessions)
    out = {"SPY": bp.Bars([bp.Bar(d, 100, 100, 100, 100, 1e8) for d in ds])}
    for k, t in enumerate(tickers):
        level = 10 + 0.01 * k
        bs = []
        for i, d in enumerate(ds):
            c = 30 - (30 - level) * i / 299 if i < 300 else level
            bs.append(bp.Bar(d, c, c, c, c, 1e6))
        out[t] = bp.Bars(bs)
    return out, ds


def new_book(tmp_path, bars, ds, filed, **kw):
    b = book(tmp_path, bars, **kw)
    b.cfg.state_dir.mkdir(parents=True, exist_ok=True)
    E.save_state(b.state_path, E.new_state(b.broker.kind, 100_000.0, filed - timedelta(days=1)))
    return b


def test_book_slots_ranking_and_held_tickers(tmp_path):
    names = [f"T{i:02d}" for i in range(25)]
    bars, ds = flat_market(names)
    filed, entry = ds[320], ds[321]
    # Deeper falls score higher: give each a distinct depth via the high.
    for k, t in enumerate(names):
        bars[t] = bp.Bars([bp.Bar(x.date, x.close * (1 if i >= 300 else 1 + 0.02 * k), x.close * (1 if i >= 300 else 1 + 0.02 * k),
                                  x.close * (1 if i >= 300 else 1 + 0.02 * k), x.close * (1 if i >= 300 else 1 + 0.02 * k), 1e6)
                           for i, x in enumerate(bars[t].bars)])
    fs = [buy_filing(t, k + 1, filed, price=price_on(bars, t, filed)) for k, t in enumerate(names)]
    fs.append(buy_filing("T24", 99, filed, price=price_on(bars, "T24", filed)))   # 2nd insider, same stock
    write_day(tmp_path / "filings", filed, fs)
    b = new_book(tmp_path, bars, ds, filed)
    r = b.run(entry)
    assert len(r.candidates) == 25                       # one per stock
    bought = [o["ticker"] for o in r.submitted]
    assert len(bought) == 20 and len(set(bought)) == 20
    # Highest scores first: the deepest falls.
    assert bought == [d.ticker for d in r.candidates[:20]]
    actions = (b.cfg.state_dir / "decisions.csv").read_text()
    assert actions.count("no free slot") == 5
    assert "another insider's buy of this stock scored higher" in actions

    # Next day a new buy in a held stock and one new stock: no slot, nothing bought.
    d2 = NYSE.add(entry, 1)
    write_day(tmp_path / "filings", entry, [buy_filing(bought[0], 7, entry, price=price_on(bars, bought[0], entry))])
    r2 = b.run(d2)
    assert not r2.submitted
    assert "already held" in (b.cfg.state_dir / "decisions.csv").read_text()


def test_book_late_and_missed_runs(tmp_path):
    bars, ds = flat_market(["AAA", "BBB"])
    filed = ds[320]
    write_day(tmp_path / "filings", filed, [buy_filing("AAA", 1, filed, price=price_on(bars, "AAA", filed))])
    late_filed = ds[310]
    write_day(tmp_path / "filings", late_filed, [buy_filing("BBB", 2, late_filed, price=price_on(bars, "BBB", late_filed))])
    # The book's last run was before ds[310]; the next run is ds[323] (2 sessions late
    # for AAA, 12 for BBB). Indexes in between are missing (404), so the run stops there.
    b = new_book(tmp_path, bars, ds, late_filed)
    r = b.run(ds[323])
    assert r.waiting_for == NYSE.add(late_filed, 1).isoformat() or r.waiting_for is not None
    # Fill in the empty days so ingest can reach AAA's day.
    d = late_filed + timedelta(days=1)
    while d < ds[323]:
        if d not in (filed,) and not (tmp_path / "filings" / f"{d}.json").exists():
            write_day(tmp_path / "filings", d, [])
        d += timedelta(days=1)
    st = E.load_state(b.state_path)
    st["last_index_day"] = (late_filed - timedelta(days=1)).isoformat()
    E.save_state(b.state_path, st)
    r = b.run(ds[323])
    status = {d.ticker: (d.status, d.flags) for d in r.decisions}
    assert status["BBB"][0] == "too_late"
    assert status["AAA"][0] == "candidate" and "late_entry_2" in status["AAA"][1]
    assert [o["ticker"] for o in r.submitted] == ["AAA"]


def test_book_missed_exit_sells_next_open(tmp_path):
    bars, ds = flat_market(["AAA"])
    filed, entry = ds[320], ds[321]
    write_day(tmp_path / "filings", filed, [buy_filing("AAA", 1, filed, price=price_on(bars, "AAA", filed))])
    b = new_book(tmp_path, bars, ds, filed)
    b.run(entry)
    b.run(ds[322])
    st = E.load_state(b.state_path)
    due = date.fromisoformat(st["positions"][0]["exit_due"])
    st["last_index_day"] = (due + timedelta(days=5)).isoformat()
    E.save_state(b.state_path, st)
    late = NYSE.add(due, 2)
    r = b.run(late)
    p = E.load_state(b.state_path)["positions"][0]
    assert p["exit_order"]["tif"] == "day" and p["status"] == "pending_exit"
    assert any("exit was due" in n for n in r.notes)
    b.run(NYSE.add(late, 1))
    p = E.load_state(b.state_path)["positions"][0]
    assert p["status"] == "closed" and p["exit_session"] == late.isoformat()


def test_book_unfilled_buy_frees_the_slot(tmp_path):
    bars, ds = flat_market(["AAA"])
    filed, entry = ds[320], ds[321]
    write_day(tmp_path / "filings", filed, [buy_filing("AAA", 1, filed, price=price_on(bars, "AAA", filed))])
    b = new_book(tmp_path, bars, ds, filed, slots=1)
    b.run(entry)
    # The stock stops trading: no bar on the entry session, so the dry run expires it.
    bars["AAA"] = bp.Bars([x for x in bars["AAA"].bars if x.date < entry])
    st = E.load_state(b.state_path)
    st["last_index_day"] = ds[330].isoformat()
    E.save_state(b.state_path, st)
    r = b.run(ds[330])
    p = E.load_state(b.state_path)["positions"][0]
    assert p["status"] == "not_filled" and "expired" in p["close_reason"]
    assert any("slot freed" in n for n in r.notes)
    assert E.live_positions(E.load_state(b.state_path)) == []


class ScriptedBroker:
    """An Alpaca stand-in: answers refreshes from a script, records submissions."""
    kind = "alpaca"

    def __init__(self):
        self.submitted, self.fills, self.qty = [], {}, {}
        self.tradable_ = True

    def equity(self):
        return 50_000.0

    def tradable(self, s):
        return self.tradable_

    def position_qty(self, s):
        return self.qty.get(s)

    def submit(self, symbol, qty, side, tif, cid, session):
        self.submitted.append((symbol, qty, side, tif, cid))
        return B.Fill(status="accepted", broker_id="o-" + cid, tif=tif)

    def refresh(self, cid, broker_id=None, **kw):
        return self.fills.get(cid, B.Fill(status="accepted"))


def test_book_with_broker_partial_fills_splits_and_retries(tmp_path):
    bars, ds = flat_market(["AAA"])
    filed, entry = ds[320], ds[321]
    write_day(tmp_path / "filings", filed, [buy_filing("AAA", 1, filed, price=price_on(bars, "AAA", filed))])
    br = ScriptedBroker()
    b = new_book(tmp_path, bars, ds, filed, broker=br)
    b.run(entry)
    (sym, qty, side, tif, cid) = br.submitted[0]
    assert qty == int(50_000 / 20 // price_on(bars, "AAA", filed))    # sized from the broker's equity
    # Partly filled, then canceled after the open.
    br.fills[cid] = B.Fill(status="canceled", filled_qty=100, filled_price=10.0,
                           filled_at=f"{entry}T13:30:00Z")
    r = b.run(ds[322])
    p = E.load_state(b.state_path)["positions"][0]
    assert p["status"] == "open" and p["entry_qty"] == 100 and any("partly" in n for n in r.notes)
    due = date.fromisoformat(p["exit_due"])
    st = E.load_state(b.state_path)
    st["last_index_day"] = due.isoformat()
    E.save_state(b.state_path, st)
    br.qty["AAA"] = 200.0                     # a 2-for-1 split while held
    b.run(due)
    sell = br.submitted[-1]
    assert sell[:4] == ("AAA", 200, "sell", "cls")
    br.fills[sell[4]] = B.Fill(status="expired", note="auction missed")
    r = b.run(NYSE.add(due, 1))
    p = E.load_state(b.state_path)["positions"][0]
    assert p["status"] == "pending_exit" and p["exit_order"]["tif"] == "day"   # retried at the open
    br.fills[br.submitted[-1][4]] = B.Fill(status="filled", filled_qty=200, filled_price=5.5,
                                           filled_at=f"{NYSE.add(due, 1)}T13:30:00Z")
    b.run(NYSE.add(due, 2))
    p = E.load_state(b.state_path)["positions"][0]
    assert p["status"] == "closed" and p["ret"] == pytest.approx(200 * 5.5 / (100 * 10.0) - 1)
    assert p["excess_net"] == p["excess"]          # Alpaca's fills already pay the spread


def test_book_skips_untradable_and_rejected(tmp_path):
    bars, ds = flat_market(["AAA", "BBB"])
    filed, entry = ds[320], ds[321]
    write_day(tmp_path / "filings", filed, [buy_filing(t, k, filed, price=price_on(bars, t, filed))
                                            for k, t in enumerate(["AAA", "BBB"])])
    br = ScriptedBroker()
    br.tradable_ = False
    b = new_book(tmp_path, bars, ds, filed, broker=br)
    b.run(entry)
    assert not br.submitted and "not tradable" in (b.cfg.state_dir / "decisions.csv").read_text()


def test_book_weekend_run_does_nothing(tmp_path):
    bars, ds = flat_market(["AAA"])
    filed = ds[320]
    write_day(tmp_path / "filings", filed, [buy_filing("AAA", 1, filed, price=price_on(bars, "AAA", filed))])
    b = new_book(tmp_path, bars, ds, filed)
    sat = filed + timedelta(days=(5 - filed.weekday()) % 7 or 7)
    assert sat.weekday() == 5
    r = b.run(sat)
    assert r.session is None and not r.submitted and not r.decisions
    assert E.load_state(b.state_path)["last_index_day"] == (filed - timedelta(days=1)).isoformat()


def test_book_lock(tmp_path):
    bars, ds = flat_market(["AAA"])
    b = new_book(tmp_path, bars, ds, ds[320])
    with E.locked(b.cfg.state_dir):
        with pytest.raises(RuntimeError, match="another paper-trade run"):
            b.run(ds[321])


def test_new_book_starts_with_the_latest_index():
    # A Monday run reads Friday's index; a Tuesday after a holiday Monday too.
    assert E.SEC_prev(date(2026, 10, 19)) + timedelta(days=1) == date(2026, 10, 16)
    assert E.SEC_prev(date(2026, 10, 14)) + timedelta(days=1) == date(2026, 10, 13)
    assert E.SEC_prev(date(2026, 9, 8)) + timedelta(days=1) == date(2026, 9, 4)   # Labor Day


# ---------------------------------------------------------------- run wiring

class FramesClient:
    def __init__(self):
        self.urls = []

    def get(self, url):
        self.urls.append(url)
        frame = url.rsplit("/", 1)[1][:-5]
        return httpx.Response(200, json={"data": [{"cik": 42, "end": "2026-06-30", "val": 5e6,
                                                   "accn": "a"}]},
                              request=httpx.Request("GET", url)) if frame != "CY2026Q4I" else \
            (_ for _ in ()).throw(RuntimeError("404"))


def test_refresh_shares(tmp_path):
    import os
    seed = tmp_path / "seed.csv"
    seed.write_text("cik,end,shares,accn,frame\n1,2024-01-02,100,x,CY2024Q1I\n9,2026-04-01,5,x,CY2026Q2I\n")
    old = 1_700_000_000
    os.utime(seed, (old, old))
    path = tmp_path / "book" / "shares.csv"
    c = FramesClient()
    t = RUN.refresh_shares(c, path, seed, date(2026, 10, 14))
    # The seed's latest quarters were re-fetched; older ones kept.
    assert [u.rsplit("/", 1)[1] for u in c.urls] == ["CY2026Q2I.json", "CY2026Q3I.json", "CY2026Q4I.json"]
    assert t.as_of("1", date(2025, 1, 1)) is not None and t.as_of("9", date(2026, 12, 31)) is None
    assert t.as_of("42", date(2026, 12, 31))[1] == 5e6
    # Fresh now: a second call fetches nothing.
    c2 = FramesClient()
    RUN.refresh_shares(c2, path, seed, date(2026, 10, 14))
    assert c2.urls == []


def test_cli_registers_paper_trade(monkeypatch, tmp_path):
    seen = {}

    def fake_run(state, model, broker, ua, **kw):
        seen.update(state=state, model=model, broker=broker, **kw)
        return E.RunResult(date(2026, 10, 14))
    monkeypatch.setattr(RUN, "run", fake_run)
    monkeypatch.setenv("EDGAR_USER_AGENT", "Test t@example.com")
    assert cli_main(["paper-trade", "--state", str(tmp_path), "--model", "m.json",
                     "--broker", "alpaca", "--today", "2026-10-14"]) == 0
    assert seen["broker"] == "alpaca" and seen["today"] == date(2026, 10, 14)


def test_report_renders_pending_and_empty(tmp_path):
    bars, ds = flat_market(["AAA"])
    filed, entry = ds[320], ds[321]
    write_day(tmp_path / "filings", filed, [buy_filing("AAA", 1, filed, price=price_on(bars, "AAA", filed))])
    b = new_book(tmp_path, bars, ds, filed)
    st = E.load_state(b.state_path)
    assert "none closed yet" in R.render(st, b.prices, entry)
    b.run(entry)
    text = R.render(E.load_state(b.state_path), b.prices, entry, 100_000)
    assert "order for" in text and "| 1 of 20 |" in text
    b.run(ds[322])
    text = R.render(E.load_state(b.state_path), b.prices, ds[330], 100_000)
    assert "AAA" in text and "+0.00%" in text


def test_outcome_unchanged_by_refactor():
    """`pre_entry` is the backtest's own code path: evaluate() still flags a missing
    trade-date close and stops on a ticker mismatch."""
    closes = [10.0] * 100
    bars = history(closes)
    e = ev(first_trade_date=date(2020, 1, 2), last_trade_date=date(2020, 1, 2),
           filing_date=bars.dates[-2])
    from tradetracker.backtest.returns import pre_entry
    o = pre_entry(e, bars, ENTRY, Outcome("ok"))
    assert "no_drift" in o.flags and "no_price_check" in o.flags and o.status == "ok"
    e2 = ev(first_trade_date=bars.dates[-5], last_trade_date=bars.dates[-5],
            filing_date=bars.dates[-2], value=50_000.0, shares=1000.0)
    assert pre_entry(e2, bars, ENTRY, Outcome("ok")).status == "ticker_mismatch"
