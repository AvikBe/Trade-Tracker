"""Replay the paper book over past years and check it trades what the backtest traded.

    python scripts/replay_paper.py --events events.csv.gz --run DIR --prices DIR \
        --shares shares.csv --start 2023-01-03 --end 2026-06-30 --out DIR

The paper book (`paper.engine.Book`, dry-run broker) runs every NYSE morning from
--start to --end. Instead of EDGAR's daily index, each morning it reads the buy events
the backtest built from the SEC data sets for the filing days since its last run (all
of them, whatever the backtest later decided). Everything else is the live code:
scoring from bars before the open, hide rules, slots, the held-ticker check, fills at
the open and close from the bars, and the 60-session exit.

The reference is the backtest's top-20 portfolio rule (`scoring.study.simulate`) applied
to the run's scored events (scored.csv.gz), with every score recomputed by the final
model, so both sides use the same model and the comparison is about the trading
mechanics. Reported:

- positions both took, positions only one took (with the reason the other didn't);
- for positions both took and closed: entry and exit days, and the net 60-day excess
  over SPY, which must equal the backtest's per-event target;
- the paper book's mean excess, hit rate and number of trades against the backtest's.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tradetracker.backtest.events import Event  # noqa: E402
from tradetracker.backtest.prices import CachePrices  # noqa: E402
from tradetracker.features.calendar import NYSE  # noqa: E402
from tradetracker.fundamentals import SharesTable  # noqa: E402
from tradetracker.paper import engine as E  # noqa: E402
from tradetracker.paper.broker import DryRun  # noqa: E402
from tradetracker.paper.ingest import LiveEvent  # noqa: E402
from tradetracker.scoring import model as M  # noqa: E402
from tradetracker.scoring.signals import Signals  # noqa: E402


def _f(v):
    return None if v in ("", None) else float(v)


def load_events(path: Path, start: date, end: date) -> dict[date, list[LiveEvent]]:
    by_day: dict[date, list[LiveEvent]] = defaultdict(list)
    with gzip.open(path, "rt") as f:
        for r in csv.DictReader(f):
            if r["side"] != "buy":
                continue
            fd = date.fromisoformat(r["filing_date"])
            if not start - timedelta(days=10) <= fd <= end:
                continue
            value = float(r["value"] or 0)
            avg = _f(r["avg_price"])
            e = Event(filer_id=int(r["filer_id"]), issuer_cik=r["issuer_cik"] or None,
                      ticker=r["ticker"], side="buy", filing_date=fd, accepted_at=None,
                      first_trade_date=date.fromisoformat(r["first_trade_date"]),
                      last_trade_date=date.fromisoformat(r["last_trade_date"]), lag_days=0,
                      lag_ratio=0.0, value=value, shares=value / avg if avg else 0.0,
                      role=r["role"] or None, is_10b5_1=r["is_10b5_1"] == "1")
            by_day[fd].append(LiveEvent(e, [f"{r['filer_id']}-{r['filing_date']}"], r["filer_id"], "", None))
    return by_day


class ReplayBook(E.Book):
    def __init__(self, *a, events_by_day, **kw):
        super().__init__(*a, **kw)
        self.events_by_day = events_by_day

    def ingest(self, state, today, res):
        d = date.fromisoformat(state["last_index_day"]) + timedelta(days=1)
        out = []
        while d < today:
            out += self.events_by_day.get(d, [])
            d += timedelta(days=1)
        state["last_index_day"] = (today - timedelta(days=1)).isoformat()
        return out


def backtest_positions(run: Path, m: M.Model, start: date, end: date, slots: int, h: int):
    """The backtest's portfolio rule on its scored events, scores from the final model."""
    by_entry = defaultdict(list)
    with gzip.open(run / "scored.csv.gz", "rt") as f:
        for r in csv.DictReader(f):
            entry = date.fromisoformat(r["entry"])
            if not start <= entry <= end:
                continue
            s = Signals(None, 0, None, 1, 0, _f(r["drawdown"]), None, 0, _f(r["dollar_volume"]),
                        _f(r["drift"]), _f(r["market_cap"]))
            sc = m.score(s)
            if sc >= m.threshold:
                by_entry[entry].append((sc, r))
    held, out = [], []
    days = []
    d = start
    while d <= end + timedelta(days=200):
        if NYSE.is_business_day(d):
            days.append(d)
        d += timedelta(days=1)
    for d in days:
        best = {}
        for sc, r in by_entry.get(d, []):
            if r["ticker"] not in best or sc > best[r["ticker"]][0]:
                best[r["ticker"]] = (sc, r)
        todays = sorted(best.values(), key=lambda x: (-x[0], x[1]["ticker"]))
        tickers = {p["ticker"] for p, _ in held}
        for sc, r in todays:
            if len(held) >= slots or not r["exit"] or r["ticker"] in tickers:
                continue
            held.append((r, date.fromisoformat(r["exit"])))
            tickers.add(r["ticker"])
            out.append(r)
        held = [(p, x) for p, x in held if x > d]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--prices", required=True)
    ap.add_argument("--shares", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    run, out = Path(args.run), Path(args.out)
    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    m = M.Model.from_json((run / "model.json").read_text())
    prices = CachePrices(args.prices)
    shares = SharesTable.load(Path(args.shares))
    events = load_events(Path(args.events), start, end)

    state_dir = out / "book"
    if state_dir.exists():
        for p in state_dir.iterdir():
            p.unlink()
    cfg = E.Config(state_dir=state_dir, model=m)
    book = ReplayBook(cfg, DryRun(prices.get), prices, shares=shares, events_by_day=events)
    state_dir.mkdir(parents=True, exist_ok=True)
    first = NYSE.on_or_before(start - timedelta(days=1))
    E.save_state(book.state_path, E.new_state("dry-run", 100_000.0, first - timedelta(days=1)))
    t0 = time.time()
    d = start
    last_entry_day = end
    # Keep running (no new events) until every position has been sold.
    stop = NYSE.add(end, 70)
    while d <= stop:
        if NYSE.is_business_day(d):
            if d > last_entry_day:
                book.events_by_day = {}
                book.cfg.slots = 0           # no new buys after --end; only exits
            book.run(d)
        d += timedelta(days=1)
    state = E.load_state(book.state_path)
    print(f"replayed {start}..{stop} in {time.time() - t0:.0f}s")

    paper = {(p["ticker"], p["entry_session"]): p for p in state["positions"]
             if p.get("entry_session")}
    unfilled = [p for p in state["positions"] if p["status"] == "not_filled"]
    bt = backtest_positions(run, m, start, end, E.SLOTS, E.HORIZON)
    btk = {(r["ticker"], r["entry"]): r for r in bt}

    both = sorted(set(paper) & set(btk))
    only_p = sorted(set(paper) - set(btk))
    only_b = sorted(set(btk) - set(paper))
    mism = []
    for k in both:
        p, r = paper[k], btk[k]
        if p["status"] != "closed":
            mism.append((k, "paper position not closed", p["status"]))
            continue
        if p["exit_session"] != r["exit"]:
            mism.append((k, "exit day", p["exit_session"], r["exit"]))
        if p.get("excess_net") is None or not math.isclose(p["excess_net"], float(r["target"]),
                                                           rel_tol=1e-9, abs_tol=1e-9):
            mism.append((k, "net excess", p.get("excess_net"), r["target"]))
    closed = [p for p in state["positions"] if p["status"] == "closed" and p.get("excess_net") is not None]
    bt_t = [float(r["target"]) for r in bt if r["target"]]
    summary = {
        "sessions": f"{start}..{end}",
        "paper positions": len(paper), "backtest positions": len(btk), "both": len(both),
        "only paper": len(only_p), "only backtest": len(only_b),
        "paper orders not filled": len(unfilled),
        "mismatches on shared positions": len(mism),
        "paper mean net excess": statistics.mean(p["excess_net"] for p in closed) if closed else None,
        "paper hit rate": sum(p["excess_net"] > 0 for p in closed) / len(closed) if closed else None,
        "backtest mean net excess": statistics.mean(bt_t) if bt_t else None,
        "backtest hit rate": sum(t > 0 for t in bt_t) / len(bt_t) if bt_t else None,
    }
    print(json.dumps(summary, indent=1))
    # Why does one side have a position the other doesn't? Look at the paper's decision.
    dec = defaultdict(list)
    with (state_dir / "decisions.csv").open() as f:
        for r in csv.DictReader(f):
            dec[(r["ticker"], r["entry"])].append(f"{r['status']}/{r['action'] or '-'}")
    lines = ["kind,ticker,entry,detail"]
    for k in only_p:
        lines.append(f"only paper,{k[0]},{k[1]},{paper[k]['status']} {paper[k].get('flags')}")
    for k in only_b:
        lines.append(f"only backtest,{k[0]},{k[1]},paper saw: {' | '.join(dec.get(k, ['nothing']))}")
    for x in mism:
        lines.append("mismatch," + ",".join(str(v) for v in x))
    (out / "replay-diff.csv").write_text("\n".join(lines) + "\n")
    (out / "replay-summary.json").write_text(json.dumps(summary, indent=1))
    for line in lines[1:40]:
        print("  " + line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
