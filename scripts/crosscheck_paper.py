"""Does the paper trader score a buy exactly as the backtest did?

    python scripts/crosscheck_paper.py --run DIR --events events.csv.gz --prices DIR \
        --shares shares.csv [--since 2015-01-01]

`--run` is a score-backtest output directory (scored.csv.gz, model.json, summary.json),
`--events` the phase 3 events file (trade dates, Form 4 price, 10b5-1 flag, status).

For every liquid open-market buy the backtest priced, the paper trader's own code
(`paper.signals.evaluate`) gets the event as the live ingest would build it and the
price bars cut off before the entry open, then:

1. Shown events (in scored.csv): drift, 52-week drawdown, median dollar volume, market
   cap and round-trip cost must equal the backtest's, the score must equal the final
   model applied to the backtest's own signals, and the above-cutoff call must agree.
2. Events the backtest hid (drift above 20%, small or unknown cap, illiquid, 10b5-1,
   ticker mismatch) must be hidden too.

The one known gap: the backtest's $2 floor uses the entry open, which the paper trader
can't know before the open, so it uses the last close. Events near $2 are listed apart.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import sys
from collections import Counter
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tradetracker.backtest.events import Event  # noqa: E402
from tradetracker.backtest.prices import CachePrices  # noqa: E402
from tradetracker.fundamentals import SharesTable  # noqa: E402
from tradetracker.paper.ingest import LiveEvent  # noqa: E402
from tradetracker.paper.signals import evaluate  # noqa: E402
from tradetracker.scoring import model as M  # noqa: E402
from tradetracker.scoring.signals import Signals  # noqa: E402

TOL = 1e-9


def _f(v: str) -> float | None:
    return None if v in ("", None) else float(v)


def _close(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(a, b, rel_tol=TOL, abs_tol=TOL)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--events", required=True)
    ap.add_argument("--prices", required=True)
    ap.add_argument("--shares", required=True)
    ap.add_argument("--since", default="2015-01-01")
    ap.add_argument("--min-cap", type=float, default=50e6)
    ap.add_argument("--out", help="write mismatches here (CSV)")
    args = ap.parse_args(argv)

    run = Path(args.run)
    m = M.Model.from_json((run / "model.json").read_text())
    shares = SharesTable.load(Path(args.shares))
    prices = CachePrices(args.prices)
    since = date.fromisoformat(args.since)

    scored, by_stock_day = {}, {}
    with gzip.open(run / "scored.csv.gz", "rt") as f:
        for r in csv.DictReader(f):
            scored[(r["filer_id"], r["ticker"], r["filing_date"])] = r
            by_stock_day[(r["ticker"], r["entry"])] = r

    counts: Counter = Counter()
    bad: list[dict] = []
    near_floor: list[str] = []
    fields = ["drift", "drawdown", "dollar_volume", "market_cap", "cost"]
    with gzip.open(args.events, "rt") as f:
        for r in csv.DictReader(f):
            if r["side"] != "buy" or r["status"] != "ok" or date.fromisoformat(r["filing_date"]) < since:
                continue
            key = (r["filer_id"], r["ticker"], r["filing_date"])
            s = scored.get(key)
            liquid = (_f(r["entry_open"]) or 0) >= 2 and (_f(r["dollar_volume"]) or 0) >= 100_000
            if s is None and not liquid:
                continue                    # never in the sample: nothing to compare
            value = float(r["value"] or 0)
            avg = _f(r["avg_price"])
            e = Event(filer_id=int(r["filer_id"]), issuer_cik=r["issuer_cik"] or None,
                      ticker=r["ticker"], side="buy", filing_date=date.fromisoformat(r["filing_date"]),
                      accepted_at=None, first_trade_date=date.fromisoformat(r["first_trade_date"]),
                      last_trade_date=date.fromisoformat(r["last_trade_date"]), lag_days=0,
                      lag_ratio=0.0, value=value, shares=value / avg if avg else 0.0,
                      role=r["role"] or None, is_10b5_1=r["is_10b5_1"] == "1")
            entry = date.fromisoformat(r["entry"])
            d = evaluate(LiveEvent(e, [], r["filer_id"], "", None), prices.get(e.ticker), entry,
                         m, shares, args.min_cap)
            live_shown = d.status in ("candidate", "below_cutoff")
            near = abs((_f(r["entry_open"]) or 0) - 2) < 0.5 or (d.last_close and abs(d.last_close - 2) < 0.5)
            sibling = by_stock_day.get((r["ticker"], r["entry"])) if s is None else None
            if sibling is not None:
                # The backtest keeps one insider per stock and entry day (its best-scored,
                # by the fold's model), so this buy was shown but deduplicated. Stock-level
                # inputs must still match the kept insider's; drift is per insider.
                counts["deduplicated by backtest (another insider kept)"] += 1
                if d.status in ("candidate", "below_cutoff"):
                    sig = d.signals
                    for k, v in (("drawdown", sig.drawdown), ("dollar_volume", sig.dollar_volume),
                                 ("market_cap", sig.market_cap), ("cost", d.cost)):
                        if not _close(v, _f(sibling[k])):
                            counts[f"MISMATCH {k} (deduplicated)"] += 1
                            bad.append({"key": key, "what": k, "live": v, "backtest": sibling[k]})
                elif d.status in ("drift", "10b5-1", "ticker_mismatch"):
                    # Per-insider rules: the kept insider passing them says nothing here.
                    counts[f"deduplicated, hidden live by a per-insider rule ({d.status})"] += 1
                elif d.status == "illiquid" and near:
                    counts["deduplicated, $2 floor gap"] += 1
                else:
                    counts["MISMATCH deduplicated, hidden live"] += 1
                    bad.append({"key": key, "what": "hidden live (deduplicated)", "live": d.status})
                continue
            if s is None:
                counts["hidden by backtest"] += 1
                if live_shown:
                    if near:
                        near_floor.append(f"{key}: shown live, hidden by backtest (entry open {r['entry_open']}, last close {d.last_close})")
                        counts["hidden by backtest, $2 floor gap"] += 1
                    else:
                        counts["MISMATCH hidden by backtest, shown live"] += 1
                        bad.append({"key": key, "what": "shown live", "live": d.status})
                else:
                    counts["hidden by both"] += 1
                continue
            counts["shown by backtest"] += 1
            if not live_shown:
                if near and d.status == "illiquid":
                    counts["shown by backtest, $2 floor gap"] += 1
                    near_floor.append(f"{key}: hidden live ({d.status}), shown by backtest (entry open {r['entry_open']}, last close {d.last_close})")
                else:
                    counts["MISMATCH shown by backtest, hidden live"] += 1
                    bad.append({"key": key, "what": "hidden live", "live": d.status})
                continue
            sig = d.signals
            live = {"drift": sig.drift, "drawdown": sig.drawdown, "dollar_volume": sig.dollar_volume,
                    "market_cap": sig.market_cap, "cost": d.cost}
            ok = True
            for k in fields:
                if not _close(live[k], _f(s[k])):
                    ok = False
                    counts[f"MISMATCH {k}"] += 1
                    bad.append({"key": key, "what": k, "live": live[k], "backtest": s[k]})
            bt_sig = Signals(role=None, value=0, stake_change=None, cluster_count=1, repeat_buys=0,
                             drawdown=_f(s["drawdown"]), track_record=None, track_n=0,
                             dollar_volume=_f(s["dollar_volume"]), drift=_f(s["drift"]),
                             market_cap=_f(s["market_cap"]))
            want = m.score(bt_sig)
            if not _close(d.score, want):
                ok = False
                counts["MISMATCH score"] += 1
                bad.append({"key": key, "what": "score", "live": d.score, "backtest": want})
            if (d.status == "candidate") != (want >= m.threshold):
                ok = False
                counts["MISMATCH above cutoff"] += 1
            counts["candidate (final model)" if d.status == "candidate" else "below cutoff (final model)"] += 1
            counts["all fields equal" if ok else "some field differs"] += 1

    print(json.dumps(dict(sorted(counts.items())), indent=1))
    for line in near_floor[:30]:
        print("  $2 floor:", line)
    if len(near_floor) > 30:
        print(f"  ... {len(near_floor) - 30} more")
    for b in bad[:30]:
        print("  MISMATCH", b)
    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["key", "what", "live", "backtest"])
            w.writeheader()
            w.writerows(bad)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
