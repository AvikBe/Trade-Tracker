"""Does the paper trader's EDGAR ingest see every Form 4 trade the SEC data sets list?

    python scripts/crosscheck_ingest.py --filings DIR --zip 2026q2_form345.zip [--zip ...]

`--filings` is the paper trader's per-day cache (`<date>.json`, from the daily form
index and each filing's submission text). The SEC's quarterly Insider Transactions
Data Sets are built by the SEC from the same filings, by a different pipeline. For each
cached day that the data sets cover, this compares:

1. Filings: every Form 4 or 4/A with an open-market line (P or S) in the data set for
   that filing date must be in the cache, and the other way round.
2. Lines: per accession, the open-market lines (side, trade date, shares, price, shares
   owned after, 10b5-1 flag) must be equal.
3. Header fields: ticker (after the same cleaning), issuer CIK, filer CIK and role.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tradetracker.edgar.bulk import parse_quarter  # noqa: E402
from tradetracker.edgar.common import EASTERN  # noqa: E402


def _d(v) -> Decimal | None:
    # The data sets round numbers to 2 decimals; the XML has them in full.
    return None if v in (None, "") else Decimal(str(v)).quantize(Decimal("0.01"), ROUND_HALF_UP)


def line_key(side, td, shares, price, after, plan):
    return (side, str(td), _d(shares), _d(price), _d(after), bool(plan))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--filings", required=True)
    ap.add_argument("--zip", action="append", required=True)
    args = ap.parse_args(argv)

    cache = {}
    for p in sorted(Path(args.filings).glob("*.json")):
        data = json.loads(p.read_text())
        cache[date.fromisoformat(data["date"])] = data

    bulk: dict[date, dict[str, object]] = {}
    for z in args.zip:
        for f in parse_quarter(z):
            d = f.filed_at.astimezone(EASTERN).date()
            if d in cache and f.trades:
                bulk.setdefault(d, {})[f.source_filing_id] = f

    counts: Counter = Counter()
    problems = []
    for d in sorted(cache):
        if d not in bulk:
            counts["days not in the data sets"] += 1
            continue
        counts["days compared"] += 1
        live = {f["accession"]: f for f in cache[d]["filings"]}
        errs = {e["accession"] for e in cache[d].get("errors", [])}
        b = bulk[d]
        for acc in sorted(set(b) | set(live)):
            if acc not in live:
                why = "unreadable live" if acc in errs else "missing live"
                counts[f"filing {why}"] += 1
                problems.append((str(d), acc, why))
                continue
            if acc not in b:
                counts["filing missing from the data set"] += 1
                problems.append((str(d), acc, "missing from the data set"))
                continue
            counts["filings in both"] += 1
            lf, bf = live[acc], b[acc]
            ll = sorted(line_key(t["side"], t["trade_date"], t["shares"], t["price"], t["owned_after"],
                                 t["10b5_1"]) for t in lf["trades"])
            bl = sorted(line_key(t.side, t.trade_date, t.shares, t.price, t.shares_owned_after,
                                 t.is_10b5_1) for t in bf.trades)
            counts["lines compared"] += len(bl)
            if ll != bl:
                counts["filings with different lines"] += 1
                problems.append((str(d), acc, f"lines live {ll} vs data set {bl}"))
            for field, a, c in (("ticker", lf["ticker"], bf.issuer_ticker),
                                ("issuer", (lf["issuer_cik"] or "").lstrip("0"), bf.issuer_cik or ""),
                                ("filer", lf["filer"], bf.filer_key),
                                ("role", lf["role"], bf.role),
                                ("form", lf["form"], bf.document_type)):
                if a != c:
                    counts[f"different {field}"] += 1
                    problems.append((str(d), acc, f"{field}: live {a!r} vs data set {c!r}"))
    print(json.dumps(dict(sorted(counts.items())), indent=1))
    for p in problems[:60]:
        print("  ", *p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
