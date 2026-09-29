"""Issuer SIC codes from EDGAR, mapped to the SPDR sector ETFs used as benchmarks.

SIC comes from `data.sec.gov/submissions/CIK##########.json` (one call per issuer,
cached in a CSV). SIC is the issuer's current code, not the one on the trade date;
industries rarely change, and it only picks the benchmark, never a signal.

The SIC to sector map is a hand-built approximation of GICS. Sector ETFs that did not
exist yet (XLRE before Oct 2015, XLC before Jun 2018), blank-check companies and
unmapped codes fall back to SPY.
"""

from __future__ import annotations

import csv
import logging
import time
from pathlib import Path

import httpx

log = logging.getLogger(__name__)

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:0>10}.json"
SECTOR_ETFS = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]

# (first SIC, last SIC, ETF). The first matching range wins, so narrow ranges come first.
SIC_RANGES: list[tuple[int, int, str]] = [
    (1311, 1389, "XLE"),   # crude oil, natural gas, drilling and oilfield services
    (2830, 2836, "XLV"),   # pharmaceuticals and biologics
    (2840, 2844, "XLP"),   # soap, cosmetics
    (2900, 2999, "XLE"),   # petroleum refining
    (3570, 3579, "XLK"),   # computers and office equipment
    (3630, 3639, "XLY"),   # household appliances
    (3650, 3652, "XLY"),   # household audio and video
    (3660, 3699, "XLK"),   # communications equipment, semiconductors, electronics
    (3710, 3716, "XLY"),   # motor vehicles
    (3751, 3751, "XLY"),   # motorcycles and bicycles
    (3810, 3829, "XLK"),   # navigation, measuring and control instruments
    (3840, 3851, "XLV"),   # medical instruments and supplies
    (3860, 3879, "XLY"),   # photographic equipment, watches
    (4810, 4899, "XLC"),   # telecom and broadcasting
    (4955, 4955, "XLI"),   # hazardous waste management
    (5122, 5122, "XLV"),   # drug wholesale
    (5400, 5499, "XLP"),   # food stores
    (5912, 5912, "XLP"),   # drug stores
    (6500, 6553, "XLRE"),  # real estate
    (6770, 6770, "SPY"),   # blank checks (SPACs)
    (6798, 6798, "XLRE"),  # REITs
    (7310, 7319, "XLC"),   # advertising
    (7370, 7379, "XLK"),   # software and computer services
    (7810, 7849, "XLC"),   # motion pictures
    (100, 999, "XLP"),     # agriculture
    (1000, 1499, "XLB"),   # other mining and quarrying
    (1500, 1799, "XLI"),   # construction
    (2000, 2199, "XLP"),   # food, beverages, tobacco
    (2200, 2399, "XLY"),   # textiles and apparel
    (2400, 2499, "XLB"),   # lumber and wood
    (2500, 2599, "XLY"),   # furniture
    (2600, 2699, "XLB"),   # paper
    (2700, 2799, "XLC"),   # printing and publishing
    (2800, 2899, "XLB"),   # other chemicals
    (3000, 3099, "XLB"),   # rubber and plastics
    (3100, 3199, "XLY"),   # leather goods
    (3200, 3399, "XLB"),   # stone, glass, primary metals
    (3400, 3569, "XLI"),   # fabricated metal, machinery
    (3580, 3629, "XLI"),   # other machinery, electrical industrial equipment
    (3640, 3649, "XLI"),   # lighting
    (3700, 3799, "XLI"),   # aircraft, ships, rail and other transport equipment
    (3800, 3899, "XLI"),   # other instruments
    (3900, 3999, "XLY"),   # other manufacturing (toys, jewelry, sporting goods)
    (4000, 4799, "XLI"),   # transportation
    (4900, 4999, "XLU"),   # utilities
    (5000, 5099, "XLI"),   # wholesale, durable goods
    (5100, 5199, "XLP"),   # wholesale, nondurable goods
    (5200, 5999, "XLY"),   # retail
    (6000, 6799, "XLF"),   # finance and insurance
    (7000, 7299, "XLY"),   # hotels, personal services
    (7300, 7399, "XLI"),   # other business services
    (7500, 7699, "XLY"),   # auto repair, misc repair
    (7900, 7999, "XLY"),   # amusement and recreation
    (8000, 8099, "XLV"),   # health services
    (8200, 8299, "XLY"),   # education
    (8700, 8799, "XLI"),   # engineering, research, management services
]


def sector_etf(sic: int | str | None) -> str:
    """The sector ETF for a SIC code; SPY when unknown or unmapped."""
    try:
        code = int(sic)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "SPY"
    for lo, hi, etf in SIC_RANGES:
        if lo <= code <= hi:
            return etf
    return "SPY"


def load_sic(path: Path) -> dict[str, str]:
    """cik -> SIC code (as text, '' when EDGAR has none)."""
    if not path.exists():
        return {}
    with path.open() as fh:
        return {r["cik"].lstrip("0"): r["sic"] for r in csv.DictReader(fh)}


def fetch_sic(ciks: list[str], path: Path, user_agent: str, *, per_second: float = 8.0) -> dict:
    """Fetch the SIC of every CIK not yet in the cache file, appending as it goes."""
    known = load_sic(path)
    todo = [c for c in dict.fromkeys(c.lstrip("0") for c in ciks if c) if c not in known]
    new_file = not path.exists()
    stats = {"cached": len(known), "requested": len(todo), "ok": 0, "missing": 0, "errors": 0}
    with httpx.Client(timeout=30, headers={"User-Agent": user_agent}) as http, path.open("a") as fh:
        w = csv.writer(fh, lineterminator="\n")
        if new_file:
            w.writerow(["cik", "sic", "sic_description", "name"])
        for i, cik in enumerate(todo):
            started = time.monotonic()
            for attempt in range(4):
                try:
                    resp = http.get(SUBMISSIONS_URL.format(cik=cik))
                except httpx.HTTPError:
                    time.sleep(2 ** attempt)
                    continue
                if resp.status_code in (429, 503):
                    time.sleep(10 * (attempt + 1))
                    continue
                break
            else:
                stats["errors"] += 1
                continue
            if resp.status_code == 404:
                w.writerow([cik, "", "", ""])
                stats["missing"] += 1
            elif resp.status_code == 200:
                d = resp.json()
                w.writerow([cik, d.get("sic") or "", d.get("sicDescription") or "", d.get("name") or ""])
                stats["ok"] += 1
            else:
                stats["errors"] += 1
            if i % 500 == 0:
                fh.flush()
                log.info("sic: %d/%d", i, len(todo))
            time.sleep(max(0.0, 1 / per_second - (time.monotonic() - started)))
    return stats
