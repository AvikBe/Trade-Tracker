"""The first study: do forward returns from the filing date depend on lag and drift?

`run_study` evaluates every event against a price source, then `render` writes the
report: returns by lag bucket and drift quintile for 5, 20 and 60 trading days, excess
over SPY (primary), the sector ETF and IWM, gross and after costs, per period and per
year, and a walk-forward test of whether the in-sample pattern holds out of sample.
"""

from __future__ import annotations

import csv
import gzip
import io
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from . import stats
from .events import Event, Exclusions
from .returns import HORIZONS, Outcome, evaluate
from .sectors import sector_etf

PRIMARY_H = 20
PERIODS = [("2015-2022", 2015, 2022), ("2023-2026", 2023, 2026)]
BENCHES = ("spy", "sector", "iwm")
MIN_LIQUID_PRICE = 2.0
MIN_LIQUID_DOLLAR_VOLUME = 100_000
WALK_FORWARD_MIN_TRAIN_YEARS = 2

LAG_BUCKETS = [
    ("fast (0-1 days, <=0.5x)", 0, 1),
    ("on time (2 days, 1x)", 2, 2),
    ("late (3-4 days, 1-2x)", 3, 4),
    ("very late (5+ days, >2x)", 5, 10**9),
]


def lag_bucket(lag_days: int) -> int:
    for i, (_, lo, hi) in enumerate(LAG_BUCKETS):
        if lo <= lag_days <= hi:
            return i
    raise ValueError(lag_days)


@dataclass
class Result:
    event: Event
    outcome: Outcome
    sector: str

    @property
    def sign(self) -> int:
        return -1 if self.event.side == "sell" else 1

    def r(self, h: int = PRIMARY_H, bench: str = "spy", net: bool = False) -> float | None:
        return self.outcome.signal(h, bench, self.sign, net)

    @property
    def ok(self) -> bool:
        return self.outcome.status == "ok"

    @property
    def liquid(self) -> bool:
        o = self.outcome
        return bool(o.entry_open and o.entry_open >= MIN_LIQUID_PRICE
                    and o.dollar_volume and o.dollar_volume >= MIN_LIQUID_DOLLAR_VOLUME)


def run_study(events: list[Event], prices, sic: dict[str, str]) -> list[Result]:
    spy = prices.get("SPY")
    if spy is None:
        raise RuntimeError(f"the price source {prices.name} has no SPY bars")
    iwm = prices.get("IWM")
    out = []
    for e in events:
        etf = sector_etf(sic.get((e.issuer_cik or "").lstrip("0")))
        sector = prices.get(etf) if etf != "SPY" else None
        outcome = evaluate(e, prices.get(e.ticker), spy, sector, etf, iwm)
        out.append(Result(e, outcome, etf))
    return out


# ---------------------------------------------------------------- formatting helpers

def pct(x: float | None, digits: int = 2) -> str:
    return "" if x is None else f"{100 * x:.{digits}f}%"


def num(x: float | None, digits: int = 2) -> str:
    return "" if x is None else f"{x:.{digits}f}"


def table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(" --- " for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _values(results: list[Result], h: int, bench: str = "spy", net: bool = False):
    vals, keys = [], []
    for x in results:
        v = x.r(h, bench, net)
        if v is not None:
            vals.append(v)
            keys.append(stats.cluster_key(x.outcome.entry, h))
    return vals, keys


def summary_row(label: str, results: list[Result], h: int, winsor) -> list:
    vals, keys = _values(results, h)
    s = stats.summarize(vals, keys, winsor)
    net = stats.summarize(_values(results, h, net=True)[0])
    sec = stats.summarize(_values(results, h, "sector")[0])
    iwm = stats.summarize(_values(results, h, "iwm")[0])
    return [label, s.n, pct(s.mean), num(s.t), num(s.t_cluster), pct(s.median), pct(s.hit, 1),
            pct(s.wmean), pct(net.mean), pct(sec.mean), pct(iwm.mean)]


SUMMARY_HEADERS = ["bucket", "n", "mean vs SPY", "t", "t (clustered)", "median", "hit rate",
                   "winsorized mean", "net of costs", "vs sector", "vs IWM"]


def drift_cutoffs(results: list[Result], q: int = 5) -> list[float]:
    return stats.quantile_cutoffs([x.outcome.drift_pct for x in results
                                   if x.outcome.drift_pct is not None], q)


def by_lag(results: list[Result]) -> dict[int, list[Result]]:
    g = defaultdict(list)
    for x in results:
        g[lag_bucket(x.event.lag_days)].append(x)
    return g


def by_drift(results: list[Result], cutoffs: list[float]) -> dict[int, list[Result]]:
    g = defaultdict(list)
    for x in results:
        if x.outcome.drift_pct is not None:
            g[stats.bucket_of(x.outcome.drift_pct, cutoffs)].append(x)
    return g


def drift_labels(cutoffs: list[float]) -> list[str]:
    edges = ["-inf"] + [f"{c:.1f}%" for c in cutoffs] + ["+inf"]
    return [f"Q{i + 1} ({edges[i]} to {edges[i + 1]})" for i in range(len(cutoffs) + 1)]


def bucket_tables(results: list[Result], h: int, cutoffs: list[float], winsor) -> str:
    lag = by_lag(results)
    drift = by_drift(results, cutoffs)
    labels = drift_labels(cutoffs)
    parts = [
        f"**By lag, {h} trading days**\n",
        table(SUMMARY_HEADERS, [summary_row(LAG_BUCKETS[i][0], lag.get(i, []), h, winsor)
                                for i in range(len(LAG_BUCKETS))]
              + [summary_row("all", results, h, winsor)]),
        f"\n**By pre-disclosure drift quintile, {h} trading days**\n",
        table(SUMMARY_HEADERS, [summary_row(labels[i], drift.get(i, []), h, winsor)
                                for i in range(len(labels))]),
    ]
    return "\n".join(parts)


def cross_table(results: list[Result], h: int, cutoffs: list[float]) -> str:
    lag = by_lag(results)
    labels = drift_labels(cutoffs)
    rows = []
    for i, (name, _, _) in enumerate(LAG_BUCKETS):
        cells = by_drift(lag.get(i, []), cutoffs)
        row = [name]
        for q in range(len(labels)):
            vals, keys = _values(cells.get(q, []), h)
            s = stats.summarize(vals, keys)
            row.append(f"{pct(s.mean)} (n={s.n}, t={num(s.t_cluster, 1)})" if s.n else "")
        rows.append(row)
    return table(["lag \\ drift"] + [f"Q{q + 1}" for q in range(len(labels))], rows)


def contrast(results: list[Result], h: int, cutoffs: list[float]) -> tuple:
    """(late minus prompt, Q5 minus Q1): each a (difference, Welch t, n_a, n_b)."""
    prompt = _values([x for x in results if x.event.lag_days <= 2], h)
    late = _values([x for x in results if x.event.lag_days > 2], h)
    d = by_drift(results, cutoffs)
    q1 = _values(d.get(0, []), h)
    q5 = _values(d.get(len(cutoffs), []), h)
    return ((*stats.diff_summary(late[0], prompt[0], late[1], prompt[1]), len(late[0]), len(prompt[0])),
            (*stats.diff_summary(q5[0], q1[0], q5[1], q1[1]), len(q5[0]), len(q1[0])))


# ---------------------------------------------------------------- walk-forward

@dataclass
class FoldResult:
    year: int
    n_train: int
    n_test: int
    lag_is: float | None
    lag_oos: float | None
    drift_is: float | None
    drift_oos: float | None
    ic: float | None
    top_minus_bottom: float | None


def _mean(v):
    return sum(v) / len(v) if v else None


def walk_forward(results: list[Result], h: int = PRIMARY_H) -> list[FoldResult]:
    """Expanding window: fit on every earlier year, test on the next, roll.

    Training events must have exited before the test year starts (purged), so no
    training return overlaps a test-year day. The fitted "model" is the spec's first
    study itself: mean return per lag bucket and per drift quintile, with quintile
    cutoffs from the training years only.
    """
    rows = [x for x in results if x.r(h) is not None and x.outcome.drift_pct is not None]
    years = sorted({x.outcome.entry.year for x in rows})
    folds = []
    for y in years[WALK_FORWARD_MIN_TRAIN_YEARS:]:
        start = date(y, 1, 1)
        train = [x for x in rows if x.outcome.legs[h]["exit"] < start]
        test = [x for x in rows if x.outcome.entry.year == y]
        if len(train) < 100 or len(test) < 20:
            continue
        cut = drift_cutoffs(train)
        overall = _mean([x.r(h) for x in train])

        def split(rs):
            late = [x.r(h) for x in rs if x.event.lag_days > 2]
            prompt = [x.r(h) for x in rs if x.event.lag_days <= 2]
            d = by_drift(rs, cut)
            q1 = [x.r(h) for x in d.get(0, [])]
            q5 = [x.r(h) for x in d.get(len(cut), [])]
            diff = lambda a, b: (_mean(a) - _mean(b)) if a and b else None  # noqa: E731
            return diff(late, prompt), diff(q5, q1)

        lag_is, drift_is = split(train)
        lag_oos, drift_oos = split(test)
        lag_means = {k: _mean([x.r(h) for x in v]) for k, v in by_lag(train).items()}
        drift_means = {k: _mean([x.r(h) for x in v]) for k, v in by_drift(train, cut).items()}

        def predict(x):
            lb = lag_means.get(lag_bucket(x.event.lag_days), overall)
            dq = drift_means.get(stats.bucket_of(x.outcome.drift_pct, cut), overall)
            return lb + dq - overall

        preds = [predict(x) for x in test]
        realized = [x.r(h) for x in test]
        ic = stats.spearman(preds, realized)
        order = sorted(range(len(test)), key=lambda i: (preds[i], test[i].outcome.entry))
        k = len(order) // 5
        tmb = (_mean([realized[i] for i in order[-k:]]) - _mean([realized[i] for i in order[:k]])
               if k else None)
        folds.append(FoldResult(y, len(train), len(test), lag_is, lag_oos, drift_is, drift_oos,
                                ic, tmb))
    return folds


def walk_forward_table(folds: list[FoldResult]) -> str:
    rows = []
    for f in folds:
        rows.append([f.year, f.n_train, f.n_test, pct(f.lag_is), pct(f.lag_oos),
                     pct(f.drift_is), pct(f.drift_oos), num(f.ic, 3), pct(f.top_minus_bottom)])
    return table(["test year", "train n", "test n", "late-prompt (train)", "late-prompt (test)",
                  "Q5-Q1 drift (train)", "Q5-Q1 drift (test)", "rank IC", "top-bottom fifth"], rows)


def walk_forward_verdict(folds: list[FoldResult]) -> list[str]:
    """Did the training-period direction of each effect hold in the test year?"""
    out = []
    for name, is_attr, oos_attr in [("lag (late minus prompt)", "lag_is", "lag_oos"),
                                    ("drift (Q5 minus Q1)", "drift_is", "drift_oos")]:
        signed = [(1 if getattr(f, is_attr) > 0 else -1) * getattr(f, oos_attr)
                  for f in folds if getattr(f, is_attr) is not None and getattr(f, oos_attr) is not None]
        if not signed:
            continue
        s = stats.summarize(signed)
        held = sum(1 for v in signed if v > 0)
        out.append(f"- {name}: the training-period direction held in {held} of {len(signed)} "
                   f"test years; mean out-of-sample spread in that direction {pct(s.mean)} "
                   f"(t across years {num(s.t)}).")
    tmb = [f.top_minus_bottom for f in folds if f.top_minus_bottom is not None]
    ics = [f.ic for f in folds if f.ic is not None]
    if tmb:
        s = stats.summarize(tmb)
        out.append(f"- combined lag + drift model: top fifth minus bottom fifth was positive in "
                   f"{sum(v > 0 for v in tmb)} of {len(tmb)} test years, mean {pct(s.mean)} "
                   f"(t across years {num(s.t)}); mean rank IC {num(_mean(ics), 3)}.")
    return out


# ---------------------------------------------------------------- coverage

def coverage_table(results: list[Result]) -> str:
    by_year: dict[int, Counter] = defaultdict(Counter)
    for x in results:
        by_year[x.event.filing_date.year][x.outcome.status] += 1
    statuses = ["ok", "no_prices", "ticker_mismatch", "no_entry_bar", "ended_before_entry",
                "not_matured"]
    rows = []
    for y in sorted(by_year):
        c = by_year[y]
        total = sum(c.values())
        rows.append([y, total] + [f"{c[s]} ({100 * c[s] / total:.0f}%)" for s in statuses])
    return table(["filing year", "events"] + statuses, rows)


def ended_early_share(results: list[Result], h: int) -> tuple[int, int]:
    legs = [x.outcome.legs[h] for x in results if h in x.outcome.legs]
    return sum("ended_early" in leg["flags"] for leg in legs), len(legs)


# ---------------------------------------------------------------- source comparison

def _key(x: Result) -> tuple:
    e = x.event
    return (e.filer_id, e.ticker, e.side, e.filing_date)


def source_comparison(primary: list[Result], other: list[Result], primary_name: str,
                      other_name: str, h: int = PRIMARY_H) -> list[str]:
    """How much do results depend on the price vendor, and on delisted names?

    Compares the two sources on the events both can price, then uses the second
    source's coverage of names the first is missing (mostly delisted) to measure
    survivorship: are the events the primary source cannot see different?
    """
    a = {_key(x): x for x in primary}
    b = {_key(x): x for x in other}
    both = [(a[k], b[k]) for k in a.keys() & b.keys()
            if a[k].ok and b[k].ok and a[k].r(h) is not None and b[k].r(h) is not None]
    lines = [f"## Price source check: {primary_name} vs {other_name}", ""]
    if not both:
        return lines + ["No events priced by both sources.", ""]
    diffs = sorted(abs(x.outcome.legs[h]["stock"] - y.outcome.legs[h]["stock"]) for x, y in both)
    ra = [x.r(h) for x, _ in both]
    rb = [y.r(h) for _, y in both]
    within = sum(d <= 0.005 for d in diffs) / len(diffs)
    lines.append(
        f"Events priced by both: {len(both)}. {h}-day stock return: median absolute "
        f"difference {pct(diffs[len(diffs) // 2], 3)}, {100 * within:.1f}% within 0.5 pp, "
        f"correlation of excess returns {num(stats.pearson(ra, rb), 3)}. Mean excess vs SPY "
        f"{pct(sum(ra) / len(ra))} ({primary_name}) vs {pct(sum(rb) / len(rb))} ({other_name}).")
    mism = Counter((a[k].outcome.status, b[k].outcome.status) for k in a.keys() & b.keys()
                   if a[k].outcome.status != b[k].outcome.status)
    if mism:
        lines.append("Status disagreements (primary, other): " + ", ".join(
            f"{p}/{o} {n}" for (p, o), n in mism.most_common(8)) + ".")
    lines.append("")

    buys = lambda rs: [x for x in rs if x.ok and x.event.side == "buy"  # noqa: E731
                       and not x.event.is_10b5_1 and x.r(h) is not None]
    other_buys = buys(other)
    seen = {_key(x) for x in buys(primary)}
    missing = [x for x in other_buys if _key(x) not in seen]
    present = [x for x in other_buys if _key(x) in seen]
    rows = []
    for label, rs in [(f"priced by {primary_name} too", present),
                      (f"missing from {primary_name}", missing),
                      (f"all {other_name} buys", other_buys)]:
        vals, keys = _values(rs, h)
        s = stats.summarize(vals, keys)
        early, legs = ended_early_share(rs, h)
        rows.append([label, s.n, pct(s.mean), num(s.t_cluster), pct(s.median), pct(s.hit, 1),
                     f"{early} of {legs}"])
    lines.append(f"**Survivorship: {other_name} buy events, {h}-day excess vs SPY**\n")
    lines.append(table(["events", "n", "mean", "t (clustered)", "median", "hit rate",
                        "history ends before exit"], rows))
    lines.append("")
    lines.append(f"**{other_name} buys by lag, {h} days**\n")
    lag = by_lag(other_buys)
    vals = [v for x in other_buys if (v := x.r(h)) is not None]
    w = stats.winsor_cutoffs(vals) if vals else None
    lines.append(table(SUMMARY_HEADERS, [summary_row(LAG_BUCKETS[i][0], lag.get(i, []), h, w)
                                         for i in range(len(LAG_BUCKETS))]))
    lines.append("")
    return lines


# ---------------------------------------------------------------- report

def dedupe_ticker_day(results: list[Result]) -> list[Result]:
    """One event per stock, side and entry day (the earliest-filed insider's)."""
    seen, out = set(), []
    for x in results:
        k = (x.event.ticker, x.event.side, x.outcome.entry)
        if k not in seen:
            seen.add(k)
            out.append(x)
    return out


def section_for(title: str, results: list[Result]) -> str:
    if not results:
        return f"## {title}\n\nNo events.\n"
    cutoffs = drift_cutoffs(results)
    parts = [f"## {title}\n"]
    for h in HORIZONS:
        hv = [v for x in results if (v := x.r(h)) is not None]
        parts.append(bucket_tables(results, h, cutoffs, stats.winsor_cutoffs(hv) if hv else None))
        parts.append("")
    groups = [
        ("role: CEO or CFO", lambda e: e.role in ("CEO", "CFO")),
        ("role: other officer", lambda e: e.role == "officer"),
        ("role: director", lambda e: e.role == "director"),
        ("role: 10% owner", lambda e: e.role == "10% owner"),
        ("role: none given", lambda e: e.role is None),
        ("cluster: 1 insider", lambda e: e.cluster_count <= 1),
        ("cluster: 2 insiders", lambda e: e.cluster_count == 2),
        ("cluster: 3+ insiders", lambda e: e.cluster_count >= 3),
    ]
    hv = [v for x in results if (v := x.r(PRIMARY_H)) is not None]
    w = stats.winsor_cutoffs(hv) if hv else None
    parts.append(f"**By role and cluster, {PRIMARY_H} days** (context for scoring; cluster "
                 "counts include 10b5-1 trades by other insiders)\n")
    parts.append(table(SUMMARY_HEADERS, [summary_row(label, [x for x in results if f(x.event)],
                                                     PRIMARY_H, w) for label, f in groups]))
    parts.append("")
    parts.append(f"**Lag x drift, mean {PRIMARY_H}-day excess vs SPY (clustered t)**\n")
    parts.append(cross_table(results, PRIMARY_H, cutoffs))
    parts.append("")

    rows = []
    for label, lo, hi in PERIODS + [(str(y), y, y) for y in
                                    sorted({x.outcome.entry.year for x in results})]:
        sub = [x for x in results if lo <= x.outcome.entry.year <= hi]
        vals, keys = _values(sub, PRIMARY_H)
        s = stats.summarize(vals, keys)
        (dl, tl, _, _), (dd, td, _, _) = contrast(sub, PRIMARY_H, cutoffs)
        rows.append([label, s.n, pct(s.mean), num(s.t_cluster), pct(s.hit, 1),
                     pct(dl), num(tl), pct(dd), num(td)])
    parts.append(f"**By period, {PRIMARY_H} days (drift quintiles from the full sample)**\n")
    parts.append(table(["period", "n", "mean vs SPY", "t (clustered)", "hit rate",
                        "late - prompt", "t (clustered)", "drift Q5 - Q1", "t (clustered)"], rows))
    parts.append("")
    folds = walk_forward(results)
    parts.append(f"**Walk-forward, {PRIMARY_H} days** (train on all earlier years, purged; "
                 "test on the next year)\n")
    parts.append(walk_forward_table(folds))
    parts.append("")
    parts += walk_forward_verdict(folds)
    parts.append("")
    return "\n".join(parts)


def render(results: list[Result], exclusions: Exclusions, source_name: str,
           extra: list[str] | None = None) -> str:
    ok = [x for x in results if x.ok]
    buys = [x for x in ok if x.event.side == "buy" and not x.event.is_10b5_1]
    sells = [x for x in ok if x.event.side == "sell" and not x.event.is_10b5_1]
    plan_sells = [x for x in ok if x.event.side == "sell" and x.event.is_10b5_1]
    ex = exclusions.counts
    early, legs = ended_early_share(buys, PRIMARY_H)
    lines = [
        "# Phase 3 backtest: returns by disclosure lag and pre-disclosure drift",
        "",
        f"Price source: **{source_name}**. Returns are excess returns, signed so positive "
        "means the insider was right (for sells, the stock lagged the benchmark). Entry is "
        "the open of the first trading day after the filing date (or after the EDGAR "
        "acceptance time when known); exit is the close h trading days later. Drift runs "
        "from the close on the first trade date to the last close before entry. Net "
        "returns subtract 10 bps per side plus a half-spread tiered by dollar volume.",
        "",
        "## Sample",
        "",
        table(["", "count"], [[k, v] for k, v in sorted(ex.items())]),
        "",
        f"Events with returns: {len(ok)} of {len(results)}. Buys (no 10b5-1) {len(buys)}, "
        f"sells (no 10b5-1) {len(sells)}, 10b5-1 sells {len(plan_sells)}. "
        f"Buy events whose price history ends before the {PRIMARY_H}-day exit "
        f"(delisted, or the vendor stops): {early} of {legs}.",
        "",
        "**Coverage by filing year** (survivorship: `no_prices` is mostly delisted names)\n",
        coverage_table(results),
        "",
    ]
    lines += extra or []
    lines.append(section_for("Buys (open-market purchases, no 10b5-1)", buys))
    liquid = [x for x in buys if x.liquid]
    lines.append(section_for(
        f"Buys, liquid only (entry price >= ${MIN_LIQUID_PRICE:.0f}, median dollar volume "
        f">= ${MIN_LIQUID_DOLLAR_VOLUME:,})", liquid))
    lines.append(section_for("Buys, one event per stock and entry day", dedupe_ticker_day(buys)))
    lines.append(section_for("Sells (no 10b5-1)", sells))
    lines.append(section_for("Sells under 10b5-1 plans", plan_sells))
    return "\n".join(lines)


# ---------------------------------------------------------------- events file

EVENT_COLUMNS = [
    "filer_id", "issuer_cik", "ticker", "side", "filing_date", "accepted_at",
    "first_trade_date", "last_trade_date", "lag_days", "lag_ratio", "value", "avg_price",
    "role", "cluster_count", "is_10b5_1", "n_lines", "status", "flags", "entry",
    "entry_open", "drift_pct", "price_ratio", "dollar_volume", "cost", "sector",
] + [f"{k}_{h}" for h in HORIZONS for k in ("exit", "stock", "spy", "sector", "iwm")]


def write_events(results: list[Result], path: Path) -> None:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(EVENT_COLUMNS)
    for x in results:
        e, o = x.event, x.outcome
        row = [e.filer_id, e.issuer_cik, e.ticker, e.side, e.filing_date, e.accepted_at or "",
               e.first_trade_date, e.last_trade_date, e.lag_days, e.lag_ratio,
               round(e.value, 2), num(e.avg_price, 4), e.role or "", e.cluster_count,
               int(e.is_10b5_1), e.n_lines, o.status, ";".join(o.flags), o.entry or "",
               num(o.entry_open, 4), num(o.drift_pct, 4), num(o.price_ratio, 4),
               num(o.dollar_volume, 0), num(o.cost, 5), o.benchmark or x.sector]
        for h in HORIZONS:
            leg = o.legs.get(h) or {}
            row += [leg.get("exit", "")] + [num(leg.get(k), 6) for k in ("stock", "spy", "sector", "iwm")]
        w.writerow(row)
    path.write_bytes(gzip.compress(buf.getvalue().encode()))
