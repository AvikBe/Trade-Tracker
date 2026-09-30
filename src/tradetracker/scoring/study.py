"""Phase 4 backtest: does the score pick buys that clear costs out of sample?

Walk-forward by year, as in phase 3: fit the model on every earlier year (purged, so no
training return overlaps the test year) and score the test year with it. Pooled over
the test years, this gives the spec's go-live metrics:

- mean 20-day excess return over SPY, net of costs, of the top-decile scores, and the
  number of test years in which it is positive;
- the hit rate (share of those buys that beat SPY over 20 days);
- the information ratio and drawdown of a top-20 portfolio that buys each day's signals
  above the training top-decile cutoff and holds them for 20 sessions;
- the same split into 2017-2022 and 2023-2026, an ablation that drops one component at
  a time, and what is left of the edge when entry comes d sessions late (the decay).

The sample is what the tracker would show: open-market buys, no 10b5-1 plan, liquid
(entry at least $2, median dollar volume at least $100k), drift at most 20%. One stock
bought by several insiders the same day is one position: the best-scored event is kept.
"""

from __future__ import annotations

import csv
import gzip
import io
import math
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

from ..backtest import stats
from ..backtest.returns import HORIZONS
from ..backtest.study import PRIMARY_H, Result, num, pct, table
from . import model as M
from .signals import Signals

MAX_DRIFT_SHOWN = 0.20
MIN_TRAIN_YEARS = 2
PORTFOLIO_SLOTS = 20
DELAYS = (0, 1, 2, 3, 5, 10)
PERIODS = [("2017-2022", 2017, 2022), ("2023-2026", 2023, 2026)]


BETA_CLIP = (0.0, 3.0)
MAX_WEIGHT = 2.0             # a vol-scaled position is at most 2 slots


@dataclass(frozen=True)
class Config:
    """What is being tested: horizon, what returns are measured against, the sample.

    - `bench`: "spy" (phase 4), "sector", "iwm", or "beta" for a beta-hedged return:
      stock minus beta x SPY, with the beta measured before entry (`add_beta_legs`).
    - `min_cap`: hide events whose market cap at entry is below this or unknown (the
      spec's $50M floor), or None for the phase 4 liquidity stand-in alone.
    - `vol_target`: size portfolio positions at vol_target / the stock's volatility
      (at most `MAX_WEIGHT` slots), or None for equal slots.
    """
    name: str = "phase 4"
    h: int = PRIMARY_H
    bench: str = "spy"
    min_cap: float | None = None
    vol_target: float | None = None

    def label(self) -> str:
        parts = [f"{self.h} days", "beta-hedged" if self.bench == "beta" else f"vs {self.bench.upper()}"]
        if self.min_cap:
            parts.append(f"cap >= ${self.min_cap / 1e6:.0f}M")
        if self.vol_target:
            parts.append(f"vol-sized to {100 * self.vol_target:.0f}%")
        return ", ".join(parts)


PHASE4 = Config()
# The variants avik asked to try on 2026-09-30: the spec's $50M market-cap floor, and
# longer holds with volatility control (beta hedge plus volatility-scaled positions).
VARIANTS = [
    PHASE4,
    Config("cap floor", min_cap=50e6),
    Config("cap floor, 60 days", h=60, min_cap=50e6),
    Config("cap floor, 20 days, vol control", bench="beta", min_cap=50e6, vol_target=0.40),
    Config("cap floor, 60 days, vol control", h=60, bench="beta", min_cap=50e6,
           vol_target=0.40),
]


def hedge_beta(s: Signals) -> float:
    b = 1.0 if s.beta is None else s.beta
    return min(max(b, BETA_CLIP[0]), BETA_CLIP[1])


def add_beta_legs(results: list[Result], signals: list[Signals]) -> None:
    """Give every leg a "beta" benchmark: beta (clipped, 1 when unknown) x SPY's return."""
    for x, s in zip(results, signals):
        b = hedge_beta(s)
        for leg in x.outcome.legs.values():
            leg["beta"] = None if leg.get("spy") is None else b * leg["spy"]


def shown(x: Result, s: Signals, cfg: Config = PHASE4) -> bool:
    """Would the tracker list this event? (The spec's hide rules, and the liquid sample.)"""
    e = x.event
    return (x.ok and e.side == "buy" and not e.is_10b5_1 and x.liquid
            and (s.drift is None or s.drift <= MAX_DRIFT_SHOWN)
            and (cfg.min_cap is None or (s.market_cap is not None and s.market_cap >= cfg.min_cap)))


@dataclass
class Scored:
    result: Result
    signals: Signals
    score: float
    quality: float
    year: int
    above: bool          # at or above the training top-decile cutoff

    def r(self, h: int = PRIMARY_H, bench: str = "spy", net: bool = False):
        return self.result.r(h, bench, net)


@dataclass
class Fold:
    year: int
    model: M.Model
    n_train: int
    scored: list[Scored]


def training_set(pool, h, start, bench: str = "spy"):
    xs, ss, ys = [], [], []
    for x, s in pool:
        v = x.r(h, bench, net=True)
        if v is not None and x.outcome.legs[h]["exit"] < start:
            xs.append(x)
            ss.append(s)
            ys.append(v)
    return xs, ss, ys


def dedupe(scored: list[Scored]) -> list[Scored]:
    """One entry per stock and entry day: the best-scored insider's."""
    best: dict[tuple, Scored] = {}
    for x in scored:
        k = (x.result.event.ticker, x.result.outcome.entry)
        if k not in best or x.score > best[k].score:
            best[k] = x
    return sorted(best.values(), key=lambda x: (x.result.outcome.entry, -x.score,
                                                x.result.event.ticker))


def walk_forward(results: list[Result], signals: list[Signals], h: int | None = None,
                 components: list[str] | None = None, beta: float | None = None,
                 min_train_years: int = MIN_TRAIN_YEARS, cfg: Config = PHASE4) -> list[Fold]:
    if h is not None:
        cfg = replace(cfg, h=h)
    pool = [(x, s) for x, s in zip(results, signals) if shown(x, s, cfg)]
    years = sorted({x.outcome.entry.year for x, _ in pool})
    folds = []
    for y in years[min_train_years:]:
        _, ss, ys = training_set(pool, cfg.h, date(y, 1, 1), cfg.bench)
        if len(ys) < 500:
            continue
        m = M.fit(ss, ys, label=f"before {y}", components=components, beta=beta)
        test = []
        for x, s in pool:
            if x.outcome.entry.year == y:
                sc = m.score(s)
                test.append(Scored(x, s, sc, m.quality(s), y, sc >= m.threshold))
        folds.append(Fold(y, m, len(ys), dedupe(test)))
    return folds


# ---------------------------------------------------------------- metrics

def _summ(rows: list[Scored], h: int = PRIMARY_H, bench: str = "spy", net: bool = True):
    vals, keys = [], []
    for x in rows:
        v = x.r(h, bench, net)
        if v is not None:
            vals.append(v)
            keys.append(stats.cluster_key(x.result.outcome.entry, h))
    return stats.summarize(vals, keys)


def top_decile_by_rank(rows: list[Scored]) -> list[Scored]:
    k = max(1, len(rows) // 10)
    return sorted(rows, key=lambda x: -x.score)[:k]


def hit_rate(rows: list[Scored], h: int = PRIMARY_H, bench: str = "spy") -> float | None:
    v = [x.r(h, bench) for x in rows if x.r(h, bench) is not None]
    return sum(r > 0 for r in v) / len(v) if v else None


def year_table(folds: list[Fold], cfg: Config = PHASE4) -> tuple[str, dict]:
    rows, pooled_top, pooled_all, pooled_rank = [], [], [], []
    yearly = []
    h, bench = cfg.h, cfg.bench
    for f in folds:
        top = [x for x in f.scored if x.above]
        rank = top_decile_by_rank(f.scored)
        s_all, s_top, s_rank = (_summ(f.scored, h, bench), _summ(top, h, bench),
                                _summ(rank, h, bench))
        pairs = [(x.score, v) for x in f.scored if (v := x.r(h, bench, net=True)) is not None]
        ic = stats.spearman([a for a, _ in pairs], [b for _, b in pairs])
        yearly.append(s_top.mean)
        rows.append([f.year, f.n_train, s_all.n, pct(s_all.mean), s_top.n, pct(s_top.mean),
                     num(s_top.t_cluster), pct(hit_rate(top, h, bench), 1), pct(s_rank.mean),
                     num(ic, 3), num(f.model.beta, 1)])
        pooled_top += top
        pooled_all += f.scored
        pooled_rank += rank
    return (table(["test year", "train n", "shown", "all, net", "above cutoff",
                   "their net mean", "t (clustered)", "hit rate", "top 10% by rank, net",
                   "rank IC", "beta"], rows),
            {"top": pooled_top, "all": pooled_all, "rank": pooled_rank, "yearly": yearly})


# ---------------------------------------------------------------- portfolio

def _daily(bars, d: date, first: bool) -> float | None:
    b = bars.at(d)
    if b is None:
        return None
    if first:
        return b.adj_close / b.adj_open - 1 if b.adj_open else None
    i = bars.index_before(d)
    return b.adj_close / bars.bars[i].adj_close - 1 if i is not None else None


@dataclass
class Portfolio:
    days: list[date]
    excess: list[float]        # daily excess over SPY, after costs
    spy: list[float]
    positions: list[int]
    trades: int

    def ir(self) -> float | None:
        if len(self.excess) < 20:
            return None
        m = sum(self.excess) / len(self.excess)
        sd = math.sqrt(sum((v - m) ** 2 for v in self.excess) / (len(self.excess) - 1))
        return m / sd * math.sqrt(252) if sd > 0 else None

    def annual_excess(self) -> float:
        return sum(self.excess) / len(self.excess) * 252 if self.excess else 0.0

    @staticmethod
    def max_drawdown(rets: list[float]) -> float:
        peak = level = 1.0
        worst = 0.0
        for r in rets:
            level *= 1 + r
            peak = max(peak, level)
            worst = min(worst, level / peak - 1)
        return worst

    def mdd(self) -> float:
        return self.max_drawdown([s + e for s, e in zip(self.spy, self.excess)])

    def spy_mdd(self) -> float:
        return self.max_drawdown(self.spy)


def position_weight(s: Signals, slots: int, vol_target: float | None) -> float:
    if vol_target is None or not s.vol:
        return 1.0 / slots
    return min(MAX_WEIGHT, vol_target / s.vol) / slots


def simulate(candidates: list[Scored], prices, h: int = PRIMARY_H,
             slots: int = PORTFOLIO_SLOTS, hedged: bool = False,
             vol_target: float | None = None) -> Portfolio:
    """Slots of 1/slots of the book each, held h sessions.

    Empty slots sit in SPY, so the book's excess over SPY is the sum of held positions'
    daily excess times their weight. Costs (the event's round trip) are charged half at
    entry and half at exit. A stock with no bar on a day contributes 0 that day.

    Volatility control: with `hedged`, each position shorts beta x its weight in SPY
    (the beta measured before entry), and with `vol_target` a position's weight is
    vol_target / its volatility, capped at `MAX_WEIGHT` slots.
    """
    spy = prices.get("SPY")
    by_day: dict[date, list[Scored]] = defaultdict(list)
    for x in candidates:
        by_day[x.result.outcome.entry].append(x)
    if not by_day:
        return Portfolio([], [], [], [], 0)
    start = min(by_day)
    end = max(leg["exit"] for x in candidates if (leg := x.result.outcome.legs.get(h)))
    days = [d for d in spy.dates if start <= d <= end]
    held: list[tuple[Scored, date]] = []      # (position, exit day)
    out = Portfolio([], [], [], [], 0)
    for d in days:
        todays = sorted(by_day.get(d, []), key=lambda x: -x.score)
        tickers = {p.result.event.ticker for p, _ in held}
        for x in todays:
            leg = x.result.outcome.legs.get(h)
            if len(held) >= slots or leg is None or x.result.event.ticker in tickers:
                continue
            held.append((x, leg["exit"]))
            tickers.add(x.result.event.ticker)
            out.trades += 1
        spy_r = _daily(spy, d, False) or 0.0
        total = 0.0
        for p, exit_day in held:
            e = p.result.event
            first = d == p.result.outcome.entry
            bars = prices.get(e.ticker)
            r = _daily(bars, d, first) if bars is not None else None
            spy_leg = (_daily(spy, d, True) if first else spy_r) or 0.0
            b = hedge_beta(p.signals) if hedged else 1.0
            ex = (r - b * spy_leg) if r is not None else 0.0
            cost = p.result.outcome.cost or 0.0
            if first:
                ex -= cost / 2
            if d == exit_day:
                ex -= cost / 2
            total += ex * position_weight(p.signals, slots, vol_target)
        out.days.append(d)
        out.excess.append(total)
        out.spy.append(spy_r)
        out.positions.append(len(held))
        held = [(p, x) for p, x in held if x > d]
    while out.positions and out.positions[-1] == 0:   # a rejected candidate's tail
        for series in (out.days, out.excess, out.spy, out.positions):
            series.pop()
    return out


def portfolio_by_year(pf: Portfolio) -> str:
    rows = []
    years = sorted({d.year for d in pf.days})
    for y in years:
        ex = [e for d, e in zip(pf.days, pf.excess) if d.year == y]
        sp = [s for d, s in zip(pf.days, pf.spy) if d.year == y]
        pos = [p for d, p in zip(pf.days, pf.positions) if d.year == y]
        sub = Portfolio([], ex, sp, pos, 0)
        rows.append([y, len(ex), num(sum(pos) / len(pos), 1), pct(sub.annual_excess()),
                     num(sub.ir()), pct(sub.mdd()), pct(sub.spy_mdd())])
    return table(["year", "days", "avg positions", "excess vs SPY (annualized)", "IR",
                  "max drawdown", "SPY max drawdown"], rows)


# ---------------------------------------------------------------- decay

def delayed_excess(x: Scored, prices, delay: int, h: int = PRIMARY_H,
                   hedged: bool = False) -> float | None:
    """Excess over SPY, or beta x SPY when hedged (net of costs), entering late."""
    bars = prices.get(x.result.event.ticker)
    spy = prices.get("SPY")
    entry = x.result.outcome.entry
    if bars is None:
        return None
    i = bars.index_on_or_after(entry)
    if i is None or i + delay + h - 1 >= len(bars):
        return None
    a, b = bars.bars[i + delay], bars.bars[i + delay + h - 1]
    si, sj = spy.index_on_or_after(a.date), spy.index_on_or_before(b.date)
    if si is None or sj is None or spy.dates[si] != a.date or not a.adj_open:
        return None
    s = spy.bars[sj].adj_close / spy.bars[si].adj_open - 1
    if hedged:
        s *= hedge_beta(x.signals)
    return b.adj_close / a.adj_open - 1 - s - (x.result.outcome.cost or 0.0)


def decay_table(top: list[Scored], prices, h: int = PRIMARY_H,
                hedged: bool = False) -> tuple[str, float | None]:
    means, ts = {}, {}
    rows = []
    for d in DELAYS:
        v = [r for x in top if (r := delayed_excess(x, prices, d, h, hedged)) is not None]
        s = stats.summarize(v)
        means[d], ts[d] = s.mean, s.t
        rows.append([d, s.n, pct(s.mean), num(s.t)])
    half = None
    base, base_t = means.get(0), ts.get(0)
    # A half-life is only worth fitting to an edge that is there (t >= 2 undelayed).
    if base is not None and base > 0 and base_t is not None and base_t >= 2:
        for d in DELAYS[1:]:
            if means[d] is not None and means[d] <= base / 2:
                half = d
                break
    what = "beta-hedged excess" if hedged else "excess vs SPY"
    return table(["entry delay (sessions)", "n", f"{h}-day net {what}", "t"], rows), half


# ---------------------------------------------------------------- report

def weights_table(folds: list[Fold]) -> str:
    names = list(M.COMPONENTS)
    rows = [[f.year] + [num(1e4 * f.model.weights.get(c, 0.0) *
                            (max(f.model.tables[c].values()) - min(f.model.tables[c].values())), 0)
                        if c in f.model.weights else "" for c in names]
            + [num(f.model.beta, 1)] for f in folds]
    return table(["test year"] + names + ["beta"], rows)


def model_tables(m: M.Model) -> str:
    parts = []
    for c, (_, labels) in M.COMPONENTS.items():
        if c not in m.tables:
            continue
        rows = [[b, m.counts[c].get(b, 0), pct(m.tables[c].get(b)),
                 pct(m.weights[c] * m.tables[c][b]) if b in m.tables[c] else ""]
                for b in labels if b in m.counts[c]]
        parts.append(f"**{c}** (weight {num(m.weights[c], 3)})\n")
        parts.append(table(["bucket", "train n", "bucket value (shrunk)", "weighted"], rows))
        parts.append("")
    return "\n".join(parts)


def criteria(pooled: dict, pf: Portfolio, cfg: Config = PHASE4) -> tuple[str, list[bool]]:
    """The spec's four go-live checks. The return bar is per 20 days, so a longer hold's
    mean is scaled to 20 days (x 20 / h) before it's compared with 1%."""
    s = _summ(pooled["top"], cfg.h, cfg.bench)
    per20 = None if s.mean is None else s.mean * PRIMARY_H / cfg.h
    yearly = [v for v in pooled["yearly"] if v is not None]
    pos = sum(v > 0 for v in yearly)
    hit = hit_rate(pooled["top"], cfg.h, cfg.bench)
    ir = pf.ir()
    mdd, smdd = pf.mdd(), pf.spy_mdd()
    ratio = mdd / smdd if smdd < 0 else None
    checks = [
        (per20 is not None and per20 > 0.01 and pos > len(yearly) / 2,
         "20-day excess return, top-decile scores", "above 1% after costs, positive in most "
         "test years", (f"{pct(s.mean)} over {cfg.h} days = {pct(per20)} per 20" if cfg.h != PRIMARY_H
                        else pct(s.mean)) + f" (t {num(s.t_cluster)}), positive in {pos} of "
         f"{len(yearly)}"),
        (ir is not None and ir > 0.5, "Information ratio, top-20 portfolio",
         "above 0.5 out of sample", num(ir)),
        (hit is not None and hit > 0.55, "Hit rate (beats the benchmark over the hold)",
         "above 55%", pct(hit, 1)),
        (ratio is not None and ratio <= 1.5, "Max drawdown vs SPY", "no worse than 1.5x",
         f"{pct(mdd)} vs SPY {pct(smdd)} ({num(ratio)}x)"),
    ]
    rows = [[name, target, got, "pass" if ok else "fail"] for ok, name, target, got in checks]
    return table(["metric", "target", "out of sample", ""], rows), [c[0] for c in checks]


def ablation(results, signals, base_top, cfg: Config = PHASE4) -> str:
    rows = []
    b = _summ(base_top, cfg.h, cfg.bench)
    rows.append(["all components", b.n, pct(b.mean), num(b.t_cluster)])
    variants = [(f"without {c}", [n for n in M.COMPONENTS if n != c], None)
                for c in M.COMPONENTS]
    variants.append(("no drift penalty (beta 0)", None, 0.0))
    for label, comps, beta in variants:
        folds = walk_forward(results, signals, components=comps, beta=beta, cfg=cfg)
        top = [x for f in folds for x in f.scored if x.above]
        s = _summ(top, cfg.h, cfg.bench)
        rows.append([label, s.n, pct(s.mean), num(s.t_cluster)])
    return table(["model", "above cutoff", f"net mean, {cfg.h} days", "t (clustered)"], rows)


def run_config(results, signals, prices, cfg: Config) -> tuple[list[Fold], dict, Portfolio]:
    folds = walk_forward(results, signals, cfg=cfg)
    _, pooled = year_table(folds, cfg)
    pf = simulate([x for f in folds for x in f.scored if x.above], prices, cfg.h,
                  hedged=cfg.bench == "beta", vol_target=cfg.vol_target)
    return folds, pooled, pf


def variants_table(results, signals, prices, configs=VARIANTS) -> tuple[str, dict]:
    """The same model under each configuration: one row per variant, all out of sample."""
    rows, out = [], {}
    for cfg in configs:
        folds, pooled, pf = run_config(results, signals, prices, cfg)
        s = _summ(pooled["top"], cfg.h, cfg.bench)
        a = _summ(pooled["all"], cfg.h, cfg.bench)
        yearly = [v for v in pooled["yearly"] if v is not None]
        _, ok = criteria(pooled, pf, cfg)
        mdd, smdd = pf.mdd(), pf.spy_mdd()
        rows.append([cfg.name, cfg.label(), a.n, pct(a.mean), s.n, pct(s.mean),
                     num(s.t_cluster), f"{sum(v > 0 for v in yearly)} of {len(yearly)}",
                     pct(hit_rate(pooled["top"], cfg.h, cfg.bench), 1), num(pf.ir()),
                     num(mdd / smdd if smdd < 0 else None), f"{sum(ok)} of 4"])
        out[cfg.name] = summary(folds, pf, cfg)
    return table(["variant", "setup", "shown", "all, net", "above cutoff", "their net mean",
                  "t (clustered)", "years positive", "hit rate", "portfolio IR",
                  "drawdown vs SPY", "criteria met"], rows), out


def render(results: list[Result], signals: list[Signals], prices, final: M.Model,
           source_name: str, run_ablation: bool = True, cfg: Config = PHASE4,
           variants: str | None = None) -> tuple[str, list[Fold], Portfolio]:
    folds, pooled, pf = run_config(results, signals, prices, cfg)
    ytable, _ = year_table(folds, cfg)
    h, bench = cfg.h, cfg.bench
    hedged = bench == "beta"
    crit, _ = criteria(pooled, pf, cfg)
    pool_n = sum(1 for x, s in zip(results, signals) if shown(x, s, cfg))
    what = "beta-hedged excess (stock minus beta x SPY)" if hedged else f"excess over {bench.upper()}"

    lines = [
        f"# Score backtest, out of sample: {cfg.name} ({cfg.label()})",
        "",
        f"Price source: **{source_name}**. Score = Q x max(0, 1 - beta x drift), with no "
        "promptness weight (P = 1). Walk-forward: each test year is scored by a model fitted "
        f"only on earlier years whose {h}-day exits fell before the test year began. Returns "
        f"are {h}-day {what} after costs unless a column says otherwise.",
        "",
        "Shown events (liquid open-market buys, no 10b5-1, drift <= 20%"
        + (f", market cap >= ${cfg.min_cap / 1e6:.0f}M" if cfg.min_cap else "")
        + f"): {pool_n}. Test years: {', '.join(str(f.year) for f in folds)}.",
        "",
        "## Spec go-live criteria",
        "",
        crit,
        "",
    ]
    if variants:
        lines += ["## Variants", "",
                  "The same model and walk-forward under each setup. Every number is out of "
                  "sample, but picking the best row after seeing them all is a choice made "
                  "in sample: five setups were tried.", "", variants, ""]
    lines += [
        "## By test year",
        "",
        "\"Above cutoff\" is what the live tracker would flag: a score at or above the "
        "training years' top-decile score. \"Top 10% by rank\" ranks within the test year, "
        "which uses the year's own distribution and so is not tradable, but shows ordering.",
        "",
        ytable,
        "",
    ]
    rows = []
    for label, lo, hi in PERIODS:
        top = [x for x in pooled["top"] if lo <= x.year <= hi]
        al = [x for x in pooled["all"] if lo <= x.year <= hi]
        st, sa = _summ(top, h, bench), _summ(al, h, bench)
        rows.append([label, sa.n, pct(sa.mean), st.n, pct(st.mean), num(st.t_cluster),
                     pct(hit_rate(top, h, bench), 1)])
    lines += ["## By period", "", table(["period", "shown", "all, net", "above cutoff",
                                         "their net mean", "t (clustered)", "hit rate"], rows), ""]

    rows = []
    for hh in HORIZONS:
        for bb in ("spy", "sector", "iwm", "beta"):
            g, n = _summ(pooled["top"], hh, bb, net=False), _summ(pooled["top"], hh, bb)
            a = _summ(pooled["all"], hh, bb)
            name = {"sector": "sector ETF", "beta": "beta x SPY"}.get(bb, bb.upper())
            rows.append([hh, name, g.n, pct(g.mean), pct(n.mean), num(n.t_cluster),
                         pct(a.mean)])
    lines += ["## Above-cutoff buys by horizon and benchmark", "",
              table(["days", "benchmark", "n", "gross", "net", "t (clustered)",
                     "all shown, net"], rows), ""]

    lines += ["## Top-20 portfolio", "",
              f"{pf.trades} positions over {len(pf.days)} sessions; average "
              f"{num(sum(pf.positions) / max(1, len(pf.positions)), 1)} of "
              f"{PORTFOLIO_SLOTS} slots filled, each held {h} sessions"
              + (", beta-hedged" if hedged else "")
              + (f", sized to {100 * cfg.vol_target:.0f}% volatility" if cfg.vol_target else "")
              + f". Annualized excess {'after the beta hedge' if hedged else 'over SPY'} "
              f"{pct(pf.annual_excess())}, information ratio "
              f"{num(pf.ir())}.", "",
              portfolio_by_year(pf), ""]

    dtable, half = decay_table(pooled["top"], prices, h, hedged)
    lines += ["## Decay: entering late", "",
              "Above-cutoff buys, entered d sessions after the first possible open and held "
              f"{h} sessions. The spec's lambda sets the half-life of this edge.", "",
              dtable, "",
              (f"The edge halves by a delay of {half} sessions." if half is not None else
               "No half-life was fitted (the undelayed edge is not significantly positive, "
               f"t < 2, or never halves); the model keeps the spec's {M.DEFAULT_HALF_LIFE}-"
               "session start value."), ""]

    lines += ["## Component weights by fold", "",
              "Each cell is the component's spread in basis points: weight x (best bucket - "
              "worst bucket). 0 means the fold dropped it.", "", weights_table(folds), ""]
    if run_ablation:
        lines += ["## Ablation: drop one component", "",
                  ablation(results, signals, pooled["top"], cfg), ""]
    lines += [f"## Final model (fitted on all years, {final.n_train} events)", "",
              f"beta {num(final.beta, 1)}, half-life {num(final.half_life, 0)} sessions, "
              f"top-decile cutoff {num(final.threshold, 3)}.", "", model_tables(final)]
    return "\n".join(lines), folds, pf


def summary(folds: list[Fold], pf: Portfolio, cfg: Config = PHASE4) -> dict:
    """Headline numbers, for the cross-check script and later comparison."""
    out = {"config": dict(cfg.__dict__), "years": {}, "portfolio": {"ir": pf.ir(), "annual_excess": pf.annual_excess(),
                                      "mdd": pf.mdd(), "spy_mdd": pf.spy_mdd(),
                                      "trades": pf.trades, "days": len(pf.days)}}
    for f in folds:
        top = [x for x in f.scored if x.above]
        s = _summ(top, cfg.h, cfg.bench)
        out["years"][str(f.year)] = {"shown": len(f.scored), "above": s.n, "net_mean": s.mean,
                                     "hit": hit_rate(top, cfg.h, cfg.bench),
                                     "beta": f.model.beta,
                                     "threshold": f.model.threshold, "n_train": f.n_train}
    pooled = [x for f in folds for x in f.scored if x.above]
    s = _summ(pooled, cfg.h, cfg.bench)
    out["pooled"] = {"above": s.n, "net_mean": s.mean, "t_cluster": s.t_cluster,
                     "hit": hit_rate(pooled, cfg.h, cfg.bench)}
    return out


def fit_final(results, signals, h: int | None = None, half_life: float | None = None,
              cfg: Config = PHASE4) -> M.Model:
    if h is not None:
        cfg = replace(cfg, h=h)
    pool = [(x, s) for x, s in zip(results, signals) if shown(x, s, cfg)]
    _, ss, ys = training_set(pool, cfg.h, date.max, cfg.bench)
    m = M.fit(ss, ys, label="all years")
    if half_life:
        m.half_life = half_life
    return m


# `target` is what the model is fitted on and judged by: the net return over cfg.h
# sessions against cfg.bench, exiting on `exit`. `gross` is the same before costs.
SCORED_COLUMNS = ["year", "filer_id", "issuer_cik", "ticker", "filing_date", "entry", "exit", "score",
                  "quality", "above", "role", "value", "stake_change", "cluster_count", "repeat_buys",
                  "drawdown", "track_record", "track_n", "dollar_volume", "drift", "cost",
                  "market_cap", "beta", "vol", "target", "gross", "excess_5", "excess_20", "excess_60"]


def _write(rows: list[Scored], path: Path, cfg: Config = PHASE4) -> None:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(SCORED_COLUMNS)
    for x in rows:
        e, o, s = x.result.event, x.result.outcome, x.signals
        leg = o.legs.get(cfg.h) or {}
        w.writerow([x.year, e.filer_id, e.issuer_cik or "", e.ticker, e.filing_date, o.entry, leg.get("exit", ""),
                    repr(x.score), repr(x.quality), int(x.above), s.role or "",
                    repr(s.value), _r(s.stake_change), s.cluster_count,
                    s.repeat_buys, _r(s.drawdown), _r(s.track_record), s.track_n,
                    _r(s.dollar_volume), _r(s.drift), _r(o.cost),
                    _r(s.market_cap), _r(s.beta), _r(s.vol),
                    _r(x.r(cfg.h, cfg.bench, net=True)), _r(x.r(cfg.h, cfg.bench))]
                   + [_r(x.r(h)) for h in HORIZONS])
    path.write_bytes(gzip.compress(buf.getvalue().encode()))


def _r(v) -> str:
    return "" if v is None else repr(v)


def write_scored(folds: list[Fold], path: Path, cfg: Config = PHASE4) -> None:
    """Every test-year event with its fold's score (full float precision)."""
    _write([x for f in folds for x in f.scored], path, cfg)


def write_pool(results: list[Result], signals: list[Signals], path: Path,
               cfg: Config = PHASE4) -> None:
    """Every shown event of every year, unscored: the training data of all folds."""
    rows = [Scored(x, s, 0.0, 0.0, x.outcome.entry.year, False)
            for x, s in zip(results, signals) if shown(x, s, cfg)]
    _write(rows, path, cfg)
