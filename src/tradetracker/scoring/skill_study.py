"""Does an insider's record of beating SPY carry over to their next buy?

Two questions, both out of sample and point in time (see `tradetracker.skill`):

1. **Persistence, no model.** Shown buys grouped by the insider's record at that hold,
   year by year: do insiders whose earlier buys beat SPY beat it again?
2. **Use in the paper-traded setup.** The two-input score (52-week drawdown and run-up
   penalty) with the record as a third input in Q, as a gate on what the model flags,
   and alone, walk-forward with the same go-live criteria (`study.SKILL_VARIANTS`).
   A gate is tested against random gates that keep the same number of flagged buys
   each year, so "fewer, better buys" can't come from luck in which buys were dropped.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import replace

from .. import skill as SK
from ..backtest import stats
from ..backtest.study import num, pct, table
from . import model as M
from . import study as S

CONSISTENT_HIT = 2 / 3       # "consistently beat SPY": at least 2 of 3 known episodes
PLACEBO_DRAWS = 500


def _scored(results, signals, cfg):
    return [S.Scored(x, s, 0.0, 0.0, x.outcome.entry.year, False)
            for x, s in zip(results, signals) if S.shown(x, s, cfg)]


def group_of(s, h: int) -> str:
    return M.bucket(f"skill {h}", s)


def consistent(s, h: int) -> bool:
    raw = s.skill_raw20 if h == 20 else s.skill_raw60
    hit = s.skill_hit20 if h == 20 else s.skill_hit60
    return (s.skill_n(h) >= SK.MIN_EPISODES and raw is not None and raw > 0
            and hit is not None and hit >= CONSISTENT_HIT - 1e-9)


def persistence(results, signals, cfg: S.Config) -> tuple[str, dict]:
    """Every shown buy, by the insider's record at cfg.h (no model)."""
    h, bench = cfg.h, cfg.bench
    rows_ = _scored(results, signals, replace(cfg, skill_filter=None))
    labels = M.COMPONENTS[f"skill {h}"][1]
    groups: dict[str, list] = defaultdict(list)
    for x in rows_:
        groups[group_of(x.signals, h)].append(x)
        if x.signals.skill(h) is not None:
            groups["any record"].append(x)
            groups["consistent winners" if consistent(x.signals, h)
                   else "record, not consistent"].append(x)
    groups["all shown"] = rows_
    years = sorted({x.year for x in rows_})
    order = ["all shown"] + labels + ["any record", "consistent winners",
                                      "record, not consistent"]
    out, table_rows = {}, []
    for g in order:
        xs = groups.get(g, [])
        s = S._summ(xs, h, bench)
        by_year = {y: S._summ([x for x in xs if x.year == y], h, bench) for y in years}
        out[g] = {"n": s.n, "net_mean": s.mean, "t_cluster": s.t_cluster,
                  "hit": S.hit_rate(xs, h, bench),
                  "years": {str(y): v.mean for y, v in by_year.items()}}
        table_rows.append([g, s.n, pct(s.mean), num(s.t_cluster),
                           pct(S.hit_rate(xs, h, bench), 1)]
                          + [pct(v.mean) if v.n else "" for v in by_year.values()])
    text = table(["insider's record", "n", f"net mean, {h} days", "t (clustered)",
                  "hit rate"] + [str(y) for y in years], table_rows)
    return text, out


def spread_by_year(results, signals, cfg: S.Config) -> tuple[str, dict]:
    """Per year: best record bucket minus all shown buys, consistent winners minus the
    rest with a record, and the rank IC of the record among buys that have one."""
    h, bench = cfg.h, cfg.bench
    rows_ = _scored(results, signals, replace(cfg, skill_filter=None))
    top_label = M.COMPONENTS[f"skill {h}"][1][-2]
    years = sorted({x.year for x in rows_})
    out, table_rows = {}, []
    tallies = {"top": 0, "cons": 0, "ic": 0}
    counted = {"top": 0, "cons": 0, "ic": 0}
    for y in years:
        xs = [x for x in rows_ if x.year == y]
        allm = S._summ(xs, h, bench).mean
        top = S._summ([x for x in xs if group_of(x.signals, h) == top_label], h, bench)
        rec = [x for x in xs if x.signals.skill(h) is not None]
        cons = S._summ([x for x in rec if consistent(x.signals, h)], h, bench)
        rest = S._summ([x for x in rec if not consistent(x.signals, h)], h, bench)
        pairs = [(x.signals.skill(h), v) for x in rec
                 if (v := x.r(h, bench, net=True)) is not None]
        ic = stats.spearman([a for a, _ in pairs], [b for _, b in pairs]) if len(pairs) > 10 else None
        d_top = None if top.mean is None or allm is None else top.mean - allm
        d_cons = None if cons.mean is None or rest.mean is None else cons.mean - rest.mean
        for k, v in (("top", d_top), ("cons", d_cons), ("ic", ic)):
            if v is not None:
                counted[k] += 1
                tallies[k] += v > 0
        out[str(y)] = {"top_minus_all": d_top, "consistent_minus_rest": d_cons, "ic": ic,
                       "n_record": len(rec), "n_top": top.n, "n_consistent": cons.n}
        table_rows.append([y, len(xs), len(rec), top.n, pct(d_top), cons.n, pct(d_cons),
                           num(ic, 3)])
    table_rows.append(["**years positive**", "", "", "",
                       f"{tallies['top']} of {counted['top']}", "",
                       f"{tallies['cons']} of {counted['cons']}",
                       f"{tallies['ic']} of {counted['ic']}"])
    out["years_positive"] = {k: [tallies[k], counted[k]] for k in tallies}
    text = table(["year", "shown", "with a record", f"record {top_label}",
                  "minus all shown", "consistent winners", "minus rest with a record",
                  "rank IC (with a record)"], table_rows)
    return text, out


def gate_placebo(results, signals, prices, cfg: S.Config, draws: int = PLACEBO_DRAWS,
                 seed: int = 7) -> tuple[str, dict]:
    """The gate's flagged buys against random gates of the same size each year, all
    drawn from what the ungated model flags."""
    base = replace(cfg, skill_filter=None)
    folds = S.walk_forward(results, signals, cfg=base)
    by_year = {f.year: [x for x in f.scored if x.above] for f in folds}
    gated = {y: [x for x in xs if S.skill_ok(x.signals, cfg)] for y, xs in by_year.items()}
    h, bench = cfg.h, cfg.bench

    def pooled_mean(sets) -> float | None:
        return S._summ([x for xs in sets.values() for x in xs], h, bench).mean

    def pf_ir(sets) -> float | None:
        return S.simulate([x for xs in sets.values() for x in xs], prices, h,
                          hedged=bench == "beta", vol_target=cfg.vol_target).ir()

    actual, actual_ir = pooled_mean(gated), pf_ir(gated)
    base_mean = pooled_mean(by_year)
    rng = random.Random(seed)
    means, irs = [], []
    for i in range(draws):
        pick = {y: rng.sample(xs, len(gated[y])) for y, xs in by_year.items()}
        means.append(pooled_mean(pick))
        if i < min(draws, 100):          # the portfolio is slower: 100 draws
            irs.append(pf_ir(pick))
    means = [m for m in means if m is not None]
    irs = [v for v in irs if v is not None]
    p_mean = (sum(m >= actual for m in means) + 1) / (len(means) + 1) if actual is not None else None
    p_ir = (sum(v >= actual_ir for v in irs) + 1) / (len(irs) + 1) if actual_ir is not None else None
    n_gated = sum(len(v) for v in gated.values())
    n_base = sum(len(v) for v in by_year.values())
    out = {"n_base": n_base, "n_gated": n_gated, "base_mean": base_mean, "gated_mean": actual,
           "gated_ir": actual_ir, "placebo_mean_median": _median(means),
           "placebo_ir_median": _median(irs), "p_mean": p_mean, "p_ir": p_ir,
           "draws": len(means), "draws_ir": len(irs)}
    text = table(["", "flagged buys", f"net mean, {h} days", "portfolio IR (20 slots)"], [
        ["model flags, no gate", n_base, pct(base_mean), num(pf_ir(by_year))],
        [f"{cfg.skill_filter} gate", n_gated, pct(actual), num(actual_ir)],
        [f"random gates, same size each year (median of {len(means)} / {len(irs)})",
         n_gated, pct(_median(means)), num(_median(irs))],
        ["share of random gates at least as good", "",
         num(p_mean, 3), num(p_ir, 3)],
    ])
    return text, out


def _median(v):
    v = sorted(v)
    if not v:
        return None
    n = len(v)
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2


def coverage(signals, results, books: dict, cfg: S.Config) -> tuple[str, dict]:
    rows_ = _scored(results, signals, cfg)
    years = sorted({x.year for x in rows_})
    out, table_rows = {}, []
    for y in years:
        xs = [x.signals for x in rows_ if x.year == y]
        r20 = sum(s.skill_20 is not None for s in xs)
        r60 = sum(s.skill_60 is not None for s in xs)
        any60 = sum(s.skill_n60 > 0 for s in xs)
        vals = sorted(s.skill_60 for s in xs if s.skill_60 is not None)
        q = (lambda p: vals[int(p * (len(vals) - 1))]) if vals else (lambda p: None)
        out[str(y)] = {"shown": len(xs), "record_20": r20, "record_60": r60,
                       "any_60": any60}
        table_rows.append([y, len(xs), any60, f"{r20} ({pct(r20 / len(xs) if xs else None, 0)})",
                           f"{r60} ({pct(r60 / len(xs) if xs else None, 0)})",
                           pct(q(0.1)), pct(q(0.5)), pct(q(0.9))])
    text = table(["year", "shown", "any earlier 60-day episode", "20-day record",
                  "60-day record", "60-day record, 10th pct", "median", "90th pct"],
                 table_rows)
    out["k"] = {str(h): b.k for h, b in books.items()}
    out["episodes"] = {str(h): b.n_episodes for h, b in books.items()}
    return text, out


def render(results, signals, prices, books: dict, variants_text: str,
           placebos: dict[str, str], persist: dict[int, str], spreads: dict[int, str],
           cov_text: str, main_year_table: str) -> str:
    ks = ", ".join(f"{h} days: k = {num(b.k, 1)} ({b.n_episodes} episodes)"
                   for h, b in sorted(books.items()))
    lines = [
        "# Insider skill: do insiders who beat SPY keep beating it?",
        "",
        "Every number is out of sample and point in time: an insider's record counts only "
        "earlier buy episodes whose exit was before the new filing date, at any company, "
        f"with at least {SK.MIN_EPISODES} episodes, shrunk toward the average insider "
        "(see `tradetracker.skill`). Returns are net of costs over SPY. Shown buys are "
        "liquid open-market buys, no 10b5-1, drift <= 20%, market cap >= $50M.",
        "",
        f"Shrinkage, estimated on episodes known before 2017: {ks}. "
        "k is the number of episodes at which a record counts half: a large k means "
        "insiders' records differ little beyond noise.",
        "",
        "## Coverage", "", cov_text, "",
    ]
    for h in sorted(persist):
        lines += [f"## Persistence at {h} days (no model)", "",
                  "Every shown buy, grouped by the insider's record at the same hold. "
                  "\"Consistent winners\" beat SPY on average and in at least 2 of 3 of "
                  "their known episodes (raw, unshrunk).", "", persist[h], "",
                  "Year by year:", "", spreads[h], ""]
    lines += ["## In the paper-traded setup (walk-forward)", "",
              "The same walk-forward and go-live criteria as PRs #9 and #11. Gates are "
              "applied to what the ungated model flags; \"skill in Q\" fits the record as "
              "a third input; \"skill alone\" ranks by the record and the run-up "
              "penalty. The last two columns are the 40-slot book the paper account runs.",
              "", variants_text, ""]
    for name, text in placebos.items():
        lines += [f"### Random-gate check: {name}", "", text, ""]
    lines += ["## Main variant by test year", "", main_year_table, ""]
    return "\n".join(lines)
