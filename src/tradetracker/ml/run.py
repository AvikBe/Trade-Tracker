"""`tt ml-study`: run every setup, the baseline, placebos and explanations; write a report."""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

from ..backtest.study import num, pct, table
from ..scoring import study as SS
from . import features as F
from . import study as ML

TOP_DEPENDENCE = 10
TOP_PAIRS = 6


def _log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def run(results, signals, events, prices, out: Path, placebos: int = 20,
        setups=None, detailed=None) -> dict:
    """Every setup with its baseline and placebos (`placebos` runs for the detailed
    setups, half as many for the rest); ablations and explanations for `detailed`."""
    out.mkdir(parents=True, exist_ok=True)
    setups = setups or ML.SETUPS
    detailed = detailed or [c for c in (ML.PRIMARY, ML.PRIMARY_RANK) if c in setups]
    SS.add_beta_legs(results, signals)
    activity = F.Activity(events)
    ML.write_events(events, out / "events-all.csv.gz")
    rows_by_cap: dict = {}

    def rows_for(cfg: ML.MLConfig):
        if cfg.min_cap not in rows_by_cap:
            _log(f"features for cap floor {cfg.min_cap}")
            rows_by_cap[cfg.min_cap] = ML.pool_rows(results, signals, prices, activity, cfg)
            ML.write_rows(rows_by_cap[cfg.min_cap],
                          out / f"features-{'cap-floor' if cfg.min_cap else 'all'}.csv.gz")
        return rows_by_cap[cfg.min_cap]

    summary: dict = {"setups": {}, "baselines": {}}
    parts: list[str] = []
    headline_rows, base_rows, placebo_rows = [], [], []
    placebo_texts: dict = {}
    kept: dict = {}
    for cfg in setups:
        rows = rows_for(cfg)
        cats = ML.categories(rows)
        _log(f"walk-forward: {cfg.name} on {len(rows)} buys")
        folds = ML.walk_forward(rows, cfg, cats, explain=cfg in detailed)
        o = ML.evaluate(cfg.name, folds, prices, cfg.scoring())
        h = ML.headline(o, cfg.scoring())
        h["rounds"] = {f.year: f.rounds for f in folds}
        summary.setdefault("folds", {})[cfg.name] = {f.year: f.n_train for f in folds}
        summary["setups"][cfg.name] = h
        headline_rows.append(ML.headline_row(h, cfg.h))
        kept[cfg.name] = (cfg, folds, o)
        SS.write_scored(folds, out / f"scored-{_slug(cfg.name)}.csv.gz", cfg.scoring())
        n_placebo = placebos if cfg in detailed else placebos // 2
        if n_placebo:
            _log(f"placebos: {n_placebo}")
            shuffles = ML.placebo(rows, cfg, prices, n_placebo, cats)
            ptext, psumm = ML.placebo_text(h, shuffles)
            summary.setdefault("placebo", {})[cfg.name] = {"summary": psumm, "runs": shuffles}
            placebo_rows.append([cfg.name, n_placebo] + [
                f"{psumm[k]['beaten']} of {psumm[k]['n']}" if k in psumm else ""
                for k in ("top_mean", "rank_mean", "mean_ic", "ir")])
            placebo_texts[cfg.name] = ptext
        bkey = cfg.baseline()
        if bkey.name + str(cfg.min_cap) not in summary["baselines"]:
            _log(f"baseline: {bkey.name}, cap {cfg.min_cap}")
            bf = SS.walk_forward(results, signals, cfg=bkey)
            bo = ML.evaluate(bkey.name, bf, prices, bkey)
            bh = ML.headline(bo, bkey)
            bh["name"] = f"two-input score, {cfg.h} days" + (", cap floor" if cfg.min_cap else "")
            summary["baselines"][bkey.name + str(cfg.min_cap)] = bh
            base_rows.append(ML.headline_row(bh, cfg.h))
            kept["base " + bh["name"]] = (cfg, bf, bo)

    for primary in detailed:
        parts.extend(_detail(primary, kept, rows_for, summary, prices, placebo_texts))

    head_placebo = ("## Placebos: walk-forwards with training labels shuffled within each year\n\n"
                    "How many shuffled runs the real model beats, per metric.\n\n"
                    + table(["setup", "runs", "above-cutoff mean", "top 10% by rank",
                             "mean rank IC", "portfolio IR"], placebo_rows) + "\n")

    # --- portfolio slots
    _log("slots")
    books = [(c.name, kept[c.name][1], c.scoring()) for c in detailed]
    for c in detailed:
        key = "base two-input score, " + f"{c.h} days" + (", cap floor" if c.min_cap else "")
        if key in kept and key[5:] not in {b[0] for b in books}:
            books.append((key[5:], kept[key][1], kept[key][0].baseline()))
    parts.append("## How many slots? (the same signals, a bigger or smaller book)\n\n"
                 + ML.slots_table(books, prices) + "\n")

    head = ["# Gradient-boosted models over the per-trade features\n",
            "## All setups against the two-input score\n",
            "Every row is out of sample, 2017-2026, on the shown buys (one per stock and entry "
            "day). Per 20 days scales a 60-day mean to compare with the 1% bar.\n",
            table(ML.HEADLINE_HEADERS, headline_rows + base_rows) + "\n"]
    if placebo_rows:
        head.append(head_placebo)
    text = "\n".join(head + parts)
    (out / "report.md").write_text(text)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    _log(f"wrote {out / 'report.md'}")
    return summary


def _detail(primary, kept, rows_for, summary, prices, placebo_texts) -> list[str]:
    parts: list[str] = []
    pcfg, pfolds, pout = kept[primary.name]
    prows = rows_for(primary)
    pcats = ML.categories(prows)

    # --- the primary setup in detail
    parts.append(f"## In detail: {primary.name}\n")
    parts.append("The closest match to what is being paper traded (60-day hold, $50M cap "
                 "floor).\n")
    parts.append("### Go-live criteria\n\n" + pout.criteria_text + "\n")
    parts.append("### By test year\n\n" + pout.year_text + "\n")
    rows = [[f.year, f.n_fit, f.n_valid, f.n_train, " ".join(map(str, f.rounds)), pct(f.cutoff, 3),
             sum(x.above for x in f.scored), len(f.scored)] for f in pfolds]
    parts.append("### Folds\n\n" + table(["test year", "fit", "validation", "refit on",
                                          "rounds per seed", "cutoff", "above", "test buys"], rows)
                 + "\n")
    parts.append("### By year, portfolio\n\n" + SS.portfolio_by_year(pout.pf) + "\n")

    if primary.name in placebo_texts:
        parts.append("### Placebo (training labels shuffled within each year)\n\n"
                     + placebo_texts[primary.name] + "\n")

    # --- ablations on the primary setup
    _log("ablations")
    ab_rows = []
    groups = sorted({F.GROUPS[f] for f in F.FEATURES})
    variants = [("all features", primary)]
    variants.append(("drawdown and run-up only (the two inputs, in a GBM)",
                     replace(primary, name="GBM two inputs", features=tuple(F.TWO_INPUT_FEATURES))))
    for g in groups:
        feats = tuple(f for f in F.FEATURES if F.GROUPS[f] != g)
        variants.append((f"without {g}", replace(primary, name=f"without {g}", features=feats)))
    for g in groups:
        feats = tuple(f for f in F.FEATURES if F.GROUPS[f] == g)
        variants.append((f"{g} only", replace(primary, name=f"{g} only", features=feats)))
    variants.append(("beta-hedged target and return", replace(primary, name="beta hedge", bench="beta")))
    variants.append(("no embargo (purge only)", replace(primary, name="no embargo", embargo_days=0)))
    abl = {}
    for label, cfg in variants:
        if cfg is primary:
            h = summary["setups"][primary.name]
        else:
            f2 = ML.walk_forward(prows, cfg, pcats, explain=False)
            h = ML.headline(ML.evaluate(label, f2, prices, cfg.scoring()), cfg.scoring())
        abl[label] = h
        ab_rows.append([label, h["above"], pct(h["top_mean"]), num(h["top_t"]),
                        pct(h["rank_mean"]), num(h["mean_ic"], 3), num(h["ir"]),
                        f"{h['criteria']} of 4"])
    summary.setdefault("ablation", {})[primary.name] = abl
    parts.append("### Ablations (same walk-forward, one change each)\n\n" + table(
        ["model", "above cutoff", "net mean", "t", "top 10% by rank", "mean rank IC",
         "portfolio IR", "criteria"], ab_rows) + "\n")

    # --- explanations
    _log("SHAP")
    X, S, scored = ML.pooled_shap(pfolds)
    imp = ML.importance(X, S, [x.year for x in scored])
    summary.setdefault("importance", {})[primary.name] = imp
    parts.append("## What the model used (SHAP, out of sample)\n")
    parts.append("Mean absolute SHAP value is how far, on average, a feature moves a buy's "
                 "predicted net return. Direction is the rank correlation between the "
                 "feature's value and its SHAP value (+1: higher is better).\n")
    parts.append(ML.importance_text(imp) + "\n")
    parts.append("### By feature group\n\n" + ML.group_importance(imp) + "\n")
    parts.append("### Top features by quintile (test buys, deduplicated)\n")
    parts.append("Mean SHAP is what the model thought; realized net mean is what happened. "
                 "Where the two agree across quintiles, the pattern is real out of sample.\n")
    stab_rows = []
    for d in imp[:TOP_DEPENDENCE]:
        f = d["feature"]
        parts.append(f"**{f}**\n\n" + ML.dependence(X, S, scored, f, primary.h, primary.bench) + "\n")
        if f not in F.CATEGORICAL and d["direction"] is not None:
            w, n, spreads = ML.year_stability(X, scored, f, primary.h, primary.bench,
                                              d["direction"] > 0)
            stab_rows.append([f, "higher" if d["direction"] > 0 else "lower", f"{w} of {n}",
                              " ".join(f"{y}: {100 * s:+.1f}" for y, s in spreads)])
    parts.append("### Is each pattern stable? Better quintile minus worse quintile, by year "
                 "(net, points)\n\n" + table(["feature", "better when", "years it held",
                                              "spread by year"], stab_rows) + "\n")
    _log("interactions")
    pairs = ML.interactions(pfolds)
    summary.setdefault("interactions", {})[primary.name] = pairs[:20]
    parts.append("### Strongest interactions (mean abs SHAP interaction)\n\n" + table(
        ["feature", "with", "strength"], [[a, b, pct(v, 3)] for a, b, v in pairs[:12]]) + "\n")
    for a, b, _ in pairs[:TOP_PAIRS]:
        if a in F.CATEGORICAL or b in F.CATEGORICAL:
            continue
        parts.append(f"**{a} x {b}**, realized net mean by tercile\n\n"
                     + ML.interaction_grid(scored, X, a, b, primary.h, primary.bench) + "\n")

    return parts


def _slug(s: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in s.lower()).strip("-").replace("--", "-")
