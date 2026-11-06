"""Gradient-boosted models (LightGBM) over every per-trade feature, walk-forward by year.

The question: given all the features we have, can a flexible model find buys that clear
costs where the hand-built scores (PRs #6 to #9) could not, and which features and
interactions does it rely on?

Protocol, chosen before looking at any test year:

- **Sample and metrics are phase 4's.** The shown buys (`scoring.study.shown`), one per
  stock and entry day, the same net returns, the same four go-live criteria, and the
  same top-20 portfolio, so every row is comparable with the two-input baseline, which
  is rerun here on the same pool.
- **Walk-forward with purge and embargo.** To score year y, a model may only train on
  buys whose exit is at least `embargo_days` before 1 January of y, so no training
  return overlaps the test year or its first days. Early stopping uses the last
  training year as validation, and that split is purged and embargoed the same way.
  The number of rounds found there is then refitted on the whole training set.
- **Fixed, conservative hyperparameters**: shallow trees (15 leaves, depth 4), at least
  300 buys per leaf, row and feature subsampling, L2 shrinkage, learning rate 0.02. They
  are not tuned on test years. Predictions average `seeds` models.
- **Targets**: "net" is the net excess return, winsorized at the training set's 1st and
  99th percentiles; "rank" is its percentile within the entry month minus one half, so
  the model learns which buys beat other buys that month, not when the market rose.
- **Cutoff**: a test buy is "above" when its prediction reaches the 90th percentile of
  the model's predictions on its own training set, as the baseline's cutoff is the
  training top decile.
- **Placebos**: the same procedure with the training labels shuffled within each year.
  A real pattern must beat nearly all of them.
- **Explanations**: SHAP values (LightGBM's `pred_contrib`) for every test buy, pooled
  over folds, so what is reported is what the models used out of sample.
"""

from __future__ import annotations

import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass, field, replace
from datetime import date, timedelta

import numpy as np
import pandas as pd

from ..backtest import stats
from ..backtest.study import num, pct, table
from ..scoring import study as SS
from . import features as F

FIRST_TEST_YEAR = 2017
MIN_TRAIN = 500
MIN_VALID = 200
TOP_QUANTILE = 0.90
WINSOR = (0.01, 0.99)
MAX_ROUNDS = 800
EARLY_STOP = 60

PARAMS = {
    "objective": "regression",
    "learning_rate": 0.02,
    "num_leaves": 15,
    "max_depth": 4,
    "min_data_in_leaf": 300,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq": 1,
    "lambda_l2": 10.0,
    "cat_smooth": 50.0,
    "min_data_per_group": 200,
    "verbose": -1,
    # More threads than idle cores makes LightGBM's OpenMP spin: 100x slower, not faster.
    "num_threads": int(os.environ.get("TT_ML_THREADS", "2")),
    "deterministic": True,
    "force_col_wise": True,
}


@dataclass(frozen=True)
class MLConfig:
    name: str
    h: int = 20
    bench: str = "spy"
    min_cap: float | None = None
    target: str = "net"                      # "net" or "rank"
    features: tuple[str, ...] = tuple(F.FEATURES)
    seeds: int = 3
    embargo_days: int = 14

    def scoring(self) -> SS.Config:
        """The matching scoring config: sample, horizon and benchmark."""
        return SS.Config(self.name, h=self.h, bench=self.bench, min_cap=self.min_cap)

    def baseline(self) -> SS.Config:
        return replace(self.scoring(), name=f"two inputs, {self.h} days", components=SS.TWO_INPUTS)


SETUPS = [
    MLConfig(f"GBM, {h} days" + (", cap floor" if cap else "") + f", {t} target",
             h=h, min_cap=50e6 if cap else None, target=t)
    for cap in (False, True) for h in (20, 60) for t in ("net", "rank")
]
PRIMARY = next(c for c in SETUPS if c.h == 60 and c.min_cap and c.target == "net")
# The same with the within-month rank target: it can't profit from timing the market.
PRIMARY_RANK = next(c for c in SETUPS if c.h == 60 and c.min_cap and c.target == "rank")


class _Stub:
    """What `scoring.study.year_table` and `summary` read from a fold's model."""
    beta = None

    def __init__(self, threshold: float):
        self.threshold = threshold


@dataclass
class MLFold:
    year: int
    n_train: int
    n_fit: int
    n_valid: int
    rounds: list[int]
    cutoff: float
    scored: list[SS.Scored]
    model: _Stub = None
    # SHAP for the deduplicated test buys: same order as `scored`, last column the bias
    shap: np.ndarray | None = None
    X: pd.DataFrame | None = None
    boosters: list = field(default_factory=list)


# ---------------------------------------------------------------- data

def pool_rows(results, signals, prices, activity: F.Activity, cfg: MLConfig) -> list[F.Row]:
    pool = [(x, s) for x, s in zip(results, signals) if SS.shown(x, s, cfg.scoring())]
    return F.build_rows(pool, prices, activity)


def categories(rows: list[F.Row]) -> dict[str, list[str]]:
    """Fixed category lists (labels only, no returns), so every fold codes them alike."""
    return {c: sorted({r.features[c] for r in rows}) for c in F.CATEGORICAL}


def frame(rows: list[F.Row], feats, cats: dict[str, list[str]]) -> pd.DataFrame:
    df = pd.DataFrame([[r.features[f] for f in feats] for r in rows], columns=list(feats))
    for c in feats:
        if c in cats:
            df[c] = pd.Categorical(df[c], categories=cats[c])
        else:
            df[c] = df[c].astype(float)
    return df


def target(rows: list[F.Row], cfg: MLConfig) -> list[float | None]:
    return [r.result.r(cfg.h, cfg.bench, net=True) for r in rows]


def exit_day(r: F.Row, h: int) -> date | None:
    leg = r.result.outcome.legs.get(h)
    return leg["exit"] if leg else None


def train_labels(rows: list[F.Row], ys: list[float], cfg: MLConfig) -> np.ndarray:
    y = np.asarray(ys, dtype=float)
    if cfg.target == "rank":
        out = np.empty_like(y)
        by_month = defaultdict(list)
        for i, r in enumerate(rows):
            e = r.result.outcome.entry
            by_month[(e.year, e.month)].append(i)
        for idx in by_month.values():
            v = pd.Series(y[idx]).rank(pct=True).to_numpy()
            out[idx] = v - 0.5 if len(idx) > 1 else 0.0
        return out
    lo, hi = np.quantile(y, WINSOR)
    return np.clip(y, lo, hi)


def shuffle_within_year(rows: list[F.Row], y: np.ndarray, seed: int) -> np.ndarray:
    rng = random.Random(seed)
    out = y.copy()
    by_year = defaultdict(list)
    for i, r in enumerate(rows):
        by_year[r.year].append(i)
    for idx in by_year.values():
        perm = idx[:]
        rng.shuffle(perm)
        out[idx] = y[perm]
    return out


# ---------------------------------------------------------------- training

def _spearman(a, b) -> float:
    ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def _spearman_eval(preds, data):
    return "ic", _spearman(preds, data.get_label()), True


def _fit(X, y, Xv=None, yv=None, rounds: int | None = None, seed: int = 0):
    import lightgbm as lgb
    params = dict(PARAMS, seed=seed, bagging_seed=seed, feature_fraction_seed=seed,
                  data_random_seed=seed)
    dtrain = lgb.Dataset(X, y, free_raw_data=False)
    if rounds is not None:
        return lgb.train(params, dtrain, num_boost_round=max(rounds, 1))
    dvalid = lgb.Dataset(Xv, yv, reference=dtrain)
    b = lgb.train(dict(params, metric="None"), dtrain, num_boost_round=MAX_ROUNDS,
                  valid_sets=[dvalid], feval=_spearman_eval,
                  callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
    return b


def split(rows: list[F.Row], ys, year: int, h: int, embargo_days: int):
    """Indices of (train, fit, valid) rows for test year `year`, purged and embargoed."""
    start = date(year, 1, 1)
    vstart = date(year - 1, 1, 1)
    gap = timedelta(days=embargo_days)
    train, fit, valid = [], [], []
    for i, (r, y) in enumerate(zip(rows, ys)):
        ex = exit_day(r, h)
        if y is None or ex is None or ex >= start - gap:
            continue
        train.append(i)
        if r.result.outcome.entry >= vstart:
            valid.append(i)
        elif ex < vstart - gap:
            fit.append(i)
    return train, fit, valid


def walk_forward(rows: list[F.Row], cfg: MLConfig, cats=None, shuffle_seed: int | None = None,
                 explain: bool = True, first_test_year: int = FIRST_TEST_YEAR) -> list[MLFold]:
    cats = cats or categories(rows)
    feats = list(cfg.features)
    ys = target(rows, cfg)
    X_all = frame(rows, feats, cats)
    years = sorted({r.year for r in rows})
    folds = []
    for y in years:
        if y < first_test_year:
            continue
        tr, fi, va = split(rows, ys, y, cfg.h, cfg.embargo_days)
        if len(fi) < MIN_TRAIN or len(va) < MIN_VALID:
            continue
        lab = np.full(len(rows), np.nan)
        lab[tr] = train_labels([rows[i] for i in tr], [ys[i] for i in tr], cfg)
        if shuffle_seed is not None:
            lab[tr] = shuffle_within_year([rows[i] for i in tr], lab[tr], shuffle_seed * 1000 + y)
        test = [i for i, r in enumerate(rows) if r.year == y]
        Xtr, Xte = X_all.iloc[tr], X_all.iloc[test]
        rounds, boosters = [], []
        for seed in range(cfg.seeds):
            es = _fit(X_all.iloc[fi], lab[fi], X_all.iloc[va], lab[va], seed=seed)
            n = max(es.best_iteration or 0, 20)
            rounds.append(n)
            boosters.append(_fit(Xtr, lab[tr], rounds=n, seed=seed))
        pred_tr = np.mean([b.predict(Xtr) for b in boosters], axis=0)
        cutoff = float(np.quantile(pred_tr, TOP_QUANTILE))
        pred = np.mean([b.predict(Xte) for b in boosters], axis=0) if test else np.array([])
        scored = [SS.Scored(rows[i].result, rows[i].signals, float(p), float(p), y, p >= cutoff)
                  for i, p in zip(test, pred)]
        kept = SS.dedupe(scored)
        fold = MLFold(y, len(tr), len(fi), len(va), rounds, cutoff, kept, _Stub(cutoff),
                      boosters=boosters)
        if explain and kept:
            pos = {id(s): j for j, s in enumerate(scored)}
            idx = [test[pos[id(s)]] for s in kept]
            Xk = X_all.iloc[idx]
            fold.X = Xk.reset_index(drop=True)
            fold.shap = np.mean([b.predict(Xk, pred_contrib=True) for b in boosters], axis=0)
        folds.append(fold)
    return folds


# ---------------------------------------------------------------- evaluation

@dataclass
class Outcome:
    name: str
    folds: list
    pooled: dict
    pf: SS.Portfolio
    checks: list[bool]
    criteria_text: str
    year_text: str


def evaluate(name: str, folds, prices, scfg: SS.Config, slots: int = SS.PORTFOLIO_SLOTS) -> Outcome:
    ytext, pooled = SS.year_table(folds, scfg)
    above = [x for f in folds for x in f.scored if x.above]
    pf = SS.simulate(above, prices, scfg.h, slots=slots, hedged=scfg.bench == "beta")
    ctext, checks = SS.criteria(pooled, pf, scfg)
    return Outcome(name, folds, pooled, pf, checks, ctext, ytext)


def headline(o: Outcome, scfg: SS.Config) -> dict:
    s = SS._summ(o.pooled["top"], scfg.h, scfg.bench)
    r = SS._summ(o.pooled["rank"], scfg.h, scfg.bench)
    a = SS._summ(o.pooled["all"], scfg.h, scfg.bench)
    yearly = [v for v in o.pooled["yearly"] if v is not None]
    ics = []
    for f in o.folds:
        pairs = [(x.score, v) for x in f.scored if (v := x.r(scfg.h, scfg.bench, net=True)) is not None]
        ic = stats.spearman([p for p, _ in pairs], [v for _, v in pairs])
        if ic is not None:
            ics.append(ic)
    return {"name": o.name, "all_mean": a.mean, "n_all": a.n, "above": s.n, "top_mean": s.mean,
            "top_t": s.t_cluster, "rank_mean": r.mean, "years_pos": sum(v > 0 for v in yearly),
            "years": len(yearly), "hit": SS.hit_rate(o.pooled["top"], scfg.h, scfg.bench),
            "ir": o.pf.ir(), "mdd_ratio": (o.pf.mdd() / o.pf.spy_mdd() if o.pf.spy_mdd() < 0 else None),
            "criteria": sum(o.checks), "mean_ic": sum(ics) / len(ics) if ics else None,
            "ic_pos": sum(v > 0 for v in ics)}


def headline_row(h: dict, horizon: int) -> list:
    per20 = None if h["top_mean"] is None else h["top_mean"] * 20 / horizon
    return [h["name"], h["n_all"], pct(h["all_mean"]), h["above"], pct(h["top_mean"]),
            pct(per20), num(h["top_t"]), f"{h['years_pos']} of {h['years']}",
            pct(h["hit"], 1), num(h["ir"]), num(h["mdd_ratio"]), num(h["mean_ic"], 3),
            f"{h['criteria']} of 4"]


HEADLINE_HEADERS = ["setup", "shown", "all, net", "above cutoff", "their net mean", "per 20 days",
                    "t (clustered)", "years positive", "hit rate", "portfolio IR",
                    "drawdown vs SPY", "mean rank IC", "criteria met"]


def placebo(rows, cfg: MLConfig, prices, n: int, cats=None, start_seed: int = 1) -> list[dict]:
    """The full walk-forward with shuffled training labels, n times."""
    out = []
    for k in range(start_seed, start_seed + n):
        folds = walk_forward(rows, cfg, cats, shuffle_seed=k, explain=False)
        out.append(headline(evaluate(f"placebo {k}", folds, prices, cfg.scoring()), cfg.scoring()))
    return out


def placebo_text(real: dict, shuffles: list[dict]) -> tuple[str, dict]:
    rows, summ = [], {}
    for key, label in [("top_mean", "above-cutoff net mean"), ("rank_mean", "top 10% by rank, net"),
                       ("mean_ic", "mean rank IC"), ("ir", "portfolio IR")]:
        vals = sorted(v for s in shuffles if (v := s[key]) is not None)
        rv = real[key]
        if not vals or rv is None:
            continue
        beat = sum(rv > v for v in vals)
        fmt = (lambda v: num(v, 3)) if key in ("mean_ic", "ir") else pct
        rows.append([label, fmt(rv), fmt(vals[len(vals) // 2]), fmt(vals[-1]),
                     f"{beat} of {len(vals)}"])
        summ[key] = {"real": rv, "median": vals[len(vals) // 2], "best": vals[-1],
                     "beaten": beat, "n": len(vals)}
    return table(["metric", "real", "placebo median", "best placebo", "real beats"], rows), summ


# ---------------------------------------------------------------- explanations

def pooled_shap(folds: list[MLFold]) -> tuple[pd.DataFrame, np.ndarray, list]:
    fs = [f for f in folds if f.shap is not None and len(f.scored)]
    X = pd.concat([f.X for f in fs], ignore_index=True)
    S = np.vstack([f.shap for f in fs])
    scored = [x for f in fs for x in f.scored]
    return X, S, scored


def importance(X: pd.DataFrame, S: np.ndarray, years: list[int]) -> list[dict]:
    """Mean |SHAP| per feature (in return points), its direction, and how stable its
    rank is across test years."""
    feats = list(X.columns)
    mag = np.abs(S[:, :-1])
    total = mag.mean(axis=0)
    yr = np.asarray(years)
    ranks = defaultdict(list)
    for y in sorted(set(years)):
        m = mag[yr == y].mean(axis=0)
        order = np.argsort(-m)
        for r, j in enumerate(order):
            ranks[feats[j]].append(r + 1)
    out = []
    for j, f in enumerate(feats):
        col = X[f]
        if f in F.CATEGORICAL:
            direction = None
        else:
            v = col.to_numpy(dtype=float)
            ok = ~np.isnan(v)
            direction = (stats.spearman(list(v[ok]), list(S[ok, j])) if ok.sum() > 10 else None)
        out.append({"feature": f, "group": F.GROUPS.get(f, ""), "mean_abs": float(total[j]),
                    "direction": direction,
                    "median_year_rank": float(np.median(ranks[f])) if ranks[f] else None,
                    "top10_years": sum(r <= 10 for r in ranks[f]), "n_years": len(ranks[f])})
    out.sort(key=lambda d: -d["mean_abs"])
    return out


def importance_text(imp: list[dict], top: int = 20) -> str:
    rows = [[i + 1, d["feature"], d["group"], pct(d["mean_abs"], 3), num(d["direction"]),
             num(d["median_year_rank"], 0), f"{d['top10_years']} of {d['n_years']}"]
            for i, d in enumerate(imp[:top])]
    return table(["rank", "feature", "group", "mean abs SHAP (return)",
                  "direction (Spearman of value vs SHAP)", "median rank by year",
                  "in top 10 by year"], rows)


def group_importance(imp: list[dict]) -> str:
    g = defaultdict(float)
    for d in imp:
        g[d["group"]] += d["mean_abs"]
    tot = sum(g.values()) or 1
    rows = [[k, pct(v, 3), pct(v / tot, 0)] for k, v in sorted(g.items(), key=lambda kv: -kv[1])]
    return table(["group", "sum of mean abs SHAP", "share"], rows)


def dependence(X: pd.DataFrame, S: np.ndarray, scored, feature: str, h: int, bench: str,
               bins: int = 5) -> str:
    """By quintile of one feature over the test buys: its mean SHAP, and the realized
    net return and hit rate there (model-free, out of sample)."""
    j = list(X.columns).index(feature)
    ys = np.array([np.nan if (v := x.r(h, bench, net=True)) is None else v for x in scored])
    if feature in F.CATEGORICAL:
        groups = [(str(c), (X[feature] == c).to_numpy()) for c in X[feature].cat.categories]
    else:
        v = X[feature].to_numpy(dtype=float)
        ok = ~np.isnan(v)
        groups = []
        if ok.sum() >= bins * 20:
            edges = np.quantile(v[ok], np.linspace(0, 1, bins + 1))
            for b in range(bins):
                lo, hi = edges[b], edges[b + 1]
                m = ok & (v >= lo) & ((v <= hi) if b == bins - 1 else (v < hi))
                groups.append((f"{_fmt(lo)} to {_fmt(hi)}", m))
        if (~ok).sum():
            groups.append(("missing", ~ok))
    rows = []
    for label, m in groups:
        if m.sum() < 20:
            continue
        yy = ys[m & ~np.isnan(ys)]
        rows.append([label, int(m.sum()), pct(float(S[m, j].mean()), 3),
                     pct(float(yy.mean())) if len(yy) else "",
                     pct(float((yy > 0).mean()), 1) if len(yy) else ""])
    return table([feature, "test buys", "mean SHAP", "realized net mean", "hit rate"], rows)


def _fmt(v: float) -> str:
    a = abs(v)
    if a >= 1e9:
        return f"{v / 1e9:.1f}B"
    if a >= 1e6:
        return f"{v / 1e6:.1f}M"
    if a >= 1000:
        return f"{v:,.0f}"
    if a >= 10:
        return f"{v:.0f}"
    return f"{v:.3g}"


def interactions(folds: list[MLFold], per_fold: int = 400, seed: int = 0) -> list[tuple]:
    """Mean |SHAP interaction| for feature pairs, on a sample of each fold's test buys,
    from the fold's first model."""
    import shap
    rng = np.random.default_rng(seed)
    total, n = None, 0
    feats = None
    for f in folds:
        if f.X is None or not len(f.X) or not f.boosters:
            continue
        k = min(per_fold, len(f.X))
        idx = rng.choice(len(f.X), size=k, replace=False)
        Xs = f.X.iloc[idx].copy()
        for c in Xs.columns:            # shap reads category codes, as LightGBM does
            if isinstance(Xs[c].dtype, pd.CategoricalDtype):
                Xs[c] = Xs[c].cat.codes.astype(float).where(Xs[c].cat.codes >= 0)
        iv = shap.TreeExplainer(f.boosters[0]).shap_interaction_values(Xs)
        m = np.abs(iv).sum(axis=0)
        total = m if total is None else total + m
        n += k
        feats = list(f.X.columns)
    if total is None:
        return []
    total /= n
    pairs = []
    for a in range(len(feats)):
        for b in range(a + 1, len(feats)):
            pairs.append((feats[a], feats[b], float(total[a, b] + total[b, a])))
    pairs.sort(key=lambda p: -p[2])
    return pairs


def interaction_grid(rows_scored, X: pd.DataFrame, a: str, b: str, h: int, bench: str) -> str:
    """Realized net mean (and count) in a 3x3 grid of terciles of two features."""
    ys = np.array([np.nan if (v := x.r(h, bench, net=True)) is None else v for x in rows_scored])

    def terc(col):
        v = X[col].to_numpy(dtype=float)
        ok = ~np.isnan(v)
        q = np.quantile(v[ok], [1 / 3, 2 / 3])
        lab = np.full(len(v), -1)
        lab[ok] = np.digitize(v[ok], q)
        return lab, q

    la, qa = terc(a)
    lb, qb = terc(b)
    names = ["low", "mid", "high"]
    rows = []
    for i in range(3):
        row = [f"{a} {names[i]}"]
        for j in range(3):
            m = (la == i) & (lb == j) & ~np.isnan(ys)
            row.append(f"{pct(float(ys[m].mean()))} (n {int(m.sum())})" if m.sum() >= 20 else "")
        rows.append(row)
    head = [f"{a} \\ {b}"] + [f"{b} {names[j]}" for j in range(3)]
    edges = (f"Terciles: {a} at {_fmt(qa[0])} and {_fmt(qa[1])}; "
             f"{b} at {_fmt(qb[0])} and {_fmt(qb[1])}.")
    return table(head, rows) + "\n\n" + edges


# ---------------------------------------------------------------- stability

def year_stability(X: pd.DataFrame, scored, feature: str, h: int, bench: str,
                   high_is_good: bool) -> tuple[int, int, list]:
    """Within each test year: does the better feature quintile beat the worse one?"""
    v = X[feature].to_numpy(dtype=float)
    ys = np.array([np.nan if (r := x.r(h, bench, net=True)) is None else r for x in scored])
    years = np.array([x.year for x in scored])
    wins, n, spreads = 0, 0, []
    for y in sorted(set(years)):
        m = (years == y) & ~np.isnan(v) & ~np.isnan(ys)
        if m.sum() < 100:
            continue
        q = np.quantile(v[m], [0.2, 0.8])
        lo, hi = ys[m & (v <= q[0])], ys[m & (v >= q[1])]
        if not len(lo) or not len(hi):
            continue
        d = float(hi.mean() - lo.mean()) * (1 if high_is_good else -1)
        spreads.append((int(y), d))
        n += 1
        wins += d > 0
    return wins, n, spreads


def slots_table(outcomes: list[tuple[str, list, SS.Config]], prices,
                slots=(10, 20, 40, 80)) -> str:
    rows = []
    for name, folds, scfg in outcomes:
        above = [x for f in folds for x in f.scored if x.above]
        for k in slots:
            pf = SS.simulate(above, prices, scfg.h, slots=k, hedged=scfg.bench == "beta")
            rows.append([name, k, pf.trades, pct(pf.annual_excess()), num(pf.ir()),
                         num(pf.mdd() / pf.spy_mdd() if pf.spy_mdd() < 0 else None)])
    return table(["setup", "slots", "positions taken", "excess vs SPY (annualized)", "IR",
                  "drawdown vs SPY"], rows)


def mean_or_nan(v: list[float]) -> float:
    return sum(v) / len(v) if v else math.nan


# ---------------------------------------------------------------- files

ROW_META = ["filer_id", "issuer_cik", "ticker", "filing_date", "entry", "year", "exit_20",
            "exit_60", "net_20", "net_60"]


def write_rows(rows: list[F.Row], path) -> None:
    """Every feature row with its identifiers and targets, for the cross-check script."""
    import csv
    import gzip
    import io
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(ROW_META + F.FEATURES)
    for r in rows:
        e, o = r.result.event, r.result.outcome
        meta = [e.filer_id, e.issuer_cik or "", e.ticker, e.filing_date, o.entry, r.year]
        meta += [(o.legs.get(h) or {}).get("exit", "") for h in (20, 60)]
        meta += ["" if (v := r.result.r(h, net=True)) is None else repr(v) for h in (20, 60)]
        feats = [r.features[f] if isinstance(r.features[f], str)
                 else ("" if r.features[f] != r.features[f] else repr(r.features[f]))
                 for f in F.FEATURES]
        w.writerow(meta + feats)
    with gzip.open(path, "wt") as fh:
        fh.write(buf.getvalue())


def write_events(events, path) -> None:
    """Every event (buys and sells): what `Activity` counts."""
    import csv
    import gzip
    with gzip.open(path, "wt", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["filer_id", "issuer_cik", "ticker", "side", "filing_date"])
        for e in events:
            w.writerow([e.filer_id, e.issuer_cik or "", e.ticker, e.side, e.filing_date])
