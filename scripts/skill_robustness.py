"""How fragile is the proven-insider gate? Variations on the gate, years left out, periods.

    python scripts/skill_robustness.py --cache pool.pkl --prices DIR \
        --insiders history.csv.gz --out robustness.md

Every gate is applied to what the ungated two-input model flags (60 days, $50M floor),
fold by fold, and compared with random gates that keep the same number of buys each year.
The main gate (shrunk record > 0, at least 3 episodes) was chosen before seeing results;
the others are sensitivity checks, so a good row among them is not a new finding.
"""

from __future__ import annotations

import argparse
import pickle
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tradetracker import skill as SK  # noqa: E402
from tradetracker.backtest import prices as bp  # noqa: E402
from tradetracker.backtest.study import num, pct, table  # noqa: E402
from tradetracker.scoring import skill_run as SR  # noqa: E402
from tradetracker.scoring import skill_study as SS  # noqa: E402
from tradetracker.scoring import study as S  # noqa: E402

GATES = {
    "shrunk record > 0 (main)": lambda s: s.skill_60 is not None and s.skill_60 > 0,
    "shrunk record > 1%": lambda s: s.skill_60 is not None and s.skill_60 > 0.01,
    "shrunk record > 2%": lambda s: s.skill_60 is not None and s.skill_60 > 0.02,
    "shrunk record > 0, hit rate >= 50%": lambda s: s.skill_60 is not None and s.skill_60 > 0
    and s.skill_hit60 >= 0.5,
    "raw mean > 0, 3+ episodes": lambda s: s.skill_n60 >= 3 and s.skill_raw60 > 0,
    "raw mean > 0, 5+ episodes": lambda s: s.skill_n60 >= 5 and s.skill_raw60 > 0,
    "raw mean > 0, 1+ episode": lambda s: s.skill_n60 >= 1 and s.skill_raw60 > 0,
    "consistent winners": lambda s: SS.consistent(s, 60),
    "20-day record > 0": lambda s: s.skill_20 is not None and s.skill_20 > 0,
    "any record (control)": lambda s: s.skill_60 is not None,
    "no record (control)": lambda s: s.skill_60 is None,
}


def pooled(sets, years=None):
    xs = [x for y, v in sets.items() if years is None or y in years for x in v]
    return S._summ(xs, 60, "spy")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--prices", required=True)
    ap.add_argument("--insiders")
    ap.add_argument("--out", required=True)
    ap.add_argument("--draws", type=int, default=500)
    a = ap.parse_args()
    results, sigs, keys = pickle.loads(Path(a.cache).read_bytes())
    prices = bp.open_source(a.prices, None)
    books = SK.build_books(results, prices, keys,
                           history=Path(a.insiders) if a.insiders else None,
                           tickers=SR.issuer_tickers(results))
    sigs = SR.attach(sigs, results, books, keys)
    cfg = S.Config("two inputs, 60 days, cap floor", h=60, min_cap=50e6,
                   components=S.TWO_INPUTS)
    folds = S.walk_forward(results, sigs, cfg=cfg)
    base = {f.year: [x for x in f.scored if x.above] for f in folds}
    years = sorted(base)
    rng = random.Random(11)

    rows = []
    for name, gate in GATES.items():
        g = {y: [x for x in v if gate(x.signals)] for y, v in base.items()}
        s_all, s_x20 = pooled(g), pooled(g, [y for y in years if y != 2020])
        s_a, s_b = pooled(g, range(2017, 2023)), pooled(g, range(2023, 2027))
        by_year = [pooled(g, [y]).mean for y in years]
        base_year = [pooled(base, [y]).mean for y in years]
        beats = sum(1 for m, b in zip(by_year, base_year) if m is not None and b is not None and m > b)
        pos = sum(1 for m in by_year if m is not None and m > 0)
        means, means_x20 = [], []
        for _ in range(a.draws):
            pick = {y: rng.sample(v, len(g[y])) for y, v in base.items()}
            means.append(pooled(pick).mean)
            means_x20.append(pooled(pick, [y for y in years if y != 2020]).mean)
        p = (sum(m >= s_all.mean for m in means if m is not None) + 1) / (len(means) + 1)
        p20 = (sum(m >= s_x20.mean for m in means_x20 if m is not None) + 1) / (len(means_x20) + 1)
        rows.append([name, s_all.n, pct(s_all.mean), num(s_all.t_cluster),
                     num(p, 3), pct(s_x20.mean), num(p20, 3), pct(s_a.mean), pct(s_b.mean),
                     f"{pos} of {len(years)}", f"{beats} of {len(years)}"])
    b_all, b_x20 = pooled(base), pooled(base, [y for y in years if y != 2020])
    rows.insert(0, ["no gate (the paper book's signals)", b_all.n, pct(b_all.mean),
                    num(b_all.t_cluster), "", pct(b_x20.mean), "",
                    pct(pooled(base, range(2017, 2023)).mean),
                    pct(pooled(base, range(2023, 2027)).mean),
                    f"{sum(1 for y in years if (pooled(base, [y]).mean or 0) > 0)} of {len(years)}", ""])
    text = table(["gate", "flagged", "net mean, 60 days", "t (clustered)",
                  "random gates at least as good", "without 2020",
                  "random gates at least as good (without 2020)", "2017-2022", "2023-2026",
                  "years positive", "years beating no gate"], rows)

    # Book: the main gate in 20 and 40 slots, with and without 2020's entries.
    main_gate = GATES["shrunk record > 0 (main)"]
    book_rows = []
    for label, keep in (("all years", lambda y: True), ("without 2020", lambda y: y != 2020)):
        for gname, sets in (("no gate", base),
                            ("main gate", {y: [x for x in v if main_gate(x.signals)]
                                           for y, v in base.items()})):
            cands = [x for y, v in sets.items() if keep(y) for x in v]
            for slots in (20, 40):
                pf = S.simulate(cands, prices, 60, slots=slots)
                book_rows.append([label, gname, slots, pf.trades, num(pf.ir()),
                                  pct(pf.annual_excess()), num(pf.mdd() / pf.spy_mdd()
                                                               if pf.spy_mdd() < 0 else None)])
    btext = table(["years", "signals", "slots", "positions", "IR", "excess/yr",
                   "drawdown vs SPY"], book_rows)
    Path(a.out).write_text("# Proven-insider gate: robustness\n\n" + __doc__.split("\n\n", 2)[2]
                           + "\n\n" + text + "\n\n## Book\n\n" + btext + "\n")
    print(text)
    print(btext)
    return 0


if __name__ == "__main__":
    sys.exit(main())
