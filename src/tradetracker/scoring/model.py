"""The v1 score: quality Q, a drift penalty and a decay, with weights from the backtest.

    score = Q * exp(-lambda * d) * max(0, 1 - beta * drift)        (spec, with P = 1)

Q is a weighted sum of components, scaled 0 to 1. Each component puts an event in a
bucket with fixed, round-number edges (so a reason line can say "stake up 25%"), and the
data decides the rest:

1. Bucket values: the mean training return of each bucket minus the overall mean,
   shrunk toward 0 by n / (n + SHRINK_N) so a bucket of 40 events can't dominate.
2. Weights: least squares of training returns on the bucket values, bounded to
   [0, 1]. Components that repeat each other share their weight, and one that doesn't
   help gets 0. A weight can't be negative (a component can't flip the meaning of its
   own table) or above 1 (the fit can't amplify a table beyond its own training means,
   which is how noise columns get large weights in sample).
3. beta: the value on BETA_GRID with the best training top-decile return.

The training target is the 20-day excess return over SPY net of costs, winsorized at
the training 1st and 99th percentiles. Promptness is not in the model: phase 3 found
lag has no out-of-sample effect, and avik accepted P = 1 on 2026-09-30.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

from .signals import NEW_STAKE, Signals

MODEL_VERSION = 1
SHRINK_N = 500
BETA_GRID = (0.0, 1.0, 2.0, 3.0, 5.0, 8.0)
DRIFT_FLOOR = -0.20          # a drop bigger than 20% earns no more bonus than 20%
DEFAULT_HALF_LIFE = 10       # trading days (spec start value for Form 4)
TOP_FRACTION = 0.10


def _edges(x: float | None, edges: list[float], labels: list[str], missing: str) -> str:
    if x is None:
        return missing
    for edge, label in zip(edges, labels):
        if x < edge:
            return label
    return labels[-1]


# name -> (bucket function, bucket labels in display order)
COMPONENTS: dict[str, tuple] = {
    "role": (
        lambda s: s.role or "none",
        ["CEO", "CFO", "officer", "director", "10% owner", "none"],
    ),
    "value": (
        lambda s: _edges(s.value, [10e3, 50e3, 250e3, 1e6],
                         ["<$10k", "$10k-50k", "$50k-250k", "$250k-1M", ">=$1M"], "<$10k"),
        ["<$10k", "$10k-50k", "$50k-250k", "$250k-1M", ">=$1M"],
    ),
    "stake": (
        lambda s: "new stake" if s.stake_change is not None and s.stake_change >= NEW_STAKE
        else _edges(s.stake_change, [0.05, 0.20, 1.0],
                    ["<5%", "5-20%", "20-100%", ">=100%"], "unknown"),
        ["<5%", "5-20%", "20-100%", ">=100%", "new stake", "unknown"],
    ),
    "cluster": (
        lambda s: "1" if s.cluster_count <= 1 else "2" if s.cluster_count == 2 else "3+",
        ["1", "2", "3+"],
    ),
    "repeat": (
        lambda s: _edges(s.repeat_buys, [1, 2, 4], ["0", "1", "2-3", "4+"], "0"),
        ["0", "1", "2-3", "4+"],
    ),
    "drawdown": (
        lambda s: _edges(s.drawdown, [-0.45, -0.25, -0.10],
                         ["<=-45%", "-45 to -25%", "-25 to -10%", "within 10%"], "unknown"),
        ["<=-45%", "-45 to -25%", "-25 to -10%", "within 10%", "unknown"],
    ),
    "track": (
        lambda s: _edges(s.track_record, [-0.05, 0.0, 0.05],
                         ["< -5%", "-5 to 0%", "0 to 5%", ">= 5%"], "none yet"),
        ["< -5%", "-5 to 0%", "0 to 5%", ">= 5%", "none yet"],
    ),
    "liquidity": (
        lambda s: _edges(s.dollar_volume, [1e6, 5e6, 50e6],
                         ["<$1M/day", "$1-5M/day", "$5-50M/day", ">=$50M/day"], "<$1M/day"),
        ["<$1M/day", "$1-5M/day", "$5-50M/day", ">=$50M/day"],
    ),
}


def bucket(name: str, s: Signals) -> str:
    return COMPONENTS[name][0](s)


def nnls(X: list[list[float]], y: list[float], iters: int = 10_000,
         upper: float = math.inf) -> list[float]:
    """Least squares with 0 <= w <= upper, by coordinate descent on the normal equations.

    With a handful of columns, X'X and X'y are tiny, so each sweep is cheap however
    many training events there are.
    """
    k = len(X[0]) if X else 0
    G = [[sum(r[i] * r[j] for r in X) for j in range(k)] for i in range(k)]
    b = [sum(r[i] * v for r, v in zip(X, y)) for i in range(k)]
    w = [0.0] * k
    for _ in range(iters):
        change = 0.0
        for j in range(k):
            if G[j][j] <= 0:
                continue
            g = b[j] - sum(G[j][i] * w[i] for i in range(k))
            new = min(upper, max(0.0, w[j] + g / G[j][j]))
            change = max(change, abs(new - w[j]))
            w[j] = new
        if change < 1e-12:
            break
    return w


@dataclass
class Model:
    tables: dict[str, dict[str, float]]      # component -> bucket -> value (return units)
    counts: dict[str, dict[str, int]]
    weights: dict[str, float]
    beta: float
    half_life: float = DEFAULT_HALF_LIFE
    overall: float = 0.0
    threshold: float | None = None           # training top-decile score cutoff
    trained_on: str = ""
    n_train: int = 0
    version: int = MODEL_VERSION
    notes: dict = field(default_factory=dict)

    # ------------------------------------------------------------ scoring

    def _range(self) -> tuple[float, float]:
        lo = sum(self.weights[c] * min(self.tables[c].values(), default=0)
                 for c in sorted(self.weights))
        hi = sum(self.weights[c] * max(self.tables[c].values(), default=0)
                 for c in sorted(self.weights))
        return lo, hi

    def raw(self, s: Signals) -> float:
        return sum(self.weights[c] * self.tables[c].get(bucket(c, s), 0.0)
                   for c in sorted(self.weights))

    def quality(self, s: Signals) -> float:
        lo, hi = self._range()
        if hi <= lo:
            return 0.5
        return (self.raw(s) - lo) / (hi - lo)

    def drift_factor(self, drift: float | None) -> float:
        if drift is None:
            return 1.0
        return max(0.0, 1.0 - self.beta * max(drift, DRIFT_FLOOR))

    def decay(self, days_since_filing: int = 0) -> float:
        return math.exp(-math.log(2) / self.half_life * days_since_filing)

    def score(self, s: Signals, days_since_filing: int = 0) -> float:
        return self.quality(s) * self.drift_factor(s.drift) * self.decay(days_since_filing)

    def contributions(self, s: Signals) -> dict[str, float]:
        """Each component's share of Q's distance from its minimum (for reasons)."""
        return {c: w * (self.tables[c].get(bucket(c, s), 0.0) - min(self.tables[c].values()))
                for c, w in self.weights.items() if w > 0}

    # ------------------------------------------------------------ persistence

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> Model:
        d = json.loads(text)
        if d.get("version") != MODEL_VERSION:
            raise ValueError(f"model version {d.get('version')} != {MODEL_VERSION}")
        return cls(**d)


def winsorize(ys: list[float], lo: float = 0.01, hi: float = 0.99) -> list[float]:
    s = sorted(ys)
    a = s[int(lo * (len(s) - 1))]
    b = s[int(hi * (len(s) - 1))]
    return [min(max(y, a), b) for y in ys]


def top_mean(scores: list[float], ys: list[float], frac: float = TOP_FRACTION) -> float | None:
    k = max(1, int(len(scores) * frac))
    order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
    return sum(ys[i] for i in order) / k if order else None


def fit(signals: list[Signals], ys: list[float], label: str = "",
        components: list[str] | None = None, beta: float | None = None) -> Model:
    """Fit bucket tables, weights and beta on training events (returns in `ys`)."""
    if len(signals) != len(ys) or not ys:
        raise ValueError("need one return per signal, and at least one")
    names = components or list(COMPONENTS)
    y = winsorize(ys)
    overall = sum(y) / len(y)

    tables, counts = {}, {}
    for c in names:
        sums: dict[str, float] = {}
        ns: dict[str, int] = {}
        for s, v in zip(signals, y):
            b = bucket(c, s)
            sums[b] = sums.get(b, 0.0) + v
            ns[b] = ns.get(b, 0) + 1
        tables[c] = {b: ns[b] / (ns[b] + SHRINK_N) * (sums[b] / ns[b] - overall) for b in ns}
        counts[c] = ns

    X = [[tables[c][bucket(c, s)] for c in names] for s in signals]
    w = nnls(X, [v - overall for v in y], upper=1.0)
    weights = dict(zip(names, w))
    m = Model(tables=tables, counts=counts, weights=weights, beta=0.0, overall=overall,
              trained_on=label, n_train=len(y))

    if beta is None:
        best = None
        for b in BETA_GRID:
            m.beta = b
            t = top_mean([m.score(s) for s in signals], y)
            if t is not None and (best is None or t > best[0] + 1e-12):
                best = (t, b)
        m.beta = best[1] if best else 0.0
    else:
        m.beta = beta
    scores = sorted((m.score(s) for s in signals), reverse=True)
    m.threshold = scores[max(0, int(len(scores) * TOP_FRACTION) - 1)]
    return m


# ---------------------------------------------------------------- reason line

def _money(v: float) -> str:
    if v >= 1e6:
        return f"${v / 1e6:.1f}M"
    if v >= 1e3:
        return f"${v / 1e3:.0f}k"
    return f"${v:.0f}"


def reason(s: Signals, ticker: str, days_since_filing: int = 0) -> str:
    """One line in the spec's style, e.g. 'CFO + 2 insiders bought $3.1M in XYZ ...'."""
    who = s.role or "Insider"
    parts = [f"{who} bought {_money(s.value)} of {ticker}"]
    if s.stake_change is not None:
        parts.append("a new stake" if s.stake_change >= NEW_STAKE
                     else f"stake +{100 * s.stake_change:.0f}%")
    if s.cluster_count >= 2:
        parts.append(f"{s.cluster_count} insiders buying within 14 days")
    parts.append("first buy in 2 years" if s.repeat_buys == 0
                 else "1 earlier buy in 2 years" if s.repeat_buys == 1
                 else f"{s.repeat_buys} earlier buys in 2 years")
    if s.drawdown is not None:
        parts.append(f"stock {abs(100 * s.drawdown):.0f}% below its 52-week high"
                     if s.drawdown < -0.005 else "stock at its 52-week high")
    if s.track_record is not None:
        parts.append(f"past buys {100 * s.track_record:+.1f}% vs SPY (n={s.track_n})")
    if s.drift is not None:
        parts.append(f"{100 * s.drift:+.1f}% since the trade")
    if days_since_filing:
        parts.append(f"filed {days_since_filing} trading days ago")
    return ", ".join(parts)
