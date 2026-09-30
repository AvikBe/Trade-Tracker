"""Summary statistics for bucketed event returns.

Event returns overlap in time: many insiders file in the same weeks, and a 20-day
return shares most of its days with the next day's. A plain t-stat treats them as
independent and overstates significance, so every summary also reports a t-stat with
standard errors clustered by entry period (calendar month for 5 and 20 days, calendar
quarter for 60 days, so each cluster spans at least one holding period).
"""

from __future__ import annotations

import bisect
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import date


@dataclass
class Summary:
    n: int
    mean: float | None = None
    median: float | None = None
    t: float | None = None           # naive
    t_cluster: float | None = None   # clustered by entry period
    hit: float | None = None         # share of events with a positive return
    wmean: float | None = None       # mean after winsorizing at the given cutoffs
    clusters: int = 0


def cluster_key(d: date, horizon: int) -> tuple[int, int]:
    return (d.year, d.month) if horizon <= 20 else (d.year, (d.month - 1) // 3)


def summarize(values: list[float], keys: list | None = None,
              winsor: tuple[float, float] | None = None) -> Summary:
    n = len(values)
    if n == 0:
        return Summary(0)
    mean = sum(values) / n
    s = Summary(n, mean=mean, median=statistics.median(values),
                hit=sum(1 for v in values if v > 0) / n)
    if n > 1:
        sd = statistics.stdev(values)
        s.t = mean / (sd / math.sqrt(n)) if sd > 0 else None
    if winsor:
        lo, hi = winsor
        s.wmean = sum(min(max(v, lo), hi) for v in values) / n
    if keys is not None and n > 1:
        # Cluster-robust (CR0 with the usual G/(G-1) correction) SE of the mean.
        sums: dict = defaultdict(float)
        for v, k in zip(values, keys):
            sums[k] += v - mean
        g = len(sums)
        s.clusters = g
        if g > 1:
            var = sum(x * x for x in sums.values()) / (n * n) * g / (g - 1)
            s.t_cluster = mean / math.sqrt(var) if var > 0 else None
    return s


def quantile_cutoffs(values: list[float], q: int) -> list[float]:
    """The q-1 inner cutoffs splitting values into q groups of (nearly) equal size."""
    v = sorted(values)
    if not v:
        return []
    return [v[min(len(v) - 1, int(len(v) * k / q))] for k in range(1, q)]


def bucket_of(x: float, cutoffs: list[float]) -> int:
    """0-based group: values equal to a cutoff go to the upper group."""
    return bisect.bisect_right(cutoffs, x)


def winsor_cutoffs(values: list[float], lo: float = 0.01, hi: float = 0.99) -> tuple[float, float]:
    v = sorted(values)
    return v[int(lo * (len(v) - 1))], v[int(hi * (len(v) - 1))]


def diff_summary(a: list[float], b: list[float], keys_a: list | None = None,
                 keys_b: list | None = None) -> tuple[float | None, float | None]:
    """Mean of a minus mean of b, with a t-stat.

    With cluster keys the standard error is cluster-robust: both groups' deviations are
    summed per cluster, so a month that lifts both groups doesn't count as evidence.
    Without keys it is Welch's t.
    """
    if len(a) < 2 or len(b) < 2:
        return None, None
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    if keys_a is None or keys_b is None:
        se = math.sqrt(statistics.variance(a) / len(a) + statistics.variance(b) / len(b))
    else:
        sums: dict = defaultdict(float)
        for v, k in zip(a, keys_a):
            sums[k] += (v - ma) / len(a)
        for v, k in zip(b, keys_b):
            sums[k] -= (v - mb) / len(b)
        g = len(sums)
        if g < 2:
            return ma - mb, None
        se = math.sqrt(sum(x * x for x in sums.values()) * g / (g - 1))
    return ma - mb, ((ma - mb) / se if se > 0 else None)


def pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3:
        return None
    mx, my = sum(x) / len(x), sum(y) / len(y)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    return sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else None


def ranks(x: list[float]) -> list[float]:
    order = sorted(range(len(x)), key=lambda i: x[i])
    r = [0.0] * len(x)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and x[order[j + 1]] == x[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x: list[float], y: list[float]) -> float | None:
    return pearson(ranks(x), ranks(y))
