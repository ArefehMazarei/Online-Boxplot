"""Batch boxplot detectors: exact quartiles, the whole series in memory.

These are the baselines the online detector is compared with. Rules:
"tukey", "siqr", "ratio" (quartile based) and "adjusted" (medcouple based).
"""
from __future__ import annotations

import numpy as np

from . import fences as fn
from .base import Outlier, outlier_score

BATCH_RULES = ("tukey", "siqr", "ratio", "qsm", "adjusted")
_MC_MAX_POINTS = 4000  # the pairwise medcouple is O(n^2); subsample above this


def medcouple(values, seed: int = 0) -> float:
    """Medcouple, a robust skewness measure in [-1, 1] (Brys et al. 2004, Eq. 3-4).

    Pairs with equal values are skipped, so ties at the median are ignored.
    For more than 4000 points a fixed-seed random subsample is used, which
    keeps the cost bounded; the estimate stays close to the full value.
    """
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    if x.size > _MC_MAX_POINTS:
        x = np.random.default_rng(seed).choice(x, _MC_MAX_POINTS, replace=False)
    m = np.median(x)
    lo, hi = x[x <= m], x[x >= m]
    denom = hi[None, :] - lo[:, None]
    ok = denom > 0
    if not ok.any():
        return 0.0
    num = (hi[None, :] - m) - (m - lo[:, None])
    return float(np.median(num[ok] / denom[ok]))


class BatchBoxplot:
    """Boxplot fences from the exact quartiles of the data it is fitted on.

    rule: "tukey" (Eq. 1), "siqr" (Eq. 2), "ratio" (Eq. 6-8),
          "qsm" (Eq. 10 with exact quartiles), "adjusted" (Eq. 5, medcouple).
    kwargs go to the rule (w, a, b).
    """

    def __init__(self, rule: str = "tukey", **kwargs):
        if rule not in BATCH_RULES:
            raise ValueError(f"unknown rule {rule!r}; choose from {BATCH_RULES}")
        self.rule, self.kwargs = rule, kwargs
        self.q: tuple[float, float, float] | None = None
        self.mc: float | None = None
        self.lower = self.upper = float("nan")

    def fit(self, values) -> "BatchBoxplot":
        x = np.asarray(values, dtype=float)
        x = x[~np.isnan(x)]
        if x.size < 4:
            raise ValueError("need at least 4 valid points")
        q1, q2, q3 = (float(v) for v in np.percentile(x, [25, 50, 75]))
        self.q = (q1, q2, q3)
        if self.rule == "adjusted":
            self.mc = medcouple(x)
            self.lower, self.upper = fn.adjusted_mc(q1, q3, self.mc, **self.kwargs)
        else:
            self.lower, self.upper = fn.QUARTILE_RULES[self.rule](q1, q2, q3, **self.kwargs)
        return self

    @property
    def iqr(self) -> float:
        return self.q[2] - self.q[0]

    def predict(self, values) -> np.ndarray:
        x = np.asarray(values, dtype=float)
        return (x < self.lower) | (x > self.upper)


def batch_detect(values, rule: str = "tukey", **kwargs) -> list[Outlier]:
    """Fit on the whole series and flag every point outside the fences."""
    x = np.asarray(values, dtype=float)
    det = BatchBoxplot(rule, **kwargs).fit(x)
    return [
        Outlier(int(i), float(x[i]), det.lower, det.upper,
                outlier_score(x[i], det.lower, det.upper, det.iqr))
        for i in np.flatnonzero(det.predict(x))
    ]


def expanding_detect(values, rule: str = "tukey", warmup: int = 200, **kwargs) -> list[Outlier]:
    """The paper's "batch scenario": test each point with fences computed from
    all data seen before it, which is what a batch method would do if it
    re-ran at every step. Cost grows with n squared, so use it for
    evaluation, not in production. Quartile rules only. NaN values are skipped.
    """
    if rule == "adjusted":
        raise ValueError("expanding mode supports quartile rules only")
    x = np.asarray(values, dtype=float)
    rule_fn = fn.QUARTILE_RULES[rule]
    out = []
    for t in range(warmup, len(x)):
        if np.isnan(x[t]):
            continue
        past = x[:t]
        past = past[~np.isnan(past)]
        if past.size < 4:
            continue
        q1, q2, q3 = np.percentile(past, [25, 50, 75])
        lo, up = rule_fn(q1, q2, q3, **kwargs)
        if x[t] < lo or x[t] > up:
            out.append(Outlier(t, float(x[t]), lo, up, outlier_score(x[t], lo, up, q3 - q1)))
    return out
