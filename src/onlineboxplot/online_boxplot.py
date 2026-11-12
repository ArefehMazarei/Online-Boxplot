"""Online boxplot outlier detector.

Re-implemented from the published method:

    Mazarei et al. (2025), "Online boxplot derived outlier detection",
    International Journal of Data Science and Analytics 19:83-97,
    https://doi.org/10.1007/s41060-024-00559-0   (open access, CC BY 4.0)

How it works:
  1. Keep a histogram with a bounded number of bins instead of the data.
  2. Read Q1, Q2, Q3 from the histogram's cumulative distribution (Alg. 1, 2).
  3. Turn the quartiles into whiskers with a rule from `fences.py`. The
     paper's rule is "qsm": an exponential model on the quartile skewness.
  4. A new point outside the whiskers is an outlier.
  5. Points more than k standard deviations from the running mean are flagged
     but not added to the histogram (Chebyshev bound, Eq. 11).

Memory is constant: the histogram never keeps more than `max_bins` bins.

Details worth knowing:
  * Quartiles are only known to one bin. When two or three of them land in the
    same bin, the box is widened by one bin width (Q1 down, Q3 up), as in the
    author's reference implementation. Without this, coarse histograms on
    skewed data would flag a large share of ordinary points.
  * floor_bins: on top of that, whiskers are never closer than this many bin
    widths to Q1/Q3 (a safety net for flat series). Set 0 to switch it off.
  * NaN values are skipped: never flagged, never learned.
  * The first `warmup` points only initialise the histogram and are not
    tested, unless flag_warmup=True in `online_detect`.
  * a and b of the "qsm" rule were fitted per series in the paper
    (see calibrate.py). The defaults (-4, 0.5) are the starting point of that
    fit and work well with the default std-wide bins.
"""
from __future__ import annotations

import math

import numpy as np

from . import fences as fn
from .base import Outlier, outlier_score


class OnlineBoxplot:
    def __init__(
        self,
        rule: str = "qsm",
        n_bins: int | str = "std",
        max_bins: int = 1000,
        k: float | None = 55.0,
        floor_bins: float = 1.0,
        **rule_kwargs,
    ):
        """
        rule:       "qsm" (paper), "tukey", "siqr" or "ratio".
        n_bins:     bins in the first histogram. "std" (default): bins one
                    standard deviation wide, i.e. ceil(range / std) bins, as in
                    the paper's simulation study and the author's reference
                    code. Or an int, e.g. 150 (the paper's setting for real
                    data). Finer bins give more precise quartiles, but then
                    the whisker constants a and b matter much more, and
                    calibrate_qsm becomes unreliable (see calibrate.py).
        max_bins:   bins are merged in pairs when this is reached (paper: 1000).
        k:          Chebyshev factor (paper: 55). None switches it off.
        floor_bins: minimum whisker distance from the box, in bin widths.
        rule_kwargs: w, a, b for the rule. Defaults: w=3, a=-4, b=0.5 for
                    "qsm" (see fences.py); w=1.5 for "tukey" and "ratio";
                    w=3 for "siqr".
        """
        if rule not in fn.QUARTILE_RULES:
            raise ValueError(f"unknown rule {rule!r}; choose from {tuple(fn.QUARTILE_RULES)}")
        if not (n_bins == "std" or (isinstance(n_bins, int) and n_bins >= 2)):
            raise ValueError('n_bins must be an int >= 2 or "std"')
        if max_bins < 4:
            raise ValueError("need max_bins >= 4")
        self.rule, self.rule_kwargs = rule, rule_kwargs
        self.n_bins, self.max_bins, self.k, self.floor_bins = n_bins, max_bins, k, floor_bins
        self._rule_fn = fn.QUARTILE_RULES[rule]
        self._counts: np.ndarray | None = None
        self._lo = 0.0        # left edge of the first bin
        self._width = 1.0     # width of every bin
        self._n = 0           # running mean / variance (Welford), O(1) memory
        self._mean = 0.0
        self._m2 = 0.0
        self.lower = self.upper = float("nan")
        self.q = (float("nan"),) * 3

    # ---------- step 1: initialise on a buffer (Algorithm 1) ----------
    def fit(self, buffer) -> "OnlineBoxplot":
        x = np.asarray(buffer, dtype=float)
        x = x[~np.isnan(x)]
        if x.size < 4:
            raise ValueError("need at least 4 valid points to initialise")
        lo, hi = float(x.min()), float(x.max())
        std = float(x.std())
        if hi == lo:  # constant buffer: make a small non-empty range
            pad = abs(lo) * 0.01 or 1.0
            lo, hi = lo - pad, hi + pad
        if self.n_bins == "std":
            nb = max(2, math.ceil((hi - lo) / std)) if std > 0 else 2
        else:
            nb = self.n_bins
        self._lo = lo
        self._width = (hi - lo) / nb
        idx = np.minimum(((x - lo) / self._width).astype(int), nb - 1)
        self._counts = np.bincount(idx, minlength=nb).astype(np.int64)
        self._n, self._mean = x.size, float(x.mean())
        self._m2 = float(((x - self._mean) ** 2).sum())
        self._refresh()
        return self

    # ---------- step 2: stream (Algorithm 2) ----------
    def update(self, x: float) -> bool:
        """Test one new point against the current whiskers, then learn from it.

        Returns True if the point is an outlier.
        """
        if self._counts is None:
            raise RuntimeError("call fit(buffer) first")
        x = float(x)
        if math.isnan(x):
            return False
        is_outlier = x < self.lower or x > self.upper
        if not self._too_extreme(x):
            self._add_to_histogram(x)
            self._welford(x)
            self._refresh()
        return is_outlier

    def score(self, x: float) -> float:
        """Distance beyond the nearest whisker in IQR units (0 if inside)."""
        return outlier_score(float(x), self.lower, self.upper, self.q[2] - self.q[0])

    @property
    def n_bins_now(self) -> int:
        return 0 if self._counts is None else len(self._counts)

    @property
    def bin_width(self) -> float:
        return self._width

    # ---------- internals ----------
    def _too_extreme(self, x: float) -> bool:
        if self.k is None or self._n < 2:
            return False
        std = math.sqrt(self._m2 / self._n)
        return std > 0 and abs(x - self._mean) > self.k * std

    def _add_to_histogram(self, x: float) -> None:
        c = self._counts
        if x >= self._lo + self._width * len(c):          # grow to the right
            need = int((x - self._lo) / self._width) + 1 - len(c)
            c = np.concatenate([c, np.zeros(need, dtype=c.dtype)])
        elif x < self._lo:                                # grow to the left
            need = int(math.ceil((self._lo - x) / self._width))
            c = np.concatenate([np.zeros(need, dtype=c.dtype), c])
            self._lo -= need * self._width
        j = min(max(int((x - self._lo) / self._width), 0), len(c) - 1)
        c[j] += 1
        while len(c) >= self.max_bins:                    # merge in pairs from the left
            if len(c) % 2:
                c = np.concatenate([c, [0]])
            c = c[0::2] + c[1::2]
            self._width *= 2
        self._counts = c

    def _welford(self, x: float) -> None:
        self._n += 1
        d = x - self._mean
        self._mean += d / self._n
        self._m2 += d * (x - self._mean)

    def _quartiles(self) -> tuple[float, float, float]:
        """Q1, Q2, Q3: midpoint of the first bin whose cumulative count
        reaches 25%, 50%, 75% of the total (Alg. 1, lines 7-21)."""
        cum = np.cumsum(self._counts)
        total = int(cum[-1])
        out = []
        for p in (0.25, 0.5, 0.75):
            i = int(np.searchsorted(cum, math.ceil(total * p), side="left"))
            out.append(self._lo + (i + 0.5) * self._width)
        return out[0], out[1], out[2]

    def _refresh(self) -> None:
        q1, q2, q3 = self._quartiles()
        bl = self._width
        if q3 == q1:        # all three in one bin: box is one bin wide each way
            q1, q3 = q1 - bl, q3 + bl
        if q3 == q2:
            q3 += bl
        if q2 == q1:
            q1 -= bl
        self.q = (q1, q2, q3)
        lower, upper = self._rule_fn(q1, q2, q3, **self.rule_kwargs)
        margin = self.floor_bins * self._width
        self.lower = min(lower, q1 - margin)
        self.upper = max(upper, q3 + margin)


def online_detect(
    values, rule: str = "qsm", warmup: int = 200, flag_warmup: bool = False,
    calibrate: bool = False, calibrate_points: int | None = None, **kwargs
) -> list[Outlier]:
    """Run the online boxplot over a series, as if it arrived point by point.

    The first `warmup` points build the first histogram. Later points are
    tested in order, each with the whiskers known just before it arrived.
    With flag_warmup=True the warm-up points are also tested, against the
    whiskers of the first histogram.
    With calibrate=True (rule "qsm" only) a and b are fitted first on the
    first `calibrate_points` points (default 2 x warmup), as in the paper;
    see calibrate.calibrate_qsm.
    NaN values are skipped.
    """
    x = np.asarray(values, dtype=float)
    if calibrate:
        if rule != "qsm":
            raise ValueError("calibrate=True applies to rule 'qsm' only")
        from .calibrate import calibrate_qsm
        fit_kw = {k_: v for k_, v in kwargs.items() if k_ in ("n_bins", "max_bins", "k", "w")}
        for name, val in calibrate_qsm(x[: (calibrate_points or 2 * warmup)], **fit_kw).items():
            kwargs.setdefault(name, val)
    det = OnlineBoxplot(rule, **kwargs).fit(x[:warmup])
    out: list[Outlier] = []
    if flag_warmup:
        lo, up, iqr = det.lower, det.upper, det.q[2] - det.q[0]
        out += [Outlier(i, float(x[i]), lo, up, outlier_score(x[i], lo, up, iqr))
                for i in np.flatnonzero((x[:warmup] < lo) | (x[:warmup] > up))]
    for t in range(warmup, len(x)):
        if np.isnan(x[t]):
            continue
        lo, up, iqr = det.lower, det.upper, det.q[2] - det.q[0]
        if det.update(x[t]):
            out.append(Outlier(t, float(x[t]), lo, up, outlier_score(x[t], lo, up, iqr)))
    return out
