"""Fit the constants a and b of the "qsm" rule, as in the paper.

The paper does not fix a and b. It chooses them per series by making the
online whiskers as close as possible to the whiskers of the batch scenario
(exact quartiles of all data seen so far), on data twice as long as the
initial buffer. The objective is

    RMSE(lower whiskers) + RMSE(upper whiskers)

between the online and the batch whiskers, minimised with scipy's default
optimiser, starting from (a, b) = (-4, 0.5). The author's reference code
calls these p1 and p2, with a = -p1 and b = p2.

Caveat: the objective measures how close the online whiskers are to the batch
whiskers, not how good the detection is. With bins one standard deviation wide
(n_bins="std", the default) it lands on sensible values. With fine bins (e.g.
150) it can drift to extreme constants that collapse the whiskers and flag a
large share of ordinary points. Use calibration with n_bins="std", and check
the share of flagged points on your data.

Needs scipy (optional dependency: pip install scipy).
"""
from __future__ import annotations

import numpy as np

from . import fences as fn
from .online_boxplot import OnlineBoxplot


def calibrate_qsm(
    values,
    n_bins: int | str = "std",
    max_bins: int = 1000,
    k: float | None = 55.0,
    w: float = 3.0,
    start: tuple[float, float] = (fn.QSM_A, fn.QSM_B),
) -> dict:
    """Return {"a": ..., "b": ...} fitted on `values`.

    The first half of `values` initialises the histogram, the second half is
    streamed. Use at least a few hundred points per half. The result is
    rounded to two decimals, like the reference code.
    """
    try:
        from scipy.optimize import minimize
    except ImportError as e:  # pragma: no cover
        raise ImportError("calibrate_qsm needs scipy: pip install scipy") from e

    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    half = len(x) // 2
    if half < 50:
        raise ValueError("need at least 100 valid points to calibrate")

    # Batch whiskers do not depend on (a, b) in their quartiles: compute once.
    batch_q = [np.percentile(x[: t + 1], [25, 50, 75]) for t in range(half, len(x))]

    def objective(params) -> float:
        a, b = -params[0], params[1]  # (p1, p2) -> (a, b)
        det = OnlineBoxplot("qsm", n_bins=n_bins, max_bins=max_bins, k=k,
                            floor_bins=0.0, w=w, a=a, b=b).fit(x[:half])
        on = [(det.lower, det.upper)]
        for v in x[half:]:
            det.update(v)
            on.append((det.lower, det.upper))
        on = np.nan_to_num(np.array(on[1:], dtype=float))
        ba = np.nan_to_num(np.array([fn.qsm(*q, w=w, a=a, b=b) for q in batch_q]))
        rmse = np.sqrt(np.mean((ba - on) ** 2, axis=0))   # lower, upper
        return float(rmse.sum())

    res = minimize(objective, [-start[0], start[1]])
    return {"a": round(-float(res.x[0]), 2), "b": round(float(res.x[1]), 2)}
