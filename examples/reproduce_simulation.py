"""Reproduce the paper's simulation study (section 5.1).

Setup as in the paper's notebook: numpy seed 0, 10,000 points, first window of
100 points, bins one standard deviation wide, Chebyshev k = 55, w = 3.
The whisker constants a and b are fitted per series (see onlineboxplot.calibrate);
the fitted values are fixed below so the script runs without scipy.
Pass --fit to fit them instead (needs scipy).

Online results are compared with two batch references:
  whole stream : fences fitted once on all data (what the paper's Table 1 reports)
  expanding    : fences from the data seen before each point
"""
import sys

import numpy as np

from onlineboxplot import batch_detect, expanding_detect, online_detect

WARMUP = 100
KW = dict(n_bins="std", k=55, floor_bins=0.0)
DATA = {
    "right-skewed  Gamma(0.3, scale 10)": (lambda: np.random.gamma(0.3, 10, 10000), dict(a=2.14, b=2.24)),
    "left-skewed  -Gamma(0.4, scale 5)": (lambda: -np.random.gamma(0.4, 5, 10000), dict(a=2.12, b=2.17)),
}


def prf(ref, found):
    tp = len(ref & found)
    p = tp / len(found) if found else float("nan")
    r = tp / len(ref) if ref else float("nan")
    f = 2 * p * r / (p + r) if tp else float("nan")
    return tp, p, r, f


for name, (make, ab) in DATA.items():
    np.random.seed(0)
    x = make()
    if "--fit" in sys.argv:
        from onlineboxplot.calibrate import calibrate_qsm
        ab = calibrate_qsm(x[:1000], n_bins="std", k=55)
    tukey = len(batch_detect(x, "tukey")) / len(x) * 100
    on = {o.index for o in online_detect(x, warmup=WARMUP, **ab, **KW)}
    whole = {o.index for o in batch_detect(x, "qsm", w=3, **ab) if o.index >= WARMUP}
    expd = {o.index for o in expanding_detect(x, "qsm", warmup=WARMUP, w=3, **ab)}
    print(f"\n{name}   a={ab['a']}, b={ab['b']}")
    print(f"  Tukey (batch, w=1.5) flags {tukey:.1f}% of all points")
    print(f"  online boxplot flags {len(on)} of {len(x) - WARMUP} points ({len(on) / (len(x) - WARMUP) * 100:.2f}%)")
    print(f"  {'reference':<14}{'#ref':>5}{'TP':>5}{'precision':>11}{'recall':>8}{'F1':>6}")
    for label, ref in (("whole stream", whole), ("expanding", expd)):
        tp, p, r, f = prf(ref, on)
        print(f"  {label:<14}{len(ref):>5}{tp:>5}{p:>11.2f}{r:>8.2f}{f:>6.2f}")
