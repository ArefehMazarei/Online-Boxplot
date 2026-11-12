"""Online boxplot vs the two batch references, on the paper's simulated data.

Setup as in the paper's notebook: numpy seed 0, 10,000 points, first window of
100 points, bins one standard deviation wide, Chebyshev k=55, w=3. The
constants a and b are fixed here (they are what the paper's optimiser returns
for these series, see calibrate.py), so the test does not need scipy.

Two batch references, and they answer different questions:
  whole stream : fences fitted once on all the data, "if I could see everything".
                 This is what the published Table 1 reports.
  expanding    : fences from the data seen before each point, "if I stored
                 everything so far". Same information as the online method at
                 each step, so the fairest comparison.
"""
import numpy as np
import pytest

from onlineboxplot import batch_detect, expanding_detect, online_detect

WARMUP = 100
KW = dict(n_bins="std", k=55, floor_bins=0.0)


def series(kind):
    np.random.seed(0)
    if kind == "right":
        return np.random.gamma(0.3, 10, 10000), dict(a=2.14, b=2.24)
    return -np.random.gamma(0.4, 5, 10000), dict(a=2.12, b=2.17)


def sets(kind):
    x, ab = series(kind)
    online = {o.index for o in online_detect(x, warmup=WARMUP, **ab, **KW)}
    whole = {o.index for o in batch_detect(x, "qsm", w=3, **ab) if o.index >= WARMUP}
    expanding = {o.index for o in expanding_detect(x, "qsm", warmup=WARMUP, w=3, **ab)}
    return online, whole, expanding


# kind: (#online, #whole-stream, TP vs whole, #expanding, TP vs expanding)
EXPECTED = {"right": (18, 16, 16, 22, 18), "left": (21, 18, 18, 25, 18)}


@pytest.mark.parametrize("kind", ["right", "left"])
def test_counts_against_both_references(kind):
    online, whole, expanding = sets(kind)
    n_on, n_whole, tp_whole, n_exp, tp_exp = EXPECTED[kind]
    assert (len(online), len(whole), len(online & whole), len(expanding), len(online & expanding)) \
        == (n_on, n_whole, tp_whole, n_exp, tp_exp)


@pytest.mark.parametrize("kind", ["right", "left"])
def test_online_is_close_to_both_references(kind):
    online, whole, expanding = sets(kind)
    for ref in (whole, expanding):
        tp = len(online & ref)
        p, r = tp / len(online), tp / len(ref)
        assert 2 * p * r / (p + r) > 0.75
