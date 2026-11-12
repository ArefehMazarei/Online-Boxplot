"""Fence rules: hand-computed values and limiting cases."""
import math

import pytest

from onlineboxplot import fences as fn

Q = (10.0, 12.0, 20.0)  # right-skewed box: Q2-Q1 = 2, Q3-Q2 = 8, IQR = 10


def test_tukey():
    assert fn.tukey(*Q) == (10 - 15, 20 + 15)
    assert fn.tukey(*Q, w=3) == (10 - 30, 20 + 30)


def test_siqr_uses_each_half_of_the_box():
    # lower: Q1 - 3*(Q2-Q1) = 4 ; upper: Q3 + 3*(Q3-Q2) = 44
    assert fn.siqr(*Q) == (4.0, 44.0)


def test_quartile_skewness():
    assert fn.quartile_skewness(*Q) == pytest.approx(0.6)       # (8-2)/10
    assert fn.quartile_skewness(0, 5, 10) == 0.0
    assert fn.quartile_skewness(10, 5, 0) == 0.0                # degenerate box, no crash
    assert fn.quartile_skewness(1, 1, 1) == 0.0


def test_ratio_skewed():
    bc = 0.6
    lo, up = fn.ratio_skewed(*Q)
    assert lo == pytest.approx(10 - 1.5 * 10 * (1 - bc) / (1 + bc))
    assert up == pytest.approx(20 + 1.5 * 10 * (1 + bc) / (1 - bc))


def test_ratio_skewed_stays_finite_at_extreme_skew():
    lo, up = fn.ratio_skewed(0, 0, 10)      # Bc = 1
    assert math.isfinite(lo) and math.isfinite(up)


def test_qsm_right_skew():
    s = 0.6
    lo, up = fn.qsm(*Q, w=3, a=-4, b=3)     # Hubert-Vandervieren constants
    assert lo == pytest.approx(10 - 3 * math.exp(-4 * s) * 10)
    assert up == pytest.approx(20 + 3 * math.exp(3 * s) * 10)


def test_qsm_is_mirror_symmetric():
    """Flipping the data (x -> -x) must flip the fences."""
    lo, up = fn.qsm(10, 12, 20)
    lo2, up2 = fn.qsm(-20, -12, -10)
    assert (lo2, up2) == pytest.approx((-up, -lo))


def test_qsm_reduces_to_tukey_style_when_symmetric():
    lo, up = fn.qsm(0, 5, 10, w=1.5)
    assert (lo, up) == pytest.approx(fn.tukey(0, 5, 10, w=1.5))


def test_default_constants():
    assert (fn.QSM_A, fn.QSM_B) == (-4.0, 0.5)
    assert (fn.HV_A, fn.HV_B) == (-4.0, 3.0)


def test_adjusted_mc_mirror_and_zero():
    assert fn.adjusted_mc(0, 10, 0.0) == pytest.approx(fn.tukey(0, 5, 10, w=1.5))
    lo, up = fn.adjusted_mc(10, 20, 0.4)
    lo2, up2 = fn.adjusted_mc(-20, -10, -0.4)
    assert (lo2, up2) == pytest.approx((-up, -lo))
