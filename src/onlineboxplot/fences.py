"""Fence (whisker) rules of the boxplot family.

Every rule is a small pure function. Quartile-based rules take (q1, q2, q3)
and return (lower, upper). They do not care where the quartiles come from, so
the same rule runs on exact quartiles (batch) and on histogram quartiles
(online).

Equation numbers refer to:
    Mazarei et al. (2025), "Online boxplot derived outlier detection",
    Int. J. Data Sci. Anal. 19:83-97, https://doi.org/10.1007/s41060-024-00559-0
"""
from __future__ import annotations

import math
from typing import Callable

Fences = tuple[float, float]

# Constants from Hubert and Vandervieren (2008), used by the adjusted boxplot.
HV_A, HV_B = -4.0, 3.0
# Constants tuned by the author for the online quartile-skewness rule.
# Hubert and Vandervieren's own values (-4, 3) are available as HV_A, HV_B.
QSM_A, QSM_B = -4.0, 0.5


def quartile_skewness(q1: float, q2: float, q3: float) -> float:
    """Quartile skewness (Bowley's coefficient), Eq. 6 and Eq. 9. In [-1, 1]."""
    iqr = q3 - q1
    if iqr <= 0:
        return 0.0
    return ((q3 - q2) - (q2 - q1)) / iqr


def tukey(q1: float, q2: float, q3: float, w: float = 1.5) -> Fences:
    """Tukey's boxplot, Eq. 1. Use w=1.5 (usual) or w=3 (extreme outliers)."""
    iqr = q3 - q1
    return q1 - w * iqr, q3 + w * iqr


def siqr(q1: float, q2: float, q3: float, w: float = 3.0) -> Fences:
    """Semi-interquartile fences of Kimber (1990), Eq. 2.

    The lower fence uses the lower half of the box (Q2 - Q1), the upper fence
    the upper half (Q3 - Q2). The paper's form is w = 3, i.e. 2 x 1.5.
    """
    return q1 - w * (q2 - q1), q3 + w * (q3 - q2)


def ratio_skewed(q1: float, q2: float, q3: float, w: float = 1.5) -> Fences:
    """Ratio-skewed boxplot of Walker et al. (2018), Eq. 6-8."""
    iqr = q3 - q1
    bc = max(min(quartile_skewness(q1, q2, q3), 0.999), -0.999)  # keep ratios finite
    r_lower = (1 - bc) / (1 + bc)
    r_upper = (1 + bc) / (1 - bc)
    return q1 - w * iqr * r_lower, q3 + w * iqr * r_upper


def qsm(
    q1: float, q2: float, q3: float,
    w: float = 3.0, a: float = QSM_A, b: float = QSM_B,
) -> Fences:
    """Online-boxplot rule: exponential model on the quartile skewness, Eq. 10.

    QSM >= 0 (right skew): lower = Q1 - w exp(a QSM) IQR, upper = Q3 + w exp(b QSM) IQR
    QSM <  0 (left skew):  lower = Q1 - w exp(-b QSM) IQR, upper = Q3 + w exp(-a QSM) IQR
    """
    iqr = q3 - q1
    s = quartile_skewness(q1, q2, q3)
    if s >= 0:
        lo_f, up_f = math.exp(a * s), math.exp(b * s)
    else:
        lo_f, up_f = math.exp(-b * s), math.exp(-a * s)
    return q1 - w * lo_f * iqr, q3 + w * up_f * iqr


def adjusted_mc(
    q1: float, q3: float, mc: float,
    w: float = 1.5, a: float = HV_A, b: float = HV_B,
) -> Fences:
    """Adjusted boxplot of Hubert and Vandervieren (2008), Eq. 5.

    Needs the medcouple `mc` of the data, so it is batch only.
    """
    iqr = q3 - q1
    if mc >= 0:
        lo_f, up_f = math.exp(a * mc), math.exp(b * mc)
    else:
        lo_f, up_f = math.exp(-b * mc), math.exp(-a * mc)
    return q1 - w * lo_f * iqr, q3 + w * up_f * iqr


# Rules that work from the three quartiles alone, selectable by name.
QUARTILE_RULES: dict[str, Callable[..., Fences]] = {
    "tukey": tukey,
    "siqr": siqr,
    "ratio": ratio_skewed,
    "qsm": qsm,
}
