"""Shared types for the detectors."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Outlier:
    """One detected outlier.

    index:  position in the input series.
    value:  the observed value.
    lower, upper: the fences that were in force when the point was tested.
    score:  how far the value is beyond the nearest fence, in units of the
            IQR (the box width) at that moment. Always > 0 for an outlier.
            It depends only on this point, not on the other outliers.
    """
    index: int
    value: float
    lower: float
    upper: float
    score: float

    def as_dict(self) -> dict:
        return asdict(self)


def outlier_score(x: float, lower: float, upper: float, iqr: float) -> float:
    """Distance beyond the nearest fence, divided by the IQR (0 if inside)."""
    scale = max(iqr, 1e-12)
    if x > upper:
        return (x - upper) / scale
    if x < lower:
        return (lower - x) / scale
    return 0.0
