"""Boxplot outlier detectors: batch baselines and the online boxplot.

    from onlineboxplot import detect
    detect(series)                                   # online boxplot (paper)
    detect(series, mode="batch", rule="adjusted")    # Hubert-Vandervieren
    detect(series, mode="expanding", rule="tukey")   # paper's batch scenario

Modes:  "online"    histogram quartiles, constant memory (the paper's method)
        "batch"     exact quartiles of the whole series
        "expanding" exact quartiles of all data seen so far, recomputed each step
Rules:  "tukey", "siqr", "ratio", "qsm"; "adjusted" in batch mode only.
"""
from __future__ import annotations

from .base import Outlier, outlier_score
from .batch import BatchBoxplot, batch_detect, expanding_detect, medcouple
from .online_boxplot import OnlineBoxplot, online_detect

MODES = ("online", "batch", "expanding")


def available_methods() -> list[str]:
    """All valid "mode-rule" combinations."""
    rules = ["tukey", "siqr", "ratio", "qsm"]
    return ([f"online-{r}" for r in rules] + [f"expanding-{r}" for r in rules]
            + [f"batch-{r}" for r in rules + ["adjusted"]])


def detect(values, mode: str = "online", rule: str = "qsm", **kwargs) -> list[Outlier]:
    """Detect outliers in one series. Extra kwargs go to the chosen detector
    (warmup, n_bins, max_bins, k, w, a, b, ...)."""
    if mode == "online":
        return online_detect(values, rule=rule, **kwargs)
    if mode == "batch":
        return batch_detect(values, rule=rule, **kwargs)
    if mode == "expanding":
        return expanding_detect(values, rule=rule, **kwargs)
    raise ValueError(f"unknown mode {mode!r}; choose from {MODES}")


__all__ = [
    "Outlier", "outlier_score", "OnlineBoxplot", "BatchBoxplot", "medcouple",
    "detect", "online_detect", "batch_detect", "expanding_detect",
    "available_methods", "MODES",
]
