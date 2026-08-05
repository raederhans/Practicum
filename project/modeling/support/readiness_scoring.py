"""Dependency-light scoring helpers shared by readiness generation and tests."""

from __future__ import annotations

import math


def _finite_number(value: object, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def score_observation(observed_rate: object, high_censoring_share: object) -> int:
    """Return the v1 observation-quality points without treating zero as missing."""

    observed = _finite_number(observed_rate, 0.0)
    censoring = _finite_number(high_censoring_share, 1.0)
    if observed >= 0.99 and censoring <= 0.01:
        return 30
    if observed >= 0.97:
        return 24
    if observed >= 0.95:
        return 18
    return 0
