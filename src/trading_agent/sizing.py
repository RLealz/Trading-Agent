"""Fractional Kelly off the calibrated probability, hard-capped at one quarter Kelly."""

from __future__ import annotations

import math

QUARTER_KELLY_CAP = 0.25


def kelly(p: float, b: float) -> float:
    """Kelly fraction for a binary bet: win prob p, payoff ratio b (avg win / avg loss). Never negative."""
    if not (math.isfinite(p) and math.isfinite(b)) or b <= 0 or not 0 <= p <= 1:
        return 0.0
    return max(p - (1 - p) / b, 0.0)


def position_fraction(p: float, b: float, multiplier: float) -> float:
    """Fraction of equity to deploy. The 0.25 cap holds whatever the config says."""
    m = min(max(multiplier, 0.0), QUARTER_KELLY_CAP)
    return m * kelly(p, b)
