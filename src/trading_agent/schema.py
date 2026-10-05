"""The decision contract: one typed verdict per candle."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class Regime(str, Enum):
    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"
    HIGH_VOL = "high_vol"
    CRISIS = "crisis"


class Direction(str, Enum):
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


class RiskState(str, Enum):
    SAFE = "safe"
    NEAR_LIMIT = "near_limit"
    REDUCE = "reduce"


_RISK_RANK = {RiskState.SAFE: 0, RiskState.NEAR_LIMIT: 1, RiskState.REDUCE: 2}


def worst(*states: RiskState) -> RiskState:
    """Most restrictive risk state. Used so a model can tighten risk, never loosen it."""
    return max(states, key=_RISK_RANK.__getitem__)


class Decision(BaseModel):
    """Typed verdict. Carries no size, price or threshold: those live in code."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    regime: Regime
    direction: Direction
    toxic_flow: bool
    setup_quality: int = Field(ge=0, le=3)
    risk_state: RiskState
    confidence: float = Field(ge=0.0, le=1.0)

    @property
    def sign(self) -> int:
        return {Direction.LONG: 1, Direction.SHORT: -1, Direction.NEUTRAL: 0}[self.direction]
