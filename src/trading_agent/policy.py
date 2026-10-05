"""Fire condition and escalation rule. Evaluated in code, never in a prompt."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .config import PolicyConfig
from .schema import Decision, Direction, Regime, RiskState


def should_fire(d: Decision, cfg: PolicyConfig) -> bool:
    return (
        d.direction is not Direction.NEUTRAL
        and not d.toxic_flow
        and d.setup_quality >= cfg.min_setup_quality
        and d.risk_state is RiskState.SAFE
        and d.confidence > cfg.fire_confidence
    )


@dataclass
class _SymState:
    low: bool = False
    crisis: bool = False
    pending: bool = False
    last_escalation: float = float("-inf")


class EscalationPolicy:
    """Escalate to Tier 1 when confidence *falls* below threshold or regime *flips* to crisis.

    Edge-triggered (not level-triggered) so a long run of low-confidence candles costs one
    deliberation, not one per candle. A trigger suppressed by cooldown or the hourly budget
    stays pending while its condition holds.
    """

    def __init__(self, cfg: PolicyConfig) -> None:
        self.cfg = cfg
        self._state: dict[str, _SymState] = {}
        self._history: deque[float] = deque()

    def should_escalate(self, symbol: str, d: Decision, now_s: float) -> bool:
        st = self._state.setdefault(symbol, _SymState())
        low = d.confidence < self.cfg.escalate_confidence
        crisis = d.regime is Regime.CRISIS
        edge = (low and not st.low) or (crisis and not st.crisis)
        st.low, st.crisis = low, crisis

        if not (low or crisis):
            st.pending = False
            return False
        if not (edge or st.pending):
            return False

        while self._history and now_s - self._history[0] >= 3600:
            self._history.popleft()
        if now_s - st.last_escalation < self.cfg.escalation_cooldown_s or len(self._history) >= self.cfg.max_escalations_per_hour:
            st.pending = True
            return False

        st.pending = False
        st.last_escalation = now_s
        self._history.append(now_s)
        return True
