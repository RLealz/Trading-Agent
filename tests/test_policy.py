import itertools

from trading_agent.config import PolicyConfig
from trading_agent.policy import EscalationPolicy, should_fire
from trading_agent.schema import Direction, Regime, RiskState


def test_fire_truth_table(decision_factory):
    cfg = PolicyConfig()
    for q, rs, conf in itertools.product(range(4), RiskState, [0.5, 0.8, 0.80001, 0.95]):
        d = decision_factory(setup_quality=q, risk_state=rs, confidence=conf)
        expected = q >= 2 and rs is RiskState.SAFE and conf > 0.80
        assert should_fire(d, cfg) is expected, (q, rs, conf)


def test_never_fire_neutral_or_toxic(decision_factory):
    cfg = PolicyConfig()
    assert not should_fire(decision_factory(direction=Direction.NEUTRAL), cfg)
    assert not should_fire(decision_factory(toxic_flow=True), cfg)


def test_escalates_on_transition_not_every_candle(decision_factory):
    pol = EscalationPolicy(PolicyConfig(escalation_cooldown_s=0, max_escalations_per_hour=100))
    hi, lo = decision_factory(confidence=0.9), decision_factory(confidence=0.55)
    assert not pol.should_escalate("X", hi, 0)
    assert pol.should_escalate("X", lo, 1)
    assert not pol.should_escalate("X", lo, 2)  # still low, already escalated
    assert not pol.should_escalate("X", hi, 3)
    assert pol.should_escalate("X", lo, 4)


def test_escalates_on_crisis_flip(decision_factory):
    pol = EscalationPolicy(PolicyConfig(escalation_cooldown_s=0, max_escalations_per_hour=100))
    assert not pol.should_escalate("X", decision_factory(), 0)
    assert pol.should_escalate("X", decision_factory(regime=Regime.CRISIS), 1)


def test_escalation_budget_and_pending(decision_factory):
    pol = EscalationPolicy(PolicyConfig(escalation_cooldown_s=60, max_escalations_per_hour=100))
    lo, hi = decision_factory(confidence=0.55), decision_factory(confidence=0.9)
    assert pol.should_escalate("X", lo, 0)
    pol.should_escalate("X", hi, 10)
    assert not pol.should_escalate("X", lo, 20)   # cooldown -> pending
    assert pol.should_escalate("X", lo, 61)       # pending fires once cooldown clears

    pol2 = EscalationPolicy(PolicyConfig(escalation_cooldown_s=0, max_escalations_per_hour=2))
    fired = 0
    for t in range(0, 40, 2):
        fired += pol2.should_escalate(f"S{t}", lo, t)
    assert fired == 2
