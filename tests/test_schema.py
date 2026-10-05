import pytest
from pydantic import ValidationError

from trading_agent.schema import Decision, Direction, Regime, RiskState, worst


def test_decision_bounds(decision_factory):
    with pytest.raises(ValidationError):
        decision_factory(setup_quality=4)
    with pytest.raises(ValidationError):
        decision_factory(confidence=1.2)
    with pytest.raises(ValidationError):
        Decision(regime="trending", direction="long", toxic_flow=False, setup_quality=2,
                 risk_state="safe", confidence=0.9, size=10)  # no extra fields: model never holds a size


def test_decision_frozen(decision_factory):
    d = decision_factory()
    with pytest.raises(ValidationError):
        d.confidence = 0.1


def test_worst_only_tightens():
    assert worst(RiskState.SAFE, RiskState.NEAR_LIMIT) is RiskState.NEAR_LIMIT
    assert worst(RiskState.REDUCE, RiskState.SAFE) is RiskState.REDUCE
    assert worst(RiskState.SAFE, RiskState.SAFE) is RiskState.SAFE


def test_enum_values_match_contract():
    assert {r.value for r in Regime} == {"trending", "mean_reverting", "high_vol", "crisis"}
    assert {d.value for d in Direction} == {"long", "short", "neutral"}
    assert {r.value for r in RiskState} == {"safe", "near_limit", "reduce"}
