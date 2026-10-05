import pytest

from trading_agent.config import AppConfig
from trading_agent.schema import Decision, Direction, Regime, RiskState


@pytest.fixture
def cfg(tmp_path):
    return AppConfig(state_dir=tmp_path / "state", symbols=("BTC/USDT",))


def make_decision(**kw):
    base = dict(
        regime=Regime.TRENDING,
        direction=Direction.LONG,
        toxic_flow=False,
        setup_quality=3,
        risk_state=RiskState.SAFE,
        confidence=0.9,
    )
    base.update(kw)
    return Decision(**base)


@pytest.fixture
def decision_factory():
    return make_decision
