import pytest

from trading_agent.config import AppConfig, ConfigError, RiskLimits, PolicyConfig, SizingConfig, load_config


def test_default_config_loads(tmp_path):
    c = load_config("config/default.toml")
    assert c.risk.max_drawdown == 0.15
    assert c.policy.fire_confidence == 0.80
    assert c.policy.escalate_confidence == 0.60


@pytest.mark.parametrize(
    "factory",
    [
        lambda: RiskLimits(max_drawdown=0.30),
        lambda: RiskLimits(max_position_frac=1.5),
        lambda: PolicyConfig(fire_confidence=0.70),
        lambda: PolicyConfig(min_setup_quality=1),
        lambda: SizingConfig(kelly_multiplier=0.5),
    ],
)
def test_config_cannot_loosen_hard_limits(factory):
    with pytest.raises(ConfigError):
        factory()


def test_live_mode_is_explicit():
    assert AppConfig().mode == "paper"
