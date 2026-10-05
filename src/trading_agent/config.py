"""Configuration. Hard limits are validated so a config file can only tighten them."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    pass


# Ceilings from the operating contract. Config may tighten, never loosen.
MAX_DRAWDOWN_CEILING = 0.15
KELLY_CEILING = 0.25
FIRE_CONFIDENCE_FLOOR = 0.80
SETUP_QUALITY_FLOOR = 2


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ConfigError(msg)


@dataclass(frozen=True)
class RiskLimits:
    max_drawdown: float = 0.15
    max_daily_loss: float = 0.03
    max_position_frac: float = 0.25
    max_gross_frac: float = 0.50
    near_limit_ratio: float = 0.80
    max_snapshot_age_ms: int = 5_000
    max_orders_per_minute: int = 20
    min_order_notional: float = 10.0

    def __post_init__(self) -> None:
        _require(0 < self.max_drawdown <= MAX_DRAWDOWN_CEILING, f"max_drawdown must be in (0, {MAX_DRAWDOWN_CEILING}]")
        _require(0 < self.max_daily_loss < self.max_drawdown, "max_daily_loss must be in (0, max_drawdown)")
        _require(0 < self.max_position_frac <= 1.0, "max_position_frac must be in (0, 1]")
        _require(self.max_position_frac <= self.max_gross_frac <= 1.0, "max_gross_frac must be in [max_position_frac, 1]")
        _require(0 < self.near_limit_ratio < 1, "near_limit_ratio must be in (0, 1)")
        _require(self.max_snapshot_age_ms > 0 and self.max_orders_per_minute > 0, "rate/staleness limits must be positive")


@dataclass(frozen=True)
class PolicyConfig:
    min_setup_quality: int = 2
    fire_confidence: float = 0.80
    escalate_confidence: float = 0.60
    escalation_cooldown_s: float = 300.0
    max_escalations_per_hour: int = 6
    tier1_ttl_blocks: int = 15
    holding_blocks: int = 30
    allow_short: bool = True          # false on spot venues

    def __post_init__(self) -> None:
        _require(SETUP_QUALITY_FLOOR <= self.min_setup_quality <= 3, "min_setup_quality must be in [2, 3]")
        _require(FIRE_CONFIDENCE_FLOOR <= self.fire_confidence < 1, "fire_confidence must be in [0.80, 1)")
        _require(0.5 <= self.escalate_confidence <= self.fire_confidence, "escalate_confidence must be in [0.5, fire_confidence]")
        _require(self.tier1_ttl_blocks > 0 and self.holding_blocks > 0, "ttl/holding must be positive")


@dataclass(frozen=True)
class SizingConfig:
    kelly_multiplier: float = 0.25
    default_payoff_ratio: float = 1.0
    rebalance_band: float = 0.10

    def __post_init__(self) -> None:
        _require(0 < self.kelly_multiplier <= KELLY_CEILING, f"kelly_multiplier must be in (0, {KELLY_CEILING}]")
        _require(self.default_payoff_ratio > 0, "default_payoff_ratio must be positive")


@dataclass(frozen=True)
class ObservationConfig:
    short_window: int = 5
    long_window: int = 30
    depth_levels: int = 5
    max_tokens: int = 400


@dataclass(frozen=True)
class ExecutionConfig:
    fee_bps: float = 5.0
    slippage_bps: float = 2.0


@dataclass(frozen=True)
class Tier1Config:
    model: str = "claude-opus-5-5"
    effort: str = "high"
    timeout_s: float = 600.0


@dataclass(frozen=True)
class ReviewConfig:
    label_horizon_blocks: int = 30
    history_days: int = 14
    min_samples: int = 200
    oos_fraction: float = 0.3


@dataclass(frozen=True)
class AppConfig:
    mode: str = "paper"
    state_dir: Path = Path("state")
    symbols: tuple[str, ...] = ("BTC/USDT", "ETH/USDT")
    block_ms: int = 60_000
    starting_equity: float = 100_000.0
    risk: RiskLimits = field(default_factory=RiskLimits)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    sizing: SizingConfig = field(default_factory=SizingConfig)
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    tier1: Tier1Config = field(default_factory=Tier1Config)
    review: ReviewConfig = field(default_factory=ReviewConfig)

    def __post_init__(self) -> None:
        _require(self.mode in ("paper", "live"), "mode must be paper or live")
        object.__setattr__(self, "state_dir", Path(self.state_dir))
        object.__setattr__(self, "symbols", tuple(self.symbols))


_SECTIONS = {
    "risk": RiskLimits,
    "policy": PolicyConfig,
    "sizing": SizingConfig,
    "observation": ObservationConfig,
    "execution": ExecutionConfig,
    "tier1": Tier1Config,
    "review": ReviewConfig,
}


def _build(cls: type, data: dict[str, Any]) -> Any:
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    _require(not unknown, f"unknown keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**data)


def load_config(path: str | Path) -> AppConfig:
    raw = tomllib.loads(Path(path).read_text())
    kwargs: dict[str, Any] = {}
    for key, value in raw.items():
        if key in _SECTIONS:
            kwargs[key] = _build(_SECTIONS[key], value)
        else:
            kwargs[key] = value
    return _build(AppConfig, kwargs)
