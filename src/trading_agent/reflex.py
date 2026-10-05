"""Tier 2 reflex: a small calibrated classifier.

Pure function of (params, snapshot) -> Verdict. No memory, no tools, no I/O.
Standardised features -> logistic score -> Platt calibration -> typed Decision.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Sequence

import numpy as np

from .schema import Decision, Direction, Regime, RiskState
from .state_engine import Snapshot

FEATURE_NAMES = (
    "imbalance",
    "microprice_bps",
    "ret_short_bps",
    "ret_long_bps",
    "rvol_bps",
    "flow_imbalance",
    "spread_bps",
)
N_FEATURES = len(FEATURE_NAMES)


@dataclass
class ReflexParams:
    symbol: str
    version: int = 0
    weights: list[float] = field(default_factory=lambda: [0.0] * N_FEATURES)
    bias: float = 0.0
    mean: list[float] = field(default_factory=lambda: [0.0] * N_FEATURES)
    std: list[float] = field(default_factory=lambda: [1.0] * N_FEATURES)
    platt_a: float = 1.0
    platt_b: float = 0.0
    payoff_ratio: float = 1.0            # estimate; refit nightly from the tape
    tier1_shrink: float = 0.5            # unproven Tier 1 confidence is pulled toward 0.5
    direction_margin: float = 0.02
    crisis_rvol_bps: float = 80.0
    high_vol_rvol_bps: float = 30.0
    trend_z: float = 1.0
    trend_window: int = 30
    signal_floor: float = 0.1            # min |imbalance| / |flow| to count as confirmation
    toxic_spread_bps: float = 15.0
    toxic_flow: float = 0.6
    disabled_regimes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ReflexParams":
        return cls(**d)


@dataclass(frozen=True)
class Verdict:
    decision: Decision
    p_up: float
    score: float  # raw (pre-calibration) logit, journaled for recalibration


def features(s: Snapshot) -> np.ndarray:
    return np.array([getattr(s, n) for n in FEATURE_NAMES], dtype=float)


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def classify_regime(s: Snapshot, p: ReflexParams) -> Regime:
    if s.rvol_bps >= p.crisis_rvol_bps:
        return Regime.CRISIS
    if s.rvol_bps >= p.high_vol_rvol_bps:
        return Regime.HIGH_VOL
    # trend z-score: long-window move relative to the noise expected over that window
    noise = max(s.rvol_bps, 1e-9) * math.sqrt(p.trend_window)
    if abs(s.ret_long_bps) / noise >= p.trend_z:
        return Regime.TRENDING
    return Regime.MEAN_REVERTING


class ReflexModel:
    def __init__(self, params: ReflexParams) -> None:
        self.params = params
        self._w = np.asarray(params.weights, dtype=float)
        self._mu = np.asarray(params.mean, dtype=float)
        self._sd = np.where(np.asarray(params.std, dtype=float) > 0, params.std, 1.0)
        self._trained = bool(np.any(self._w != 0))

    def score(self, s: Snapshot) -> float:
        return float(np.dot(self._w, (features(s) - self._mu) / self._sd) + self.params.bias)

    def calibrate(self, score: float) -> float:
        return _sigmoid(self.params.platt_a * score + self.params.platt_b)

    def p_up(self, s: Snapshot) -> float:
        return self.calibrate(self.score(s)) if self._trained else 0.5

    def judge(self, s: Snapshot) -> Verdict:
        p = self.params
        regime = classify_regime(s, p)
        risk = {Regime.CRISIS: RiskState.REDUCE, Regime.HIGH_VOL: RiskState.NEAR_LIMIT}.get(regime, RiskState.SAFE)

        if not (self._trained and s.warm):
            d = Decision(regime=regime, direction=Direction.NEUTRAL, toxic_flow=False,
                         setup_quality=0, risk_state=risk, confidence=0.5)
            return Verdict(d, 0.5, 0.0)

        score = self.score(s)
        p_up = self.calibrate(score)
        conf = max(p_up, 1.0 - p_up)
        if p_up > 0.5 + p.direction_margin:
            sign, direction = 1, Direction.LONG
        elif p_up < 0.5 - p.direction_margin:
            sign, direction = -1, Direction.SHORT
        else:
            sign, direction = 0, Direction.NEUTRAL

        quality = 0
        toxic = s.spread_bps >= p.toxic_spread_bps
        if sign:
            if s.flow_imbalance * sign >= p.signal_floor:
                quality += 1
            if s.imbalance * sign >= p.signal_floor:
                quality += 1
            if (regime is Regime.TRENDING and s.ret_long_bps * sign > 0) or (
                regime is Regime.MEAN_REVERTING and s.ret_short_bps * sign < 0
            ):
                quality += 1
            # informed one-sided aggression against our side
            toxic = toxic or (s.flow_imbalance * sign <= -p.toxic_flow)
        if regime.value in p.disabled_regimes:
            quality = 0

        d = Decision(regime=regime, direction=direction, toxic_flow=toxic,
                     setup_quality=min(quality, 3), risk_state=risk, confidence=conf)
        return Verdict(d, p_up, score)


# ---------------------------------------------------------------- training

def build_dataset(snaps: Sequence[Snapshot], horizon: int) -> tuple[list[Snapshot], np.ndarray]:
    """Warm snapshots paired with the sign of the mid move `horizon` blocks later.

    Labels use the future by construction; they exist only offline and never enter a snapshot.
    """
    xs, ys = [], []
    for i in range(len(snaps) - horizon):
        s = snaps[i]
        if not s.warm:
            continue
        xs.append(s)
        ys.append(1.0 if snaps[i + horizon].mid > s.mid else 0.0)
    return xs, np.asarray(ys, dtype=float)


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1e-2, iters: int = 50) -> tuple[np.ndarray, float]:
    """L2-regularised logistic regression via Newton-Raphson. Returns (weights, bias)."""
    n, k = X.shape
    Xb = np.hstack([X, np.ones((n, 1))])
    beta = np.zeros(k + 1)
    reg = np.full(k + 1, l2)
    reg[-1] = 0.0
    for _ in range(iters):
        z = np.clip(Xb @ beta, -30, 30)
        p = 1 / (1 + np.exp(-z))
        g = Xb.T @ (p - y) / n + reg * beta
        H = (Xb * (p * (1 - p))[:, None]).T @ Xb / n + np.diag(reg) + 1e-9 * np.eye(k + 1)
        step = np.linalg.solve(H, g)
        beta -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return beta[:-1], float(beta[-1])


def fit_platt(scores: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    w, b = fit_logistic(np.asarray(scores, dtype=float).reshape(-1, 1), np.asarray(y, dtype=float), l2=0.0)
    return float(w[0]), b


def train_reflex(
    symbol: str,
    snaps: Sequence[Snapshot],
    y: np.ndarray,
    holdout: float = 0.3,
    l2: float = 1e-2,
    base: ReflexParams | None = None,
) -> ReflexParams:
    """Fit on the first (1-holdout), Platt-calibrate on the chronological holdout."""
    if len(snaps) < 50:
        raise ValueError("not enough samples to train")
    X = np.vstack([features(s) for s in snaps])
    cut = int(len(X) * (1 - holdout))
    mu = X[:cut].mean(axis=0)
    sd = X[:cut].std(axis=0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    w, b = fit_logistic(Z[:cut], y[:cut], l2=l2)
    a, c = fit_platt(Z[cut:] @ w + b, y[cut:])
    params = ReflexParams(**(base.to_dict() if base else {"symbol": symbol}))
    params.symbol = symbol
    params.weights = [float(v) for v in w]
    params.bias = b
    params.mean = [float(v) for v in mu]
    params.std = [float(v) for v in sd]
    params.platt_a, params.platt_b = a, c
    return params
