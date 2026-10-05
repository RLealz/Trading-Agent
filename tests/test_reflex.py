import time

import numpy as np

from trading_agent.config import ObservationConfig
from trading_agent.feeds import BlockClose, SyntheticFeed
from trading_agent.reflex import ReflexModel, ReflexParams, build_dataset, fit_platt, train_reflex
from trading_agent.review import brier
from trading_agent.schema import Decision, Direction, Regime, RiskState
from trading_agent.state_engine import AccountState, StateEngine


def snapshots(seed, n, sym="BTC/USDT"):
    e = StateEngine(sym, ObservationConfig())
    out = []
    for ev in SyntheticFeed([sym], block_ms=1000, seed=seed, n_blocks=n):
        if isinstance(ev, BlockClose):
            out.append(e.close_block(ev.ts, AccountState()))
        else:
            e.on_event(ev)
    return out


def test_untrained_model_is_neutral_and_typed():
    m = ReflexModel(ReflexParams(symbol="BTC/USDT"))
    v = m.judge(snapshots(0, 40)[-1])
    assert isinstance(v.decision, Decision)
    assert v.decision.direction is Direction.NEUTRAL
    assert v.p_up == 0.5


def test_cold_snapshot_is_neutral():
    m = ReflexModel(ReflexParams(symbol="BTC/USDT", weights=[1.0] * 7))
    v = m.judge(snapshots(0, 3)[0])
    assert v.decision.direction is Direction.NEUTRAL
    assert v.decision.setup_quality == 0


def test_trained_model_beats_coin_flip_out_of_sample():
    train = snapshots(1, 3000)
    test = snapshots(2, 1500)
    params = train_reflex("BTC/USDT", *build_dataset(train, horizon=30))
    m = ReflexModel(params)
    Xs, ys = build_dataset(test, horizon=30)
    p = np.array([m.p_up(s) for s in Xs])
    assert brier(p, ys) < brier(np.full_like(p, ys.mean()), ys)


def test_confidence_is_directional_probability():
    train = snapshots(1, 2000)
    m = ReflexModel(train_reflex("BTC/USDT", *build_dataset(train, horizon=30)))
    for s in snapshots(3, 200)[40:]:
        v = m.judge(s)
        assert abs(v.decision.confidence - max(v.p_up, 1 - v.p_up)) < 1e-9


def test_platt_corrects_overconfidence():
    rng = np.random.default_rng(0)
    true_logit = rng.normal(0, 1, 5000)
    y = (rng.random(5000) < 1 / (1 + np.exp(-true_logit))).astype(float)
    a, b = fit_platt(true_logit * 4.0, y)  # model is 4x overconfident
    assert 0.15 < a < 0.4


def test_latency_double_digit_ms_budget():
    train = snapshots(1, 1500)
    m = ReflexModel(train_reflex("BTC/USDT", *build_dataset(train, horizon=30)))
    s = train[-1]
    t0 = time.perf_counter()
    for _ in range(1000):
        m.judge(s)
    assert (time.perf_counter() - t0) / 1000 < 0.01


def test_crisis_regime_tightens_risk_state():
    p = ReflexParams(symbol="BTC/USDT", weights=[1.0] * 7, crisis_rvol_bps=0.0)
    v = ReflexModel(p).judge(snapshots(0, 40)[-1])
    assert v.decision.regime is Regime.CRISIS
    assert v.decision.risk_state is RiskState.REDUCE


def test_disabled_regime_zeroes_setup_quality():
    train = snapshots(1, 1500)
    params = train_reflex("BTC/USDT", *build_dataset(train, horizon=30))
    params.disabled_regimes = [r.value for r in Regime]
    m = ReflexModel(params)
    assert all(m.judge(s).decision.setup_quality == 0 for s in train[40:200])


def test_params_roundtrip():
    p = ReflexParams(symbol="X", weights=[0.1] * 7, version=3, disabled_regimes=["crisis"])
    assert ReflexParams.from_dict(p.to_dict()) == p
