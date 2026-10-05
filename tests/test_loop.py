import concurrent.futures

import pytest

from trading_agent.config import AppConfig, PolicyConfig
from trading_agent.execution import PaperExchange
from trading_agent.feeds import SyntheticFeed
from trading_agent.journal import Journal
from trading_agent.loop import TradingLoop
from trading_agent.portfolio import Portfolio
from trading_agent.reflex import ReflexParams, Verdict
from trading_agent.risk import CapitalGate, KillSwitch, RiskGate
from trading_agent.schema import Direction, RiskState


class SyncExecutor(concurrent.futures.Executor):
    def submit(self, fn, *a, **kw):
        f = concurrent.futures.Future()
        f.set_result(fn(*a, **kw))
        return f


class FixedReflex:
    def __init__(self, decision, p_up=0.9):
        self.decision, self.p_up_ = decision, p_up
        self.params = ReflexParams(symbol="BTC/USDT")

    def judge(self, s):
        return Verdict(self.decision, self.p_up_, 0.0)


class RecordingDeliberator:
    def __init__(self, decision):
        self.decision, self.calls = decision, []

    def deliberate(self, snapshot, history, reflex, thesis=None):
        self.calls.append(snapshot.ts)
        return self.decision


def build(cfg, reflex, deliberator=None):
    ks = KillSwitch(cfg.state_dir)
    ks.arm()
    gate = RiskGate(cfg.risk, ks, CapitalGate(cfg.state_dir, mode=cfg.mode))
    return TradingLoop(
        cfg,
        exchange=PaperExchange(cfg.execution),
        portfolio=Portfolio(cfg.starting_equity),
        gate=gate,
        models={s: reflex for s in cfg.symbols},
        journal=Journal(cfg.state_dir),
        deliberator=deliberator,
        executor=SyncExecutor(),
    )


def test_every_candle_journaled_fills_and_misses(cfg, decision_factory):
    loop = build(cfg, FixedReflex(decision_factory(confidence=0.5, direction=Direction.NEUTRAL)))
    loop.run(SyntheticFeed(list(cfg.symbols), cfg.block_ms, seed=1, n_blocks=50))
    recs = loop.journal.read_all()
    assert len(recs) == 50
    assert all(not r["fired"] for r in recs)
    assert all("p_up" in r and "snapshot" in r for r in recs)


def test_fires_and_sizes_in_code(cfg, decision_factory):
    loop = build(cfg, FixedReflex(decision_factory()))
    loop.run(SyntheticFeed(list(cfg.symbols), cfg.block_ms, seed=1, n_blocks=40))
    recs = loop.journal.read_all()
    fired = [r for r in recs if r["fired"] and r["fill"]]
    assert fired
    first = fired[0]
    # quarter Kelly at p=0.9, b=1 -> 0.2 of equity, then clipped by 25% position cap -> 0.2
    notional = abs(first["fill"]["qty"] * first["fill"]["price"])
    assert notional == pytest.approx(0.2 * cfg.starting_equity, rel=0.02)


def test_model_cannot_loosen_risk_state(cfg, decision_factory):
    loop = build(cfg, FixedReflex(decision_factory(risk_state=RiskState.SAFE)))
    loop.portfolio.day_start_equity = cfg.starting_equity * 1.04  # 4/104 = 3.8% loss: past the 3% daily limit
    loop.run(SyntheticFeed(list(cfg.symbols), cfg.block_ms, seed=1, n_blocks=40))
    recs = loop.journal.read_all()
    assert all(r["final_risk_state"] == "reduce" for r in recs)
    assert not any(r["fill"] for r in recs)


def test_escalation_is_async_and_causal(cfg, decision_factory):
    cfg2 = AppConfig(state_dir=cfg.state_dir, symbols=cfg.symbols,
                     policy=PolicyConfig(escalation_cooldown_s=0, max_escalations_per_hour=100))
    t1 = decision_factory(direction=Direction.SHORT, confidence=0.85)
    delib = RecordingDeliberator(t1)
    loop = build(cfg2, FixedReflex(decision_factory(confidence=0.55)), delib)
    loop.run(SyntheticFeed(list(cfg2.symbols), cfg2.block_ms, seed=1, n_blocks=40))
    recs = loop.journal.read_all()
    assert len(delib.calls) == 1
    esc = next(r for r in recs if r["escalated"])
    assert esc["source"] == "reflex"  # escalating block trades on the cheap verdict
    later = [r for r in recs if r["ts"] > esc["ts"]][: cfg2.policy.tier1_ttl_blocks]
    assert all(r["source"] == "tier1" for r in later)
    expired = [r for r in recs if r["ts"] > esc["ts"]][cfg2.policy.tier1_ttl_blocks:]
    assert all(r["source"] == "reflex" for r in expired)


def test_tier1_failure_means_no_trade(cfg, decision_factory):
    class Broken:
        def deliberate(self, *a, **k):
            return None

    cfg2 = AppConfig(state_dir=cfg.state_dir, symbols=cfg.symbols,
                     policy=PolicyConfig(escalation_cooldown_s=0, max_escalations_per_hour=100))
    loop = build(cfg2, FixedReflex(decision_factory(confidence=0.55)), Broken())
    loop.run(SyntheticFeed(list(cfg2.symbols), cfg2.block_ms, seed=1, n_blocks=40))
    assert not any(r["fill"] for r in loop.journal.read_all())


def test_kill_switch_trip_flattens(cfg, decision_factory):
    loop = build(cfg, FixedReflex(decision_factory()))
    feed = iter(SyntheticFeed(list(cfg.symbols), cfg.block_ms, seed=1, n_blocks=80))
    loop.run(feed, max_blocks=40)
    assert loop.portfolio.position("BTC/USDT") != 0
    loop.gate.kill.trip("operator")
    loop.run(feed, max_blocks=5)
    assert loop.portfolio.position("BTC/USDT") == 0
