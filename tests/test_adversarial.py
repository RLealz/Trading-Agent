"""Regression tests for findings from the adversarial review pass (docs/WHERE_THIS_BREAKS.md)."""

import numpy as np

from trading_agent.config import AppConfig, PolicyConfig
from trading_agent.execution import Fill
from trading_agent.feeds import SyntheticFeed
from trading_agent.portfolio import Portfolio
from trading_agent.reflex import ReflexParams
from trading_agent.review import label_records, paired_improvement_t
from trading_agent.schema import Direction
from trading_agent.schema_store import SchemaStore
from tests.test_loop import FixedReflex, build

SYM = "BTC/USDT"


def test_restart_does_not_reset_drawdown_clock(tmp_path):
    pf = Portfolio(100_000.0)
    pf.apply_fill(Fill("a", SYM, 100.0, 100.0, 0.0, 1))
    pf.mark(SYM, 150.0)
    pf.mark(SYM, 120.0)
    pf.save(tmp_path / "pf.json")
    restored = Portfolio.load(tmp_path / "pf.json", 100_000.0)
    assert restored.peak_equity == pf.peak_equity
    assert restored.drawdown() == pf.drawdown() > 0


def test_flat_book_past_max_drawdown_still_trips(cfg, decision_factory):
    loop = build(cfg, FixedReflex(decision_factory(direction=Direction.NEUTRAL, confidence=0.5)))
    loop.portfolio.peak_equity = cfg.starting_equity / 0.80  # 20% under peak, no position
    loop.run(SyntheticFeed(list(cfg.symbols), cfg.block_ms, seed=1, n_blocks=2))
    assert loop.gate.kill.tripped()


def test_no_shorts_on_spot(cfg, decision_factory):
    spot = AppConfig(state_dir=cfg.state_dir, symbols=cfg.symbols, policy=PolicyConfig(allow_short=False))
    loop = build(spot, FixedReflex(decision_factory(direction=Direction.SHORT)))
    loop.run(SyntheticFeed(list(spot.symbols), spot.block_ms, seed=1, n_blocks=20))
    assert not any(r["fill"] for r in loop.journal.read_all())


def test_promoted_schema_picked_up_next_block(cfg, decision_factory):
    store = SchemaStore(cfg.state_dir)
    loop = build(cfg, FixedReflex(decision_factory()))
    loop.store = store
    feed = iter(SyntheticFeed(list(cfg.symbols), cfg.block_ms, seed=1, n_blocks=10))
    loop.run(feed, max_blocks=2)
    store.save(ReflexParams(symbol=SYM))
    loop.run(feed, max_blocks=1)
    assert loop.models[SYM].params.version == 1


def test_labels_skip_gaps():
    recs = [{"symbol": SYM, "block": i, "ts": t, "snapshot": {"mid": float(i + 1)}}
            for i, t in enumerate([0, 60_000, 10_000_000, 10_060_000])]
    assert len(label_records(recs, horizon=1, block_ms=60_000)) == 2


def test_noise_is_not_improvement():
    rng = np.random.default_rng(0)
    y = (rng.random(500) < 0.5).astype(float)
    p = np.full(500, 0.5)
    assert paired_improvement_t(p, p + rng.normal(0, 0.01, 500), y) < 2.33
    assert paired_improvement_t(np.full(500, 0.9), p, y) > 2.33
