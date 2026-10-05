import json
import time

import pytest

from trading_agent.config import RiskLimits
from trading_agent.execution import Fill
from trading_agent.portfolio import Portfolio
from trading_agent.risk import CapitalGate, KillSwitch, OrderIntent, RiskGate
from trading_agent.schema import RiskState

SYM = "BTC/USDT"


@pytest.fixture
def setup(tmp_path):
    ks = KillSwitch(tmp_path)
    ks.arm()
    gate = RiskGate(RiskLimits(), ks, CapitalGate(tmp_path, mode="paper"))
    pf = Portfolio(100_000.0)
    pf.mark(SYM, 100.0)
    return gate, ks, pf


def intent(target, current=0.0, price=100.0, ts=1_000, snap_ts=1_000):
    return OrderIntent(symbol=SYM, target_qty=target, current_qty=current, price=price, ts_ms=ts, snapshot_ts_ms=snap_ts)


def test_unarmed_kill_switch_fails_closed(tmp_path):
    gate = RiskGate(RiskLimits(), KillSwitch(tmp_path), CapitalGate(tmp_path, mode="paper"))
    pf = Portfolio(100_000.0)
    pf.mark(SYM, 100.0)
    r = gate.check(intent(10), pf)
    assert not r.approved and "kill" in r.reason


def test_heartbeat_removed_fails_closed(setup, tmp_path):
    gate, ks, pf = setup
    (tmp_path / "heartbeat").unlink()
    assert not gate.check(intent(10), pf).approved


def test_tripped_allows_reduce_only(setup):
    gate, ks, pf = setup
    pf.apply_fill(Fill("a", SYM, 50.0, 100.0, 0.0, 1))
    ks.trip("test")
    assert not gate.check(intent(100, current=50), pf).approved
    r = gate.check(intent(0, current=50), pf)
    assert r.approved and r.reduce_only and r.qty == pytest.approx(-50)


def test_drawdown_trips_kill_switch(setup):
    gate, ks, pf = setup
    pf.apply_fill(Fill("a", SYM, 500.0, 100.0, 0.0, 1))
    pf.mark(SYM, 69.0)  # 500 * 31 = 15.5k loss on 100k
    assert gate.risk_state(pf) is RiskState.REDUCE
    assert not gate.check(intent(600, current=500, price=69.0), pf).approved
    assert ks.tripped()


def test_daily_loss_blocks_new_risk(setup):
    gate, ks, pf = setup
    pf.apply_fill(Fill("a", SYM, 100.0, 100.0, 0.0, 1))
    pf.mark(SYM, 68.0)  # 3.2k loss
    assert gate.risk_state(pf) is RiskState.REDUCE
    assert not gate.check(intent(200, current=100, price=68.0), pf).approved
    assert not ks.tripped()


def test_near_limit_state(setup):
    gate, ks, pf = setup
    pf.apply_fill(Fill("a", SYM, 100.0, 100.0, 0.0, 1))
    pf.mark(SYM, 75.0)  # 2.5k loss = 83% of daily limit
    assert gate.risk_state(pf) is RiskState.NEAR_LIMIT
    assert not gate.check(intent(150, current=100, price=75.0), pf).approved


def test_position_cap_clips(setup):
    gate, ks, pf = setup
    r = gate.check(intent(10_000), pf)  # 1M notional requested
    assert r.approved and r.qty == pytest.approx(250.0)  # 25% of 100k / 100


def test_stale_snapshot_rejected(setup):
    gate, ks, pf = setup
    assert not gate.check(intent(10, ts=100_000, snap_ts=1_000), pf).approved


def test_order_rate_limit(setup):
    gate, ks, pf = setup
    approved = sum(gate.check(intent(1 + i), pf).approved for i in range(30))
    assert approved == RiskLimits().max_orders_per_minute


def test_min_order_notional(setup):
    gate, ks, pf = setup
    assert not gate.check(intent(0.01), pf).approved


def test_live_requires_capital_approval(tmp_path):
    ks = KillSwitch(tmp_path)
    ks.arm()
    gate = RiskGate(RiskLimits(), ks, CapitalGate(tmp_path, mode="live"))
    pf = Portfolio(100_000.0)
    pf.mark(SYM, 100.0)
    assert not gate.check(intent(10), pf).approved
    CapitalGate.approve(tmp_path, amount=5_000.0, operator="ops", days=1)
    r = gate.check(intent(10_000), pf)
    assert r.approved and r.qty == pytest.approx(50.0)  # clipped to approved capital


def test_expired_capital_approval(tmp_path):
    CapitalGate.approve(tmp_path, amount=5_000.0, operator="ops", days=1)
    data = json.loads((tmp_path / "capital.json").read_text())
    data["expires_at"] = time.time() - 1
    (tmp_path / "capital.json").write_text(json.dumps(data))
    assert CapitalGate(tmp_path, mode="live").approved_capital() is None


def test_kill_reset_requires_operator(tmp_path):
    ks = KillSwitch(tmp_path)
    ks.arm()
    ks.trip("x")
    with pytest.raises(ValueError):
        ks.reset(operator="")
    ks.reset(operator="ricardo")
    assert not ks.tripped()
