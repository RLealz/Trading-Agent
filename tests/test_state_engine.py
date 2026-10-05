import dataclasses

import pytest

from trading_agent.config import ObservationConfig
from trading_agent.feeds import BlockClose, SyntheticFeed
from trading_agent.state_engine import AccountState, BookUpdate, CausalityError, StateEngine, Trade

SYM = "BTC/USDT"


def book(ts, bid=100.0, ask=100.1, bsz=1.0, asz=1.0):
    return BookUpdate(ts=ts, symbol=SYM, bids=((bid, bsz),), asks=((ask, asz),))


def test_basic_fields():
    e = StateEngine(SYM, ObservationConfig())
    e.on_event(book(1, 100.0, 100.2, bsz=3.0, asz=1.0))
    e.on_event(Trade(ts=2, symbol=SYM, price=100.2, size=2.0, aggressor="buy"))
    s = e.close_block(10, AccountState())
    assert s.mid == pytest.approx(100.1)
    assert s.spread_bps == pytest.approx(0.2 / 100.1 * 1e4)
    assert s.imbalance == pytest.approx(0.5)
    assert s.flow_imbalance == pytest.approx(1.0)
    assert s.ts == 10 and s.block == 0


def test_out_of_order_event_rejected():
    e = StateEngine(SYM, ObservationConfig())
    e.on_event(book(5))
    with pytest.raises(CausalityError):
        e.on_event(book(4))


def test_block_cannot_close_before_last_event():
    e = StateEngine(SYM, ObservationConfig())
    e.on_event(book(50))
    with pytest.raises(CausalityError):
        e.close_block(40, AccountState())


def test_no_book_no_snapshot():
    e = StateEngine(SYM, ObservationConfig())
    with pytest.raises(CausalityError):
        e.close_block(1, AccountState())


def _run(events, symbol):
    e = StateEngine(symbol, ObservationConfig())
    out = []
    for ev in events:
        if isinstance(ev, BlockClose):
            out.append(e.close_block(ev.ts, AccountState()))
        elif ev.symbol == symbol:
            e.on_event(ev)
    return out


def test_prefix_invariance_no_lookahead():
    """A snapshot at block k must be identical whether or not future events exist."""
    events = list(SyntheticFeed(["BTC/USDT"], block_ms=1000, seed=7, n_blocks=80))
    full = _run(events, "BTC/USDT")
    cut = next(i for i, ev in enumerate(events) if isinstance(ev, BlockClose) and ev.ts == full[40].ts)
    prefix = _run(events[: cut + 1], "BTC/USDT")
    assert prefix[-1] == full[40]


def test_snapshot_has_no_future_fields():
    names = {f.name for f in dataclasses.fields(StateEngine(SYM, ObservationConfig()).snapshot_type)}
    assert not any(n.startswith(("label", "future", "fwd", "y_")) for n in names)


def test_token_budget():
    events = list(SyntheticFeed(["BTC/USDT"], block_ms=1000, seed=1, n_blocks=60))
    for s in _run(events, "BTC/USDT"):
        assert s.token_estimate() < 400
        assert s.render()


def test_warmup_flag():
    events = list(SyntheticFeed(["BTC/USDT"], block_ms=1000, seed=1, n_blocks=40))
    snaps = _run(events, "BTC/USDT")
    assert not snaps[0].warm
    assert snaps[-1].warm
