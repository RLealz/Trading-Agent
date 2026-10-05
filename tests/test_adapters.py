"""ccxt adapters against a fake client (no network)."""

import itertools
import time

from trading_agent.config import ObservationConfig
from trading_agent.execution import CcxtExchange, Order
from trading_agent.feeds import BlockClose, CcxtPollingFeed
from trading_agent.state_engine import AccountState, BookUpdate, StateEngine, Trade


class FakeCcxt:
    def __init__(self):
        self.orders = []
        self._t = itertools.count(1)

    def fetch_order_book(self, symbol, limit=5):
        return {"bids": [[100.0, 1.0, 0], [99.9, 2.0, 0]], "asks": [[100.1, 1.5, 0]]}

    def fetch_trades(self, symbol, since=None):
        # venue timestamps deliberately far in the past: must not leak into causal stamping
        n = next(self._t)
        return [{"timestamp": n, "price": 100.0, "amount": 0.5, "side": "buy" if n % 2 else "sell"}]

    def create_order(self, symbol, type_, side, amount, price, params):
        self.orders.append((symbol, type_, side, amount, params))
        if amount > 100:
            raise RuntimeError("insufficient balance")
        return {"filled": amount, "average": 100.05, "fee": {"cost": 0.01}}


def test_polling_feed_is_causal_and_closes_blocks():
    feed = CcxtPollingFeed(FakeCcxt(), ["BTC/USDT"], block_ms=60, poll_s=0.005)
    eng = StateEngine("BTC/USDT", ObservationConfig())
    closes, events = 0, 0
    deadline = time.time() + 5
    for ev in feed:
        if isinstance(ev, BlockClose):
            if eng.last_mid is not None:
                eng.close_block(ev.ts, AccountState())  # raises CausalityError on any ordering bug
            closes += 1
        else:
            assert isinstance(ev, (BookUpdate, Trade))
            eng.on_event(ev)
            events += 1
        if closes >= 3 or time.time() > deadline:
            break
    assert closes >= 3 and events > 0


def test_exchange_maps_fills_and_swallows_venue_errors():
    c = FakeCcxt()
    ex = CcxtExchange(c)
    f = ex.place(Order("id1", "BTC/USDT", -2.0, True, 1), 100.0, 1.0)
    assert f.qty == -2.0 and f.price == 100.05
    assert c.orders[0][2] == "sell" and c.orders[0][4] == {"clientOrderId": "id1", "reduceOnly": True}
    assert ex.place(Order("id2", "BTC/USDT", 500.0, False, 2), 100.0, 1.0) is None
