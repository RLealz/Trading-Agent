"""Order/fill types and venues. Sizes arrive here already decided and gated in code."""

from __future__ import annotations

import itertools
import logging
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from .config import ExecutionConfig

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Order:
    client_id: str
    symbol: str
    qty: float          # signed: + buy, - sell
    reduce_only: bool
    ts_ms: int


@dataclass(frozen=True)
class Fill:
    client_id: str
    symbol: str
    qty: float
    price: float
    fee: float
    ts_ms: int

    def to_dict(self) -> dict:
        return asdict(self)


class Exchange(Protocol):
    def place(self, order: Order, mid: float, spread_bps: float) -> Fill | None: ...


_ids = itertools.count()


def new_client_id(symbol: str, ts_ms: int) -> str:
    return f"ta-{symbol.replace('/', '')}-{ts_ms}-{next(_ids)}"


class PaperExchange:
    """Crosses half the spread plus slippage, pays taker fees. Deliberately pessimistic."""

    def __init__(self, cfg: ExecutionConfig) -> None:
        self.cfg = cfg

    def place(self, order: Order, mid: float, spread_bps: float) -> Fill | None:
        sign = 1 if order.qty > 0 else -1
        px = mid * (1 + sign * (spread_bps / 2 + self.cfg.slippage_bps) / 1e4)
        fee = abs(order.qty * px) * self.cfg.fee_bps / 1e4
        return Fill(order.client_id, order.symbol, order.qty, px, fee, order.ts_ms)


class CcxtExchange:
    """Live venue via a ccxt client. Only reachable in mode=live behind the capital gate."""

    def __init__(self, client: Any) -> None:
        self.client = client

    def place(self, order: Order, mid: float, spread_bps: float) -> Fill | None:
        side = "buy" if order.qty > 0 else "sell"
        params = {"clientOrderId": order.client_id}
        if order.reduce_only:
            params["reduceOnly"] = True
        try:
            res = self.client.create_order(order.symbol, "market", side, abs(order.qty), None, params)
        except Exception:  # venue errors must never crash the loop; the miss is journaled
            log.exception("order rejected by venue: %s", order)
            return None
        filled = float(res.get("filled") or 0.0)
        if filled <= 0:
            return None
        price = float(res.get("average") or res.get("price") or mid)
        fee = float((res.get("fee") or {}).get("cost") or 0.0)
        return Fill(order.client_id, order.symbol, filled if order.qty > 0 else -filled, price, fee, order.ts_ms)
