"""Deterministic state engine: order book + trades -> compact causal snapshot per block.

Causality rules:
- events must arrive in non-decreasing timestamp order, otherwise CausalityError;
- a block can only close at a timestamp >= the last event seen;
- every snapshot field is a function of events with ts <= snapshot.ts and of the
  account state as of that instant. Nothing else enters.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass

from .config import ObservationConfig


class CausalityError(RuntimeError):
    pass


class TokenBudgetError(RuntimeError):
    pass


@dataclass(frozen=True)
class BookUpdate:
    ts: int
    symbol: str
    bids: tuple[tuple[float, float], ...]  # best first
    asks: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class Trade:
    ts: int
    symbol: str
    price: float
    size: float
    aggressor: str  # "buy" | "sell"


@dataclass(frozen=True)
class AccountState:
    inventory: float = 0.0   # signed position notional / equity
    drawdown: float = 0.0    # from peak equity, >= 0
    daily_pnl: float = 0.0   # fraction of session-start equity


@dataclass(frozen=True)
class Snapshot:
    ts: int
    symbol: str
    block: int
    mid: float
    spread_bps: float
    imbalance: float
    microprice_bps: float
    ret_short_bps: float
    ret_long_bps: float
    rvol_bps: float
    flow_imbalance: float
    inventory: float
    drawdown: float
    daily_pnl: float
    warm: bool

    def render(self) -> str:
        return (
            f"{self.symbol} t={self.ts} b={self.block} mid={self.mid:.6g} spr={self.spread_bps:.2f}bp "
            f"imb={self.imbalance:+.3f} mp={self.microprice_bps:+.2f}bp r_s={self.ret_short_bps:+.1f}bp "
            f"r_l={self.ret_long_bps:+.1f}bp vol={self.rvol_bps:.1f}bp flow={self.flow_imbalance:+.3f} "
            f"inv={self.inventory:+.3f} dd={self.drawdown:.4f} dpnl={self.daily_pnl:+.4f} warm={int(self.warm)}"
        )

    def token_estimate(self) -> int:
        # Conservative: numeric text tokenises at roughly 2.5-3.5 chars/token.
        return math.ceil(len(self.render()) / 2.5)

    def to_dict(self) -> dict:
        return asdict(self)


class StateEngine:
    snapshot_type = Snapshot

    def __init__(self, symbol: str, cfg: ObservationConfig) -> None:
        self.symbol = symbol
        self.cfg = cfg
        self._last_ts: int | None = None
        self._book: BookUpdate | None = None
        self._buy = 0.0
        self._sell = 0.0
        self._mids: deque[float] = deque(maxlen=cfg.long_window + 1)
        self._block = 0

    @property
    def last_mid(self) -> float | None:
        if self._book is None:
            return None
        return (self._book.bids[0][0] + self._book.asks[0][0]) / 2

    def _advance(self, ts: int) -> None:
        if self._last_ts is not None and ts < self._last_ts:
            raise CausalityError(f"{self.symbol}: event ts {ts} < last ts {self._last_ts}")
        self._last_ts = ts

    def on_event(self, ev: BookUpdate | Trade) -> None:
        if ev.symbol != self.symbol:
            raise ValueError(f"event for {ev.symbol} sent to {self.symbol} engine")
        self._advance(ev.ts)
        if isinstance(ev, BookUpdate):
            if not ev.bids or not ev.asks:
                return
            self._book = ev
        elif isinstance(ev, Trade):
            if ev.aggressor == "buy":
                self._buy += ev.size
            else:
                self._sell += ev.size

    def close_block(self, ts: int, account: AccountState) -> Snapshot:
        self._advance(ts)
        if self._book is None:
            raise CausalityError(f"{self.symbol}: no book before block close at {ts}")
        bids, asks = self._book.bids, self._book.asks
        bid, bsz = bids[0]
        ask, asz = asks[0]
        mid = (bid + ask) / 2
        spread_bps = (ask - bid) / mid * 1e4

        n = self.cfg.depth_levels
        bdepth = sum(s for _, s in bids[:n])
        adepth = sum(s for _, s in asks[:n])
        imbalance = (bdepth - adepth) / (bdepth + adepth) if bdepth + adepth > 0 else 0.0
        micro = (bid * asz + ask * bsz) / (bsz + asz) if bsz + asz > 0 else mid
        microprice_bps = (micro - mid) / mid * 1e4

        flow_total = self._buy + self._sell
        flow = (self._buy - self._sell) / flow_total if flow_total > 0 else 0.0

        self._mids.append(mid)
        mids = list(self._mids)
        s, l = self.cfg.short_window, self.cfg.long_window
        ret_s = math.log(mid / mids[-1 - s]) * 1e4 if len(mids) > s else 0.0
        ret_l = math.log(mid / mids[0]) * 1e4 if len(mids) > 1 else 0.0
        if len(mids) > 2:
            rets = [math.log(b / a) for a, b in zip(mids[:-1], mids[1:])]
            mu = sum(rets) / len(rets)
            rvol = math.sqrt(sum((r - mu) ** 2 for r in rets) / (len(rets) - 1)) * 1e4
        else:
            rvol = 0.0

        snap = Snapshot(
            ts=ts,
            symbol=self.symbol,
            block=self._block,
            mid=mid,
            spread_bps=spread_bps,
            imbalance=imbalance,
            microprice_bps=microprice_bps,
            ret_short_bps=ret_s,
            ret_long_bps=ret_l,
            rvol_bps=rvol,
            flow_imbalance=flow,
            inventory=account.inventory,
            drawdown=account.drawdown,
            daily_pnl=account.daily_pnl,
            warm=len(mids) > l,
        )
        if snap.token_estimate() >= self.cfg.max_tokens:
            raise TokenBudgetError(f"snapshot exceeds token budget: {snap.token_estimate()}")
        self._block += 1
        self._buy = self._sell = 0.0
        return snap
