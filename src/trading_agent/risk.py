"""Deterministic risk layer. The model cannot argue with anything in this file."""

from __future__ import annotations

import json
import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from .config import RiskLimits
from .portfolio import Portfolio
from .schema import RiskState


class KillSwitch:
    """File-backed kill switch.

    - `arm()` writes a heartbeat; `verify()` checks the mechanism is live (armed in-process,
      heartbeat present, state dir writable). Verification failure blocks every order.
    - `trip()` writes KILL. Only a named human can `reset()`.
    """

    def __init__(self, state_dir: str | Path) -> None:
        self.dir = Path(state_dir)
        self._armed = False

    @property
    def kill_file(self) -> Path:
        return self.dir / "KILL"

    @property
    def heartbeat(self) -> Path:
        return self.dir / "heartbeat"

    def arm(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self.heartbeat.write_text(json.dumps({"pid": os.getpid(), "armed_at": time.time()}))
        self._armed = True

    def verify(self) -> bool:
        return self._armed and self.heartbeat.exists() and os.access(self.dir, os.W_OK)

    def tripped(self) -> bool:
        return self.kill_file.exists()

    def trip(self, reason: str) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        if not self.tripped():
            self.kill_file.write_text(json.dumps({"reason": reason, "at": time.time()}))

    def reason(self) -> str | None:
        return json.loads(self.kill_file.read_text())["reason"] if self.tripped() else None

    def reset(self, operator: str) -> None:
        if not operator or not operator.strip():
            raise ValueError("kill switch reset requires a named operator")
        if self.tripped():
            log = self.dir / "kill_resets.log"
            with log.open("a") as f:
                f.write(json.dumps({"operator": operator, "reason": self.reason(), "at": time.time()}) + "\n")
            self.kill_file.unlink()


class CapitalGate:
    """Human-approved capital ceiling. The single decision the operator owns."""

    FILE = "capital.json"

    def __init__(self, state_dir: str | Path, mode: str) -> None:
        self.path = Path(state_dir) / self.FILE
        self.mode = mode

    def approved_capital(self, now: float | None = None) -> float | None:
        if self.mode == "paper":
            return float("inf")
        if not self.path.exists():
            return None
        try:
            d = json.loads(self.path.read_text())
            amount, expires, operator = float(d["approved_capital"]), float(d["expires_at"]), d["approved_by"]
        except (KeyError, ValueError, TypeError, json.JSONDecodeError):
            return None
        if not operator or (now or time.time()) >= expires or amount <= 0:
            return None
        return amount

    @classmethod
    def approve(cls, state_dir: str | Path, amount: float, operator: str, days: float) -> Path:
        if not operator.strip() or amount <= 0 or days <= 0:
            raise ValueError("approval needs operator, positive amount and duration")
        p = Path(state_dir)
        p.mkdir(parents=True, exist_ok=True)
        now = time.time()
        tmp = p / (cls.FILE + ".tmp")
        tmp.write_text(json.dumps({"approved_capital": amount, "approved_by": operator,
                                   "approved_at": now, "expires_at": now + days * 86400}))
        os.replace(tmp, p / cls.FILE)
        return p / cls.FILE


@dataclass(frozen=True)
class OrderIntent:
    symbol: str
    target_qty: float
    current_qty: float
    price: float
    ts_ms: int
    snapshot_ts_ms: int


@dataclass(frozen=True)
class GateResult:
    approved: bool
    qty: float          # signed order delta
    reduce_only: bool
    reason: str


class RiskGate:
    def __init__(self, limits: RiskLimits, kill: KillSwitch, capital: CapitalGate) -> None:
        self.limits = limits
        self.kill = kill
        self.capital = capital
        self._orders: deque[int] = deque()

    def risk_state(self, pf: Portfolio) -> RiskState:
        L = self.limits
        dd, dl = pf.drawdown(), pf.daily_loss()
        if self.kill.tripped() or dd >= L.max_drawdown or dl >= L.max_daily_loss:
            return RiskState.REDUCE
        if dd >= L.near_limit_ratio * L.max_drawdown or dl >= L.near_limit_ratio * L.max_daily_loss:
            return RiskState.NEAR_LIMIT
        return RiskState.SAFE

    def enforce(self, pf: Portfolio) -> None:
        """Called every block, order or not: a flat book past max drawdown still trips the switch."""
        if pf.drawdown() >= self.limits.max_drawdown:
            self.kill.trip(f"max drawdown {pf.drawdown():.4f} >= {self.limits.max_drawdown}")

    def _reject(self, reason: str) -> GateResult:
        return GateResult(False, 0.0, False, reason)

    def check(self, it: OrderIntent, pf: Portfolio) -> GateResult:
        L = self.limits
        # 1. kill switch verified before every single order; fail closed
        if not self.kill.verify():
            return self._reject("kill switch not armed")
        self.enforce(pf)

        reduce_only = abs(it.target_qty) <= abs(it.current_qty) and it.target_qty * it.current_qty >= 0
        if self.kill.tripped() and not reduce_only:
            return self._reject(f"kill switch tripped: {self.kill.reason()}")

        # 2. data freshness and order rate
        if it.ts_ms - it.snapshot_ts_ms > L.max_snapshot_age_ms:
            return self._reject("stale snapshot")
        while self._orders and it.ts_ms - self._orders[0] >= 60_000:
            self._orders.popleft()
        if len(self._orders) >= L.max_orders_per_minute:
            return self._reject("order rate limit")
        if it.price <= 0:
            return self._reject("bad price")

        target = it.target_qty
        if not reduce_only:
            state = self.risk_state(pf)
            if state is not RiskState.SAFE:
                return self._reject(f"risk state {state.value}: no new risk")
            equity = pf.equity()
            cap = L.max_position_frac * equity
            gross_room = L.max_gross_frac * equity - pf.gross_notional(exclude=it.symbol)
            approved = self.capital.approved_capital()
            if approved is None:
                return self._reject("no valid capital approval")
            capital_room = approved - pf.gross_notional(exclude=it.symbol)
            max_notional = max(0.0, min(cap, gross_room, capital_room))
            if abs(target) * it.price > max_notional:
                target = (1 if target > 0 else -1) * max_notional / it.price

        delta = target - it.current_qty
        if abs(delta) * it.price < L.min_order_notional:
            return self._reject("below min order notional")
        self._orders.append(it.ts_ms)
        return GateResult(True, delta, reduce_only, "ok")
