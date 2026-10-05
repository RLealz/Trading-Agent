"""The live loop: one pass per block per symbol.

snapshot -> reflex verdict -> (active Tier 1 verdict?) -> escalation -> worst(model, code) risk
-> target in code -> risk gate -> exchange -> portfolio -> journal (fill or miss)
"""

from __future__ import annotations

import concurrent.futures
import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Iterable, Protocol

from .config import AppConfig
from .execution import Exchange, Order, new_client_id
from .feeds import BlockClose
from .journal import Journal, utc_date
from .policy import EscalationPolicy, should_fire
from .portfolio import Portfolio
from .reflex import ReflexModel, Verdict
from .risk import OrderIntent, RiskGate
from .schema import Decision, Direction, RiskState, worst
from .schema_store import SchemaStore
from .sizing import position_fraction
from .state_engine import AccountState, BookUpdate, Snapshot, StateEngine, Trade

log = logging.getLogger(__name__)


class Judge(Protocol):
    params: object

    def judge(self, s: Snapshot) -> Verdict: ...


class Deliberator(Protocol):
    def deliberate(self, snapshot: Snapshot, history: list[Snapshot], reflex: Decision,
                   thesis: str | None = None) -> Decision | None: ...


@dataclass
class _Tier1Verdict:
    decision: Decision
    as_of_ts: int
    expires_ts: int


class TradingLoop:
    def __init__(
        self,
        cfg: AppConfig,
        exchange: Exchange,
        portfolio: Portfolio,
        gate: RiskGate,
        models: dict[str, Judge],
        journal: Journal,
        deliberator: Deliberator | None = None,
        executor: concurrent.futures.Executor | None = None,
        clock_ms: Callable[[], int] | None = None,
        store: SchemaStore | None = None,
        theses: dict[str, str] | None = None,
        portfolio_path: str | None = None,
    ) -> None:
        self.cfg = cfg
        self.exchange = exchange
        self.portfolio = portfolio
        self.gate = gate
        self.models = models
        self.journal = journal
        self.deliberator = deliberator
        self.executor = executor or concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="tier1")
        self.clock_ms = clock_ms
        self.store = store
        self.theses = theses or {}
        self.portfolio_path = portfolio_path
        self.engines = {s: StateEngine(s, cfg.observation) for s in cfg.symbols}
        self.escalation = EscalationPolicy(cfg.policy)
        self.history: dict[str, deque[Snapshot]] = {s: deque(maxlen=cfg.observation.long_window) for s in cfg.symbols}
        self.entry_block: dict[str, int] = {}
        self.tier1: dict[str, _Tier1Verdict] = {}
        self.pending: dict[str, tuple[concurrent.futures.Future, int, Decision]] = {}

    # ------------------------------------------------------------- feed driver

    def run(self, feed: Iterable, max_blocks: int | None = None) -> int:
        blocks = 0
        for ev in feed:
            if isinstance(ev, BlockClose):
                self.on_block_close(ev.ts)
                blocks += 1
                if max_blocks is not None and blocks >= max_blocks:
                    break
            elif isinstance(ev, (BookUpdate, Trade)) and ev.symbol in self.engines:
                self.engines[ev.symbol].on_event(ev)
        return blocks

    def on_block_close(self, ts: int) -> None:
        self.portfolio.roll_session(utc_date(ts))
        self._reload_schemas()
        for sym, eng in self.engines.items():
            if eng.last_mid is not None:
                self.portfolio.mark(sym, eng.last_mid)
        self.gate.enforce(self.portfolio)
        for sym, eng in self.engines.items():
            if eng.last_mid is None:
                continue
            pf = self.portfolio
            eq = pf.equity()
            account = AccountState(inventory=pf.notional(sym) / eq if eq > 0 else 0.0,
                                   drawdown=pf.drawdown(), daily_pnl=pf.daily_pnl())
            self.on_snapshot(eng.close_block(ts, account))
        if self.portfolio_path:
            self.portfolio.save(self.portfolio_path)

    def _reload_schemas(self) -> None:
        """Pick up whatever the nightly review shipped, at the first block after it lands."""
        if not self.store:
            return
        for sym in self.cfg.symbols:
            v = self.store.current_version(sym)
            if v is not None and v != getattr(self.models[sym].params, "version", None):
                log.info("%s reflex schema -> v%d", sym, v)
                self.models[sym] = ReflexModel(self.store.load_version(sym, v))

    # ------------------------------------------------------------- one block

    def _now(self, snap: Snapshot) -> int:
        return self.clock_ms() if self.clock_ms else snap.ts

    def _collect_tier1(self, sym: str) -> None:
        job = self.pending.get(sym)
        if not job or not job[0].done():
            return
        fut, as_of, reflex = self.pending.pop(sym)
        try:
            d = fut.result()
        except Exception:
            log.exception("tier1 deliberation failed for %s", sym)
            d = None
        if d is None:
            # failure or refusal: the cheap tier was unsure and the expensive one gave no answer,
            # so stand down for the TTL rather than trade on the unsure verdict
            d = reflex.model_copy(update={"direction": Direction.NEUTRAL, "setup_quality": 0, "confidence": 0.5})
        self.tier1[sym] = _Tier1Verdict(d, as_of, as_of + self.cfg.policy.tier1_ttl_blocks * self.cfg.block_ms)

    def on_snapshot(self, snap: Snapshot) -> dict:
        sym, cfg = snap.symbol, self.cfg
        model = self.models[sym]
        t0 = time.perf_counter()
        verdict = model.judge(snap)
        reflex_ms = (time.perf_counter() - t0) * 1000
        decision, source = verdict.decision, "reflex"
        params = model.params

        # Tier 1 verdict applies only to blocks strictly after the snapshot it was computed from
        self._collect_tier1(sym)
        t1 = self.tier1.get(sym)
        if t1 and t1.as_of_ts < snap.ts <= t1.expires_ts:
            shrink = getattr(params, "tier1_shrink", 0.5)
            decision = t1.decision.model_copy(update={"confidence": 0.5 + shrink * (t1.decision.confidence - 0.5)})
            source = "tier1"
        elif t1 and snap.ts > t1.expires_ts:
            del self.tier1[sym]

        escalated = False
        if (self.deliberator is not None and sym not in self.pending
                and self.escalation.should_escalate(sym, verdict.decision, snap.ts / 1000)):
            hist = list(self.history[sym])
            fut = self.executor.submit(self.deliberator.deliberate, snap, hist, verdict.decision, self.theses.get(sym))
            self.pending[sym] = (fut, snap.ts, verdict.decision)
            escalated = True
        self.history[sym].append(snap)

        code_risk = self.gate.risk_state(self.portfolio)
        final_risk = worst(decision.risk_state, code_risk)
        decision = decision.model_copy(update={"risk_state": final_risk})

        current = self.portfolio.position(sym)
        target = current
        fired = should_fire(decision, cfg.policy) and (cfg.policy.allow_short or decision.sign > 0)
        payoff = getattr(params, "payoff_ratio", cfg.sizing.default_payoff_ratio)
        if final_risk is RiskState.REDUCE:
            target = 0.0
        elif fired:
            frac = position_fraction(decision.confidence, payoff, cfg.sizing.kelly_multiplier)
            target = decision.sign * frac * self.portfolio.equity() / snap.mid
            if target != 0:
                self.entry_block[sym] = snap.block
        elif current != 0 and snap.block - self.entry_block.get(sym, snap.block) >= cfg.policy.holding_blocks:
            target = 0.0

        # hysteresis: do not churn small rebalances of an existing same-side position
        if target != 0 and current != 0 and target * current > 0 and \
                abs(target - current) <= cfg.sizing.rebalance_band * abs(current):
            target = current

        order_rec = fill_rec = None
        reason = "no change"
        if target != current:
            res = self.gate.check(OrderIntent(sym, target, current, snap.mid, self._now(snap), snap.ts), self.portfolio)
            reason = res.reason
            if res.approved:
                order = Order(new_client_id(sym, snap.ts), sym, res.qty, res.reduce_only, snap.ts)
                order_rec = {"qty": order.qty, "reduce_only": order.reduce_only, "client_id": order.client_id}
                fill = self.exchange.place(order, snap.mid, snap.spread_bps)
                if fill:
                    self.portfolio.apply_fill(fill)
                    fill_rec = fill.to_dict()
                    if self.portfolio.position(sym) == 0:
                        self.entry_block.pop(sym, None)

        rec = {
            "ts": snap.ts, "symbol": sym, "block": snap.block,
            "snapshot": snap.to_dict(),
            "schema_version": getattr(params, "version", None),
            "score": verdict.score, "p_up": verdict.p_up, "reflex_ms": round(reflex_ms, 4),
            "reflex": verdict.decision.model_dump(mode="json"),
            "decision": decision.model_dump(mode="json"),
            "tier1_raw_confidence": t1.decision.confidence if source == "tier1" else None,
            "regime": decision.regime.value, "direction": decision.direction.value,
            "source": source, "escalated": escalated,
            "code_risk_state": code_risk.value, "final_risk_state": final_risk.value,
            "fired": fired, "target_qty": target, "position_before": current,
            "order": order_rec, "fill": fill_rec, "gate_reason": reason,
            "equity": self.portfolio.equity(), "drawdown": self.portfolio.drawdown(),
        }
        self.journal.write(rec)
        return rec

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)
