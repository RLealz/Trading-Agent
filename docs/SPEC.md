# SPEC — Two-Tier Crypto Trading Agent

Status: v0.1 · Owner: operator (capital gates only) · Engineer: agent

## 1. Objective

A continuously running crypto trading agent that surveys the market, isolates
mispriced setups and executes them without supervision. "Done" is a running
process, not a document. The human approves capital gates and nothing else.

## 2. Non-negotiable invariants

| ID  | Invariant | Enforced in |
|-----|-----------|-------------|
| I1  | Every threshold, size and order is computed in code. No model output is ever used as a number that sizes or gates an order without passing through code. | `policy.py`, `sizing.py`, `risk.py` |
| I2  | The fast model (Tier 2) has no memory, no tools, no side effects. It maps a typed `Snapshot` to a typed `Decision`. | `reflex.py` |
| I3  | Fire only when `setup_quality >= 2` AND `risk_state == safe` AND `confidence > 0.80` (plus `direction != neutral`, `toxic_flow == false`). | `policy.should_fire` |
| I4  | Size = fractional Kelly on the calibrated probability, hard-capped at 0.25 Kelly. | `sizing.py` |
| I5  | Max drawdown 15 % (trips the kill switch), max daily loss, max position. | `risk.RiskGate` |
| I6  | The kill switch is verified (armed + not tripped) before **every** order. Verification failure fails closed. | `risk.KillSwitch.verify` |
| I7  | The model can only tighten risk state, never loosen it: `final = worst(model, code)`. | `loop.py` |
| I8  | Snapshots are strictly causal: no field contains information unavailable at the snapshot timestamp; out-of-order events are rejected. | `state_engine.py` |
| I9  | Snapshot render is < 400 tokens (conservative estimate, enforced at runtime). | `state_engine.Snapshot.render` |
| I10 | Escalate to Tier 1 when confidence falls below 0.60 or regime flips to crisis. Cheap by default; escalations are rate-limited. | `policy.EscalationPolicy` |
| I11 | Every candle is journaled — fills **and** misses — with the raw score, so nightly review can grade calibration on the full tape. | `journal.py` |
| I12 | Nightly review scores Brier, refits calibration, and promotes a new schema only if it is measurably better out-of-sample. Hard risk limits are never touched by review. | `review.py` |
| I13 | Live trading requires a human capital approval file (amount, operator, expiry). Missing/expired → no orders. | `risk.CapitalGate` |
| I14 | Research output must cite primary sources with dates; un-sourced metrics are rejected unless labelled estimates. | `research.validate_report` |

## 3. Decision contract

```
regime         trending | mean_reverting | high_vol | crisis
direction      long | short | neutral
toxic_flow     boolean
setup_quality  0-3
risk_state     safe | near_limit | reduce
confidence     0.0-1.0
```

## 4. Stages and stop gates

| Stage | Artefact | Stop gate (must hold before next stage) |
|-------|----------|------------------------------------------|
| 1 Spec | this file | invariants enumerated, each mapped to a module |
| 2 Architecture | `docs/ARCHITECTURE.md` | every invariant has an owner module and a test |
| 3 Failing tests | `tests/` | suite runs and fails for the right reason (missing implementation) |
| 4 Implementation | `src/trading_agent/` | full suite green; paper loop runs end-to-end |
| 5 Review | `docs/WHERE_THIS_BREAKS.md` | adversarial pass written; findings fixed or accepted explicitly |
| 6 Deploy | `deploy/` | paper mode by default; live requires I13 approval file |

## 5. Out of scope (v0.1)

- Cross-venue routing, leverage/margin, options.
- Autonomous changes to hard risk limits (by design, never in scope).
