# Architecture

```
MARKET ──► STATE ENGINE ──► TIER 2 REFLEX ──► RISK GATE ──► ORDER ┄┄► NIGHTLY REVIEW
order book  deterministic    calibrated         code only      sized in   Brier · recalibrate
            causal <400 tok  logistic, µs-ms    kill switch    code       promote if OOS-better
                 ┊                ┊ escalate (conf<0.60 | crisis), rate-limited      ┊
                 └┄┄┄┄┄┄┄┄┄► TIER 1 DELIBERATION (Claude, async, TTL-bounded) ◄┄┄┄┄┘
                                  research · thesis · re-read · narrative review
```

Solid = live path, every candle. Dotted = control path, rare.

## Module map

| Module | Role | Invariants |
|--------|------|-----------|
| `config.py` | TOML → frozen dataclasses. Hard limits live here and nowhere else. | I1, I5 |
| `schema.py` | `Decision`, enums, `worst()` | §3 contract, I7 |
| `state_engine.py` | Book/trade events → `Snapshot` per block. Monotonic ts guard. Token budget. | I8, I9 |
| `reflex.py` | Tier 2: standardised features → logistic → Platt calibration → `Decision`. Pure function of `(params, snapshot)`. Also offline trainer. | I2 |
| `policy.py` | `should_fire`, `EscalationPolicy` (transition-triggered, cooldown, hourly budget). | I3, I10 |
| `sizing.py` | Kelly, fractional, quarter-Kelly cap. | I4 |
| `risk.py` | `KillSwitch`, `CapitalGate`, `RiskGate` (drawdown, daily loss, position cap, staleness, order-rate). | I5, I6, I13 |
| `portfolio.py` | Cash, positions, marks, peak and day-start equity. | — |
| `execution.py` | `Exchange` protocol, `PaperExchange` (spread + slippage + fees), `CcxtExchange` (live, optional). | — |
| `journal.py` | Append-only JSONL tape, one file per UTC session. | I11 |
| `schema_store.py` | Versioned per-symbol reflex params, atomic `current` pointer. | I12 |
| `review.py` | Labels from the tape, Brier/log-loss/reliability, Platt refit, OOS promotion, report. | I12 |
| `deliberation.py` | Tier 1 via Anthropic SDK: structured `Decision` output, refusal → no trade, errors → no trade. | I1, I10 |
| `research.py` | Tier 1 discovery: web search → typed `ResearchReport` → code validation of sourcing. | I14 |
| `feeds.py` | `SyntheticFeed` (labelled synthetic), `CsvReplayFeed`, `CcxtPollingFeed`. | I8 |
| `loop.py` | Orchestrates one block per symbol; async Tier 1 with as-of stamping. | I6, I7 |
| `cli.py` | `paper`, `train`, `review`, `research`, `kill`, `reset-kill`, `approve-capital`. | I13 |

## Data flow per block

1. Feed events → `StateEngine` (rejects out-of-order).
2. Block close → `Snapshot(ts, …, inventory, drawdown, daily_pnl)`.
3. `ReflexModel.judge(snapshot)` → `Decision` + raw score.
4. If a Tier 1 verdict is active (arrived after its as-of ts, inside TTL) it replaces the reflex decision.
5. `EscalationPolicy` may submit an async Tier 1 job (never blocks the loop; the current block trades on the cheap verdict or not at all).
6. `final.risk_state = worst(decision.risk_state, RiskGate.risk_state(portfolio))`.
7. Target position: fire → Kelly size; `reduce` → flat; holding period expired → flat; else hold.
8. `RiskGate.check(order)` → `Exchange.place` → `Portfolio.apply_fill`.
9. Journal record (fill or miss).

## Why Tier 1 is async

A deliberation takes seconds to minutes; candles do not wait. The loop never trades on a
verdict computed from a snapshot older than the verdict's TTL, and a verdict is only applied
to blocks strictly after the snapshot it was computed from.

## Persistence layout (`state_dir`)

```
state/
  KILL                  tripped kill switch (presence = tripped), human reset only
  heartbeat             kill-switch arming heartbeat
  capital.json          human capital approval (live only)
  schemas/<SYM>/v<N>.json, current.json
  journal/<YYYY-MM-DD>.jsonl
  reports/<YYYY-MM-DD>.md
```
