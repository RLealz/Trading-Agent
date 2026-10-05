# Trading-Agent

A two-tier crypto trading agent. A fast, calibrated classifier judges every candle. A large
reasoning model deliberates only when the fast one is unsure. Every threshold, size and order
lives in code the models cannot override.

```
MARKET → STATE ENGINE → TIER 2 REFLEX → RISK GATE → ORDER ┄┄► NIGHTLY REVIEW
          causal <400 tok   calibrated µs     kill switch   ¼-Kelly    Brier · recalibrate
                 ┊ escalate: conf < 0.60 or crisis (edge-triggered, budgeted)
                 └┄┄► TIER 1 DELIBERATION (Claude, async, TTL-bounded, typed output)
```

Fire only when `setup_quality ≥ 2`, `risk_state = safe` and `confidence > 0.80`, all checked in
code.

## Quickstart

```bash
pip install -e ".[dev]"               # add ".[live]" for exchange adapters
pytest                                # 81 tests, one per spec invariant and review finding

trading-agent train --blocks 6000     # fit reflex schema v1 per symbol (synthetic by default)
trading-agent paper --blocks 2880     # paper loop; every candle journaled to state/journal/
trading-agent review --date 2025-10-10  # Brier, reliability, refit, promote only if significantly better
trading-agent status
```

Paper-trade on live public data (no keys needed):

```bash
trading-agent paper --live-data binance            # add --tier1 to enable Claude deliberation
```

Tier 1 (needs `ANTHROPIC_API_KEY` or `ant auth login`):

```bash
trading-agent paper --tier1 ...      # deliberation on escalation
trading-agent review --narrate       # advisory commentary appended to the nightly report
trading-agent research               # regime + 5-10 sourced theses -> state/research/
```

## Operator controls (the human owns capital gates and nothing else)

```bash
trading-agent approve-capital --amount 5000 --operator <name> --days 7   # required for mode=live
trading-agent kill --reason "manual"                                     # flatten-only from next block
trading-agent reset-kill --operator <name>                               # logged to state/kill_resets.log
```

Live mode refuses to start without a valid approval. It also needs `mode = "live"` in the config.
It should not be used until the **blocks live** items in
[`docs/WHERE_THIS_BREAKS.md`](docs/WHERE_THIS_BREAKS.md) are closed.

## Documents

| Stage | Document |
|-------|----------|
| Spec + invariants | [`docs/SPEC.md`](docs/SPEC.md) |
| Architecture + module map | [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) |
| Research deliverable (runtime, sourced) | [`docs/RESEARCH.md`](docs/RESEARCH.md) |
| Adversarial review | [`docs/WHERE_THIS_BREAKS.md`](docs/WHERE_THIS_BREAKS.md) |
| Deploy | [`deploy/`](deploy/): Dockerfile, systemd loop service, nightly review timer |

## Layout

```
src/trading_agent/
  config.py        hard limits; refuses looser-than-contract values
  schema.py        Decision contract, worst()
  state_engine.py  causal snapshot engine, token budget
  reflex.py        Tier 2 classifier + Platt calibration + trainer
  policy.py        fire condition, escalation policy
  sizing.py        fractional Kelly, quarter-Kelly cap
  risk.py          kill switch, capital gate, risk gate
  portfolio.py     equity, drawdown, persisted risk clocks
  execution.py     paper + ccxt venues
  loop.py          per-block orchestration, async Tier 1
  journal.py       append-only tape
  schema_store.py  versioned reflex schemas, atomic pointer
  review.py        nightly review and promotion
  deliberation.py  Tier 1 verdicts (Claude, structured output)
  research.py      Tier 1 discovery + sourcing validator
  feeds.py         synthetic / CSV replay / ccxt polling
  cli.py           trading-agent entry point
```
