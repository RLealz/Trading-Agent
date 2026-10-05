# Research — discovery deliverable

No theses are committed to this file, on purpose. A thesis is only valid with live primary
sources and their dates. A thesis written into the repository is stale the day after it is
committed, and anything not fetched live would be a fabricated metric.

Theses are produced at runtime by the deliberation tier:

```bash
export ANTHROPIC_API_KEY=...        # or `ant auth login`
trading-agent research              # writes state/research/<date>.md and .json
```

## What the run produces (`research.ResearchReport`)

| Field | Content |
|-------|---------|
| `regime` + `regime_reasoning` + `regime_sources` | Regime from BTC and ETH structure, dominance, stablecoin supply, funding, open interest and the narrative in rotation. |
| `candidates` (5–10) | Assets whose price has detached from fundamentals, adoption or a dated catalyst. |
| per candidate `metrics` | Emission schedule, unlock calendar, revenue, fee capture, TVL, real active users and holder concentration. Each metric has a dated source **or** `is_estimate = true`. |
| `value_accrual` | One of `accrues`, `partial`, `does_not_accrue` or `unknown`, with an argued answer: does value reach the token holder, or stop at the protocol? |
| `catalysts` | Each marked `confirmed` (needs a dated source) or speculative. |
| `bear_case` | The strongest case against the thesis. |
| `invalidation` | The single observation that would kill the thesis. |

## How it is validated (in code, `research.validate_report`)

- The report must contain 5–10 candidates.
- A metric without a dated source that is not labelled an estimate rejects the report.
- A confirmed catalyst without a dated source rejects the report.
- An empty bear case, invalidation or value-accrual argument rejects the report.
- A rejected report is written for inspection but **not** promoted to `latest.json`, so it never
  reaches the trading loop.

## How a thesis is used

Only as context for a Tier 1 re-read, framed as untrusted data. A thesis never sets a direction,
a size or a threshold.
