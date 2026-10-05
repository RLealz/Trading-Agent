# WHERE THIS BREAKS

This is the adversarial review, done before anything ships. It prefers the system that survives
over the one that backtests prettier. Each item says whether it is **fixed**, **mitigated** or
**open**. Open items block live capital until someone closes them.

## The five questions

**1. Is the edge structural, or just an incentive that ends?**
There is no proven edge yet. The reflex learns from order-flow and book imbalance. On the
synthetic feed that signal exists because the generator puts it there. On a real venue,
top-of-book imbalance at a one-minute horizon is heavily arbitraged and decays as liquidity
providers adapt. Treat every result before live-data paper trading as plumbing validation, not
evidence of edge. **Open:** run at least 30 days of `paper --live-data` and require the nightly
Brier score to beat climatology on every symbol before asking for capital.

**2. Is the catalyst already in the price?**
Research separates confirmed catalysts from speculation and forces an invalidation trigger per
thesis. It cannot tell whether the market has already priced a dated unlock or listing. The
thesis reaches Tier 1 only as context, never as a direction. **Mitigated** by design: a thesis
can't fire a trade on its own.

**3. Does value actually accrue to the token?**
`value_accrual` is a required field and needs an argued answer. The validator checks that the
reasoning is there. It cannot check that the reasoning is correct. **Open:** a human should read
the research report before it informs any symbol selection.

**4. Does the strategy survive fees and slippage?**
The paper exchange crosses half the spread, adds 2 bps of slippage and pays 5 bps in taker fees
on every fill. The nightly payoff-ratio estimate is net of round-trip fees. On the 2,880-candle
synthetic run, fees were 15.26 of 177.69 gross P&L (8.6%) at very low turnover. At one-minute horizons, fees will
dominate any small edge. **Open:** a maker-only execution mode. Taker-only is the honest default
until then.

**5. Has any hard limit been delegated to the model?**
No.

- Every threshold lives in `config.py`, which refuses values looser than the contract allows
  (drawdown above 15%, Kelly above 0.25, fire confidence below 0.80, setup quality below 2).
- The model can only tighten `risk_state`: the loop takes `worst(model, code)`.
- Tier 1 output passes the same fire condition and risk gate as Tier 2. Its confidence is also
  shrunk toward 0.5 until the nightly review has graded enough Tier 1 verdicts to earn more
  trust.
- The review cannot reach `RiskLimits`.

## Findings from the adversarial pass

| # | Finding | Status |
|---|---------|--------|
| 1 | A process restart reset `peak_equity` and `day_start_equity`, which silently cleared the drawdown and daily-loss clocks. | **Fixed:** portfolio state is persisted each block and restored on start (`Portfolio.save/load`). |
| 2 | The kill switch tripped on max drawdown only when an order was attempted, so a flat book past 15% stayed untripped. | **Fixed:** `RiskGate.enforce` runs every block. |
| 3 | A fixed Brier-delta threshold promoted noise. A 0.0009 "gain" on 432 OOS samples passed it. | **Fixed:** promotion requires a paired t-statistic > 2.33 on the per-candle squared-error differences. |
| 4 | A promoted schema waited for the next UTC session roll, so a review finishing at 00:05 shipped a day late. | **Fixed:** the `current.json` pointer is checked every block. |
| 5 | Labels were computed by record offset, so a restart gap mislabelled outcomes across the gap. | **Fixed:** pairs spanning more than 1.5× the horizon are dropped. |
| 6 | Spot venues cannot short, so short signals became venue rejections. | **Fixed:** `policy.allow_short = false`. |
| 7 | Research text compiled from the web flows into the Tier 1 prompt, which is a prompt-injection path. | **Mitigated:** the thesis is framed as untrusted data, and Tier 1 has no tools, no sizing authority and shrunk confidence. The worst case is a forced neutral or a bad direction that still has to clear code gates. |
| 8 | **Calibrated models rarely say > 0.80.** On the synthetic tape, the reflex beat climatology out of sample (Brier 0.2305 vs 0.2569) with reliable bins, yet fired once in 5,760 candles. | **Accepted.** That is the gate working. Lowering 0.80 is a capital decision for the human, not a tuning knob. Longer horizons or better features are the legitimate levers. |
| 9 | The Tier 1 escalation rule as literally written (confidence < 0.60) holds on most candles of a calibrated model. | **Mitigated:** escalation is edge-triggered, with a per-symbol cooldown and an hourly budget. Cost is bounded at about 6 deliberations/hour. |
| 10 | The Tier 1 call is synchronous inside one worker thread, so a hung call blocks escalations for all symbols until the 600 s timeout. | **Open, low severity:** the candle path never waits. Only deliberation latency degrades. |
| 11 | The payoff-ratio estimate uses all directional calls, not only fills, so selection bias goes in an unknown direction. | **Open:** switch to fills-only once fill count is meaningful. It is clamped to [0.5, 2.0] meanwhile. |
| 12 | **Live position reconciliation does not exist.** The portfolio is local; the venue is the truth. Partial fills, manual intervention or a missed fill response desynchronise them. | **Open, blocks live.** Reconcile balances and positions against the venue at start and every N blocks, and trip the kill switch on mismatch. |
| 13 | `CcxtExchange` treats `fee.cost` as quote currency. Some venues charge fees in base or in a rebate token. | **Open, blocks live** for venues that don't charge in quote. |
| 14 | `CcxtPollingFeed` stamps events at receive time. That is causal, but it means a slow poll looks like stale data. | **Mitigated:** the gate rejects snapshots older than `max_snapshot_age_ms`. |
| 15 | The synthetic feed is not a market. Any P&L it produces is meaningless. | **Labelled** in code and docs. |
| 16 | `KillSwitch.verify` checks heartbeat presence, not freshness. A dead supervisor leaves a valid-looking heartbeat. | **Open:** use a supervisor-refreshed heartbeat with a max age. |

## Kill criteria for the whole project

Stop and do not request capital if, after 30 days of live-data paper trading, any of these hold:

- Out-of-sample Brier score does not beat climatology on a majority of symbols.
- Fill-only realised P&L, net of fees, is not positive.
- The kill switch tripped for any reason other than a drill.
