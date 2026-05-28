# nuPlan Closed-Loop Bridge Audit

- Bridge scene count: `30`
- Replay-infeasible scenes: `15`
- Replay-safe scenes: `15`
- Executed failures: `30`
- Executed safe: `0`

## Replay Vs Execution

| Replay class | Executed class | Count |
|---|---|---:|
| replay_infeasible | proxy_safe_but_realized_failed | 10 |
| replay_infeasible | replay_infeasible_but_executed_safe | 0 |
| replay_infeasible | replay_safe_but_executed_failed | 5 |
| replay_infeasible | executed_safe | 0 |
| replay_safe | proxy_safe_but_realized_failed | 0 |
| replay_safe | replay_infeasible_but_executed_safe | 0 |
| replay_safe | replay_safe_but_executed_failed | 15 |
| replay_safe | executed_safe | 0 |

## Selector Comparison

| Selector | Scenes | Executed failures | Failure rate | Mean min clearance |
|---|---:|---:|---:|---:|
| proxy_selector | 10 | 10 | 1.000 | -1.271 |
| replay_calibrated | 10 | 10 | 1.000 | -2.782 |
| replay_oracle | 10 | 10 | 1.000 | -4.492 |
