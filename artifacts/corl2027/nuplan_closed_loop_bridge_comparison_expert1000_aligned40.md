# nuPlan Closed-Loop Bridge Audit

- Bridge scene count: `120`
- Replay-infeasible scenes: `60`
- Replay-safe scenes: `60`
- Executed failures: `120`
- Executed safe: `0`

## Replay Vs Execution

| Replay class | Executed class | Count |
|---|---|---:|
| replay_infeasible | proxy_safe_but_realized_failed | 40 |
| replay_infeasible | replay_infeasible_but_executed_safe | 0 |
| replay_infeasible | replay_safe_but_executed_failed | 20 |
| replay_infeasible | executed_safe | 0 |
| replay_safe | proxy_safe_but_realized_failed | 0 |
| replay_safe | replay_infeasible_but_executed_safe | 0 |
| replay_safe | replay_safe_but_executed_failed | 60 |
| replay_safe | executed_safe | 0 |

## Selector Comparison

| Selector | Scenes | Executed box failures | Box fail rate | Aligned replay failures | Aligned fail rate | Mean aligned clearance |
|---|---:|---:|---:|---:|---:|---:|
| proxy_selector | 40 | 40 | 1.000 | 20 | 0.500 | 0.827 |
| replay_calibrated | 40 | 40 | 1.000 | 20 | 0.500 | 1.042 |
| replay_oracle | 40 | 40 | 1.000 | 0 | 0.000 | 1.639 |
