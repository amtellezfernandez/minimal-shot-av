# nuPlan Closed-Loop Bridge Audit

- Bridge scene count: `30`
- Bridge base scene count: `10`
- Replay-infeasible scenes: `15`
- Replay-safe scenes: `15`
- Executed failures: `30`
- Executed safe: `0`

## Replay Vs Execution

| Replay class | Executed class | Count |
|---|---|---:|
| replay_infeasible | proxy_safe_but_realized_failed | 5 |
| replay_infeasible | replay_infeasible_but_executed_safe | 0 |
| replay_infeasible | replay_safe_but_executed_failed | 10 |
| replay_infeasible | executed_safe | 0 |
| replay_safe | proxy_safe_but_realized_failed | 0 |
| replay_safe | replay_infeasible_but_executed_safe | 0 |
| replay_safe | replay_safe_but_executed_failed | 15 |
| replay_safe | executed_safe | 0 |

## Selector Comparison

Absolute box-clearance failures use nuPlan executed geometry. Aligned replay failures use the logged-actor replay-equivalent clearance already used by the replay study, so this table should be read as directional transfer evidence rather than absolute closed-loop safety.

| Selector | Scenes | Executed box failures | Box fail rate | Aligned replay failures | Aligned fail rate | Mean aligned clearance |
|---|---:|---:|---:|---:|---:|---:|
| failure_gate | 10 | 10 | 1.000 | 0 | 0.000 | 1.245 |
| proxy_selector | 10 | 10 | 1.000 | 5 | 0.500 | 0.736 |
| replay_oracle | 10 | 10 | 1.000 | 0 | 0.000 | 1.522 |
