# nuPlan Closed-Loop Bridge Audit

- Bridge scene count: `120`
- Bridge base scene count: `40`
- Replay-infeasible scenes: `60`
- Replay-safe scenes: `60`
- Executed failures: `120`
- Executed safe: `0`

## Replay Vs Execution

| Replay class | Executed class | Count |
|---|---|---:|
| replay_infeasible | proxy_safe_but_realized_failed | 20 |
| replay_infeasible | replay_infeasible_but_executed_safe | 0 |
| replay_infeasible | replay_safe_but_executed_failed | 40 |
| replay_infeasible | executed_safe | 0 |
| replay_safe | proxy_safe_but_realized_failed | 0 |
| replay_safe | replay_infeasible_but_executed_safe | 0 |
| replay_safe | replay_safe_but_executed_failed | 60 |
| replay_safe | executed_safe | 0 |

## Selector Comparison

Absolute box-clearance failures use nuPlan executed geometry. Aligned replay failures use the logged-actor replay-equivalent clearance already used by the replay study, so this table should be read as directional transfer evidence rather than absolute closed-loop safety.

| Selector | Scenes | Executed box failures | Box fail rate | Aligned replay failures | Aligned fail rate | Mean aligned clearance | Progress | ADE3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| failure_gate | 40 | 40 | 1.000 | 0 | 0.000 | 1.497 | 9.988 | 7.680 |
| proxy_selector | 40 | 40 | 1.000 | 20 | 0.500 | 0.827 | 23.295 | 2.054 |
| replay_oracle | 40 | 40 | 1.000 | 0 | 0.000 | 1.639 | 5.194 | 10.020 |
