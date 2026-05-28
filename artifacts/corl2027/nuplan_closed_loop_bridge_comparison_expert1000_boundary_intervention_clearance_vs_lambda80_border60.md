# nuPlan Closed-Loop Bridge Audit

- Bridge scene count: `240`
- Replay-infeasible scenes: `56`
- Replay-safe scenes: `184`
- Executed failures: `237`
- Executed safe: `3`

## Replay Vs Execution

| Replay class | Executed class | Count |
|---|---|---:|
| replay_infeasible | proxy_safe_but_realized_failed | 11 |
| replay_infeasible | replay_infeasible_but_executed_safe | 3 |
| replay_infeasible | replay_safe_but_executed_failed | 42 |
| replay_infeasible | executed_safe | 0 |
| replay_safe | proxy_safe_but_realized_failed | 11 |
| replay_safe | replay_infeasible_but_executed_safe | 0 |
| replay_safe | replay_safe_but_executed_failed | 173 |
| replay_safe | executed_safe | 0 |

## Selector Comparison

| Selector | Scenes | Executed box failures | Box fail rate | Aligned replay failures | Aligned fail rate | Mean aligned clearance |
|---|---:|---:|---:|---:|---:|---:|
| border_reference | 60 | 60 | 1.000 | 22 | 0.367 | 0.812 |
| boundary_intervention | 60 | 60 | 1.000 | 22 | 0.367 | 0.906 |
| proxy_selector | 60 | 57 | 0.950 | 41 | 0.683 | 0.108 |
| replay_oracle | 60 | 60 | 1.000 | 22 | 0.367 | 1.314 |
