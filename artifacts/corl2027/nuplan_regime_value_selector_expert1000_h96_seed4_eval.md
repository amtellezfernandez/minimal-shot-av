# nuPlan Regime-Value Selector

- Input replay JSON: `artifacts/corl2027/nuplan_public_replay_study_interaction1000_expert/replay.json`
- Split unit: `source_db_file`
- Train scenes: `845`
- Holdout scenes: `155`
- Marginal retention floor: `0.500`

## Regime Counts

- Train: `{"intervention_recovery_required": 67, "marginal_recovery_available": 197, "safe_proxy_keep": 305, "unrecoverable_no_safe_token": 276}`
- Holdout: `{"intervention_recovery_required": 71, "marginal_recovery_available": 11, "safe_proxy_keep": 68, "unrecoverable_no_safe_token": 5}`

## Train

| Selector | Replay-infeasible | Rate | Recovery | Progress | ADE3 | FDE3 | Entropy |
|---|---:|---:|---:|---:|---:|---:|---:|
| proxy_selector | 540 | 0.639 | 0/264 | 28.784 | 2.529 | 4.422 | 2.224 |
| safety_value_baseline | 289 | 0.342 | 251/264 | 21.793 | 4.678 | 7.697 | 2.891 |
| regime_value | 287 | 0.340 | 253/264 | 21.985 | 4.437 | 7.291 | 2.873 |
| regime_objective_oracle | 276 | 0.327 | 264/264 | 26.249 | 3.660 | 6.356 | 2.356 |
| replay_oracle | 276 | 0.327 | 264/264 | 13.159 | 8.004 | 13.315 | 2.710 |

### Token Histograms

- `proxy_selector`: `{"evasive_left": 93, "evasive_right": 112, "lane_recover": 117, "maintain": 399, "nudge_left": 12, "nudge_right": 28, "slow_yield": 84}`
- `safety_value_baseline`: `{"crawl": 50, "evasive_left": 109, "evasive_right": 160, "lane_recover": 53, "maintain": 156, "nudge_left": 26, "nudge_right": 28, "slow_yield": 186, "stop": 77}`
- `regime_value`: `{"crawl": 35, "evasive_left": 104, "evasive_right": 160, "lane_recover": 66, "maintain": 125, "nudge_left": 4, "nudge_right": 95, "slow_yield": 191, "stop": 65}`
- `regime_objective_oracle`: `{"crawl": 21, "evasive_left": 37, "evasive_right": 99, "lane_recover": 55, "maintain": 408, "nudge_left": 4, "nudge_right": 42, "slow_yield": 133, "stop": 46}`
- `replay_oracle`: `{"crawl": 129, "evasive_left": 110, "evasive_right": 166, "lane_recover": 24, "maintain": 57, "nudge_left": 7, "nudge_right": 18, "slow_yield": 94, "stop": 240}`

## Holdout

| Selector | Replay-infeasible | Rate | Recovery | Progress | ADE3 | FDE3 | Entropy |
|---|---:|---:|---:|---:|---:|---:|---:|
| proxy_selector | 87 | 0.561 | 0/82 | 23.482 | 2.159 | 3.629 | 2.253 |
| safety_value_baseline | 5 | 0.032 | 82/82 | 11.370 | 6.626 | 10.508 | 2.054 |
| regime_value | 5 | 0.032 | 82/82 | 12.183 | 6.276 | 10.005 | 2.002 |
| regime_objective_oracle | 5 | 0.032 | 82/82 | 13.046 | 6.008 | 9.652 | 2.003 |
| replay_oracle | 5 | 0.032 | 82/82 | 9.125 | 7.570 | 12.098 | 2.218 |

### Token Histograms

- `proxy_selector`: `{"evasive_left": 6, "lane_recover": 12, "maintain": 36, "nudge_left": 11, "nudge_right": 52, "slow_yield": 38}`
- `safety_value_baseline`: `{"crawl": 65, "evasive_left": 30, "lane_recover": 1, "maintain": 6, "nudge_left": 3, "slow_yield": 42, "stop": 8}`
- `regime_value`: `{"crawl": 72, "evasive_left": 22, "lane_recover": 2, "maintain": 12, "nudge_left": 8, "slow_yield": 39}`
- `regime_objective_oracle`: `{"crawl": 71, "evasive_left": 5, "lane_recover": 5, "maintain": 25, "nudge_right": 9, "slow_yield": 40}`
- `replay_oracle`: `{"crawl": 26, "evasive_left": 29, "lane_recover": 1, "maintain": 6, "nudge_left": 3, "slow_yield": 31, "stop": 59}`

