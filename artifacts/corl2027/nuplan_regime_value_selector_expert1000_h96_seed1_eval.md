# nuPlan Regime-Value Selector

- Input replay JSON: `artifacts/corl2027/nuplan_public_replay_study_interaction1000_expert/replay.json`
- Split unit: `source_db_file`
- Train scenes: `845`
- Holdout scenes: `155`
- Marginal retention floor: `0.500`

## Regime Counts

- Train: `{"intervention_recovery_required": 109, "marginal_recovery_available": 175, "safe_proxy_keep": 298, "unrecoverable_no_safe_token": 263}`
- Holdout: `{"intervention_recovery_required": 29, "marginal_recovery_available": 33, "safe_proxy_keep": 75, "unrecoverable_no_safe_token": 18}`

## Train

| Selector | Replay-infeasible | Rate | Recovery | Progress | ADE3 | FDE3 | Entropy |
|---|---:|---:|---:|---:|---:|---:|---:|
| proxy_selector | 547 | 0.647 | 0/284 | 28.472 | 2.388 | 4.001 | 2.411 |
| safety_value_baseline | 273 | 0.323 | 276/284 | 19.799 | 5.246 | 8.498 | 3.020 |
| regime_value | 271 | 0.321 | 278/284 | 18.680 | 5.513 | 8.886 | 2.893 |
| regime_objective_oracle | 263 | 0.311 | 284/284 | 24.713 | 4.026 | 6.758 | 2.456 |
| replay_oracle | 263 | 0.311 | 284/284 | 12.179 | 8.327 | 13.705 | 2.591 |

### Token Histograms

- `proxy_selector`: `{"evasive_left": 99, "evasive_right": 112, "lane_recover": 125, "maintain": 337, "nudge_left": 12, "nudge_right": 64, "slow_yield": 96}`
- `safety_value_baseline`: `{"crawl": 121, "evasive_left": 121, "evasive_right": 164, "lane_recover": 47, "maintain": 80, "nudge_left": 38, "nudge_right": 51, "slow_yield": 140, "stop": 83}`
- `regime_value`: `{"crawl": 104, "evasive_left": 127, "evasive_right": 167, "lane_recover": 63, "maintain": 39, "nudge_left": 13, "nudge_right": 50, "slow_yield": 192, "stop": 90}`
- `regime_objective_oracle`: `{"crawl": 75, "evasive_left": 42, "evasive_right": 96, "lane_recover": 58, "maintain": 356, "nudge_right": 23, "slow_yield": 161, "stop": 34}`
- `replay_oracle`: `{"crawl": 132, "evasive_left": 125, "evasive_right": 159, "lane_recover": 14, "maintain": 38, "nudge_left": 9, "nudge_right": 5, "slow_yield": 106, "stop": 257}`

## Holdout

| Selector | Replay-infeasible | Rate | Recovery | Progress | ADE3 | FDE3 | Entropy |
|---|---:|---:|---:|---:|---:|---:|---:|
| proxy_selector | 80 | 0.516 | 0/62 | 25.180 | 2.929 | 5.926 | 1.595 |
| safety_value_baseline | 23 | 0.148 | 57/62 | 15.013 | 5.131 | 8.677 | 2.807 |
| regime_value | 24 | 0.155 | 56/62 | 16.185 | 5.005 | 8.604 | 2.831 |
| regime_objective_oracle | 18 | 0.116 | 62/62 | 21.421 | 4.012 | 7.460 | 2.196 |
| replay_oracle | 18 | 0.116 | 62/62 | 14.471 | 5.808 | 9.968 | 2.847 |

### Token Histograms

- `proxy_selector`: `{"lane_recover": 4, "maintain": 98, "nudge_left": 11, "nudge_right": 16, "slow_yield": 26}`
- `safety_value_baseline`: `{"crawl": 24, "evasive_left": 13, "evasive_right": 8, "lane_recover": 6, "maintain": 7, "nudge_left": 12, "nudge_right": 13, "slow_yield": 53, "stop": 19}`
- `regime_value`: `{"crawl": 15, "evasive_left": 14, "evasive_right": 19, "lane_recover": 9, "maintain": 17, "nudge_left": 3, "nudge_right": 10, "slow_yield": 52, "stop": 16}`
- `regime_objective_oracle`: `{"crawl": 17, "evasive_right": 3, "lane_recover": 2, "maintain": 77, "nudge_left": 4, "nudge_right": 28, "slow_yield": 12, "stop": 12}`
- `replay_oracle`: `{"crawl": 23, "evasive_left": 14, "evasive_right": 7, "lane_recover": 11, "maintain": 25, "nudge_left": 1, "nudge_right": 13, "slow_yield": 19, "stop": 42}`

