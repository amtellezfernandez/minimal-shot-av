# nuPlan Regime-Value Selector

- Input replay JSON: `artifacts/corl2027/nuplan_public_replay_study_interaction1000_expert/replay.json`
- Split unit: `source_db_file`
- Train scenes: `700`
- Holdout scenes: `300`
- Marginal retention floor: `0.500`

## Regime Counts

- Train: `{"intervention_recovery_required": 117, "marginal_recovery_available": 159, "safe_proxy_keep": 320, "unrecoverable_no_safe_token": 104}`
- Holdout: `{"intervention_recovery_required": 21, "marginal_recovery_available": 49, "safe_proxy_keep": 53, "unrecoverable_no_safe_token": 177}`

## Train

| Selector | Replay-infeasible | Rate | Recovery | Progress | ADE3 | FDE3 | Entropy |
|---|---:|---:|---:|---:|---:|---:|---:|
| proxy_selector | 380 | 0.543 | 0/276 | 25.728 | 2.581 | 4.539 | 2.376 |
| safety_value_baseline | 104 | 0.149 | 276/276 | 13.993 | 6.496 | 10.622 | 2.768 |
| regime_value | 114 | 0.163 | 267/276 | 15.290 | 5.916 | 9.720 | 2.783 |
| regime_objective_oracle | 104 | 0.149 | 276/276 | 20.740 | 4.386 | 7.468 | 2.604 |
| replay_oracle | 104 | 0.149 | 276/276 | 11.221 | 7.702 | 12.717 | 2.677 |

### Token Histograms

- `proxy_selector`: `{"evasive_left": 92, "evasive_right": 45, "lane_recover": 72, "maintain": 299, "nudge_left": 17, "nudge_right": 73, "slow_yield": 102}`
- `safety_value_baseline`: `{"crawl": 142, "evasive_left": 132, "evasive_right": 36, "lane_recover": 10, "maintain": 60, "nudge_left": 40, "nudge_right": 11, "slow_yield": 159, "stop": 110}`
- `regime_value`: `{"crawl": 136, "evasive_left": 98, "evasive_right": 40, "lane_recover": 58, "maintain": 62, "nudge_left": 6, "nudge_right": 28, "slow_yield": 207, "stop": 65}`
- `regime_objective_oracle`: `{"crawl": 92, "evasive_left": 42, "evasive_right": 30, "lane_recover": 56, "maintain": 247, "nudge_left": 4, "nudge_right": 47, "slow_yield": 157, "stop": 25}`
- `replay_oracle`: `{"crawl": 113, "evasive_left": 132, "evasive_right": 42, "lane_recover": 22, "maintain": 55, "nudge_left": 8, "nudge_right": 13, "slow_yield": 103, "stop": 212}`

## Holdout

| Selector | Replay-infeasible | Rate | Recovery | Progress | ADE3 | FDE3 | Entropy |
|---|---:|---:|---:|---:|---:|---:|---:|
| proxy_selector | 247 | 0.823 | 0/70 | 33.173 | 2.217 | 3.740 | 2.082 |
| safety_value_baseline | 187 | 0.623 | 60/70 | 25.075 | 4.961 | 8.123 | 2.605 |
| regime_value | 185 | 0.617 | 62/70 | 22.637 | 5.524 | 8.952 | 2.559 |
| regime_objective_oracle | 177 | 0.590 | 70/70 | 32.282 | 3.178 | 5.463 | 1.575 |
| replay_oracle | 177 | 0.590 | 70/70 | 15.598 | 8.484 | 14.080 | 2.197 |

### Token Histograms

- `proxy_selector`: `{"evasive_left": 7, "evasive_right": 67, "lane_recover": 57, "maintain": 136, "nudge_left": 6, "nudge_right": 7, "slow_yield": 20}`
- `safety_value_baseline`: `{"crawl": 18, "evasive_left": 9, "evasive_right": 115, "lane_recover": 12, "maintain": 61, "nudge_left": 12, "nudge_right": 9, "slow_yield": 31, "stop": 33}`
- `regime_value`: `{"crawl": 19, "evasive_left": 7, "evasive_right": 115, "lane_recover": 5, "maintain": 27, "nudge_left": 13, "nudge_right": 16, "slow_yield": 68, "stop": 30}`
- `regime_objective_oracle`: `{"evasive_right": 69, "lane_recover": 4, "maintain": 186, "nudge_right": 4, "slow_yield": 16, "stop": 21}`
- `replay_oracle`: `{"crawl": 42, "evasive_left": 7, "evasive_right": 124, "lane_recover": 3, "maintain": 8, "nudge_left": 2, "nudge_right": 5, "slow_yield": 22, "stop": 87}`

