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
| safety_value_baseline | 104 | 0.149 | 276/276 | 14.534 | 6.173 | 10.056 | 2.743 |
| regime_value | 109 | 0.156 | 271/276 | 16.226 | 5.612 | 9.207 | 2.828 |
| regime_objective_oracle | 104 | 0.149 | 276/276 | 20.740 | 4.386 | 7.468 | 2.604 |
| replay_oracle | 104 | 0.149 | 276/276 | 11.221 | 7.702 | 12.717 | 2.677 |

### Token Histograms

- `proxy_selector`: `{"evasive_left": 92, "evasive_right": 45, "lane_recover": 72, "maintain": 299, "nudge_left": 17, "nudge_right": 73, "slow_yield": 102}`
- `safety_value_baseline`: `{"crawl": 55, "evasive_left": 128, "evasive_right": 38, "lane_recover": 31, "maintain": 42, "nudge_left": 19, "nudge_right": 29, "slow_yield": 194, "stop": 164}`
- `regime_value`: `{"crawl": 123, "evasive_left": 114, "evasive_right": 41, "lane_recover": 55, "maintain": 80, "nudge_left": 4, "nudge_right": 32, "slow_yield": 185, "stop": 66}`
- `regime_objective_oracle`: `{"crawl": 92, "evasive_left": 42, "evasive_right": 30, "lane_recover": 56, "maintain": 247, "nudge_left": 4, "nudge_right": 47, "slow_yield": 157, "stop": 25}`
- `replay_oracle`: `{"crawl": 113, "evasive_left": 132, "evasive_right": 42, "lane_recover": 22, "maintain": 55, "nudge_left": 8, "nudge_right": 13, "slow_yield": 103, "stop": 212}`

## Holdout

| Selector | Replay-infeasible | Rate | Recovery | Progress | ADE3 | FDE3 | Entropy |
|---|---:|---:|---:|---:|---:|---:|---:|
| proxy_selector | 247 | 0.823 | 0/70 | 33.173 | 2.217 | 3.740 | 2.082 |
| safety_value_baseline | 184 | 0.613 | 65/70 | 24.980 | 4.689 | 7.538 | 2.557 |
| regime_value | 196 | 0.653 | 51/70 | 21.749 | 6.219 | 10.304 | 2.519 |
| regime_objective_oracle | 177 | 0.590 | 70/70 | 32.282 | 3.178 | 5.463 | 1.575 |
| replay_oracle | 177 | 0.590 | 70/70 | 15.598 | 8.484 | 14.080 | 2.197 |

### Token Histograms

- `proxy_selector`: `{"evasive_left": 7, "evasive_right": 67, "lane_recover": 57, "maintain": 136, "nudge_left": 6, "nudge_right": 7, "slow_yield": 20}`
- `safety_value_baseline`: `{"crawl": 3, "evasive_left": 7, "evasive_right": 116, "lane_recover": 17, "maintain": 40, "nudge_left": 14, "nudge_right": 12, "slow_yield": 53, "stop": 38}`
- `regime_value`: `{"crawl": 47, "evasive_left": 7, "evasive_right": 115, "lane_recover": 15, "maintain": 39, "nudge_left": 1, "nudge_right": 5, "slow_yield": 44, "stop": 27}`
- `regime_objective_oracle`: `{"evasive_right": 69, "lane_recover": 4, "maintain": 186, "nudge_right": 4, "slow_yield": 16, "stop": 21}`
- `replay_oracle`: `{"crawl": 42, "evasive_left": 7, "evasive_right": 124, "lane_recover": 3, "maintain": 8, "nudge_left": 2, "nudge_right": 5, "slow_yield": 22, "stop": 87}`

