# nuPlan Regime-Value Selector

- Input replay JSON: `artifacts/corl2027/nuplan_public_replay_study_interaction1000_expert/replay.json`
- Split unit: `source_db_file`
- Train scenes: `700`
- Holdout scenes: `300`
- Marginal retention floor: `0.500`

## Regime Counts

- Train: `{"intervention_recovery_required": 92, "marginal_recovery_available": 173, "safe_proxy_keep": 272, "unrecoverable_no_safe_token": 163}`
- Holdout: `{"intervention_recovery_required": 46, "marginal_recovery_available": 35, "safe_proxy_keep": 101, "unrecoverable_no_safe_token": 118}`

## Train

| Selector | Replay-infeasible | Rate | Recovery | Progress | ADE3 | FDE3 | Entropy |
|---|---:|---:|---:|---:|---:|---:|---:|
| proxy_selector | 428 | 0.611 | 0/265 | 27.518 | 2.392 | 3.938 | 2.472 |
| safety_value_baseline | 168 | 0.240 | 260/265 | 17.214 | 5.820 | 9.326 | 2.863 |
| regime_value | 173 | 0.247 | 255/265 | 19.448 | 4.888 | 7.755 | 2.773 |
| regime_objective_oracle | 163 | 0.233 | 265/265 | 23.064 | 4.180 | 6.949 | 2.545 |
| replay_oracle | 163 | 0.233 | 265/265 | 13.043 | 7.658 | 12.482 | 2.571 |

### Token Histograms

- `proxy_selector`: `{"evasive_left": 92, "evasive_right": 106, "lane_recover": 93, "maintain": 258, "nudge_left": 11, "nudge_right": 64, "slow_yield": 76}`
- `safety_value_baseline`: `{"crawl": 106, "evasive_left": 125, "evasive_right": 154, "lane_recover": 16, "maintain": 20, "nudge_left": 25, "nudge_right": 56, "slow_yield": 117, "stop": 81}`
- `regime_value`: `{"crawl": 90, "evasive_left": 127, "evasive_right": 168, "lane_recover": 37, "maintain": 33, "nudge_left": 8, "nudge_right": 48, "slow_yield": 159, "stop": 30}`
- `regime_objective_oracle`: `{"crawl": 75, "evasive_left": 42, "evasive_right": 96, "lane_recover": 54, "maintain": 247, "nudge_right": 19, "slow_yield": 150, "stop": 17}`
- `replay_oracle`: `{"crawl": 92, "evasive_left": 118, "evasive_right": 159, "lane_recover": 12, "maintain": 31, "nudge_left": 7, "nudge_right": 1, "slow_yield": 88, "stop": 192}`

## Holdout

| Selector | Replay-infeasible | Rate | Recovery | Progress | ADE3 | FDE3 | Entropy |
|---|---:|---:|---:|---:|---:|---:|---:|
| proxy_selector | 199 | 0.663 | 0/81 | 28.998 | 2.658 | 5.142 | 1.882 |
| safety_value_baseline | 127 | 0.423 | 73/81 | 15.951 | 6.275 | 10.495 | 2.652 |
| regime_value | 134 | 0.447 | 65/81 | 17.126 | 6.224 | 10.687 | 2.656 |
| regime_objective_oracle | 118 | 0.393 | 81/81 | 26.861 | 3.659 | 6.676 | 1.879 |
| replay_oracle | 118 | 0.393 | 81/81 | 11.346 | 8.587 | 14.628 | 2.613 |

### Token Histograms

- `proxy_selector`: `{"evasive_left": 7, "evasive_right": 6, "lane_recover": 36, "maintain": 177, "nudge_left": 12, "nudge_right": 16, "slow_yield": 46}`
- `safety_value_baseline`: `{"crawl": 56, "evasive_left": 44, "evasive_right": 5, "lane_recover": 6, "maintain": 7, "nudge_left": 23, "nudge_right": 19, "slow_yield": 99, "stop": 41}`
- `regime_value`: `{"crawl": 35, "evasive_left": 25, "evasive_right": 13, "lane_recover": 17, "maintain": 26, "nudge_left": 9, "nudge_right": 18, "slow_yield": 126, "stop": 31}`
- `regime_objective_oracle`: `{"crawl": 17, "evasive_right": 3, "lane_recover": 6, "maintain": 186, "nudge_left": 4, "nudge_right": 32, "slow_yield": 23, "stop": 29}`
- `replay_oracle`: `{"crawl": 63, "evasive_left": 21, "evasive_right": 7, "lane_recover": 13, "maintain": 32, "nudge_left": 3, "nudge_right": 17, "slow_yield": 37, "stop": 107}`

