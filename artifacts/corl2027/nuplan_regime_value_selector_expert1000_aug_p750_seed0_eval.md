# nuPlan Regime-Value Selector

- Input replay JSON: `artifacts/corl2027/nuplan_public_replay_study_interaction1000_expert/replay.json`
- Split unit: `source_db_file`
- Train scenes: `700`
- Holdout scenes: `300`
- Marginal retention floor: `0.500`

## Regime Counts

- Train: `{"intervention_recovery_required": 63, "marginal_recovery_available": 123, "safe_proxy_keep": 282, "unrecoverable_no_safe_token": 232}`
- Holdout: `{"intervention_recovery_required": 75, "marginal_recovery_available": 85, "safe_proxy_keep": 91, "unrecoverable_no_safe_token": 49}`

## Train

| Selector | Replay-infeasible | Rate | Recovery | Progress | ADE3 | FDE3 | Entropy |
|---|---:|---:|---:|---:|---:|---:|---:|
| proxy_selector | 418 | 0.597 | 0/186 | 27.353 | 2.566 | 4.651 | 2.142 |
| safety_value_baseline | 237 | 0.339 | 181/186 | 19.974 | 4.679 | 7.683 | 2.922 |
| regime_value | 238 | 0.340 | 180/186 | 17.705 | 5.456 | 9.060 | 2.624 |
| regime_objective_oracle | 232 | 0.331 | 186/186 | 25.521 | 3.477 | 6.185 | 2.216 |
| replay_oracle | 232 | 0.331 | 186/186 | 13.239 | 7.340 | 12.214 | 2.703 |

### Token Histograms

- `proxy_selector`: `{"evasive_left": 16, "evasive_right": 112, "lane_recover": 109, "maintain": 337, "nudge_left": 17, "nudge_right": 25, "slow_yield": 84}`
- `safety_value_baseline`: `{"crawl": 31, "evasive_left": 93, "evasive_right": 152, "lane_recover": 61, "maintain": 105, "nudge_left": 11, "nudge_right": 38, "slow_yield": 112, "stop": 97}`
- `regime_value`: `{"crawl": 32, "evasive_left": 3, "evasive_right": 233, "lane_recover": 53, "maintain": 69, "nudge_left": 53, "nudge_right": 11, "slow_yield": 98, "stop": 148}`
- `regime_objective_oracle`: `{"crawl": 17, "evasive_left": 2, "evasive_right": 86, "lane_recover": 55, "maintain": 364, "nudge_left": 4, "nudge_right": 42, "slow_yield": 84, "stop": 46}`
- `replay_oracle`: `{"crawl": 73, "evasive_left": 94, "evasive_right": 158, "lane_recover": 21, "maintain": 53, "nudge_left": 7, "nudge_right": 18, "slow_yield": 71, "stop": 205}`

## Holdout

| Selector | Replay-infeasible | Rate | Recovery | Progress | ADE3 | FDE3 | Entropy |
|---|---:|---:|---:|---:|---:|---:|---:|
| proxy_selector | 209 | 0.697 | 0/160 | 29.383 | 2.251 | 3.479 | 2.240 |
| safety_value_baseline | 49 | 0.163 | 160/160 | 17.055 | 6.705 | 10.792 | 2.645 |
| regime_value | 60 | 0.200 | 149/160 | 17.536 | 6.591 | 10.618 | 2.352 |
| regime_objective_oracle | 49 | 0.163 | 160/160 | 21.128 | 5.299 | 8.458 | 2.342 |
| replay_oracle | 49 | 0.163 | 160/160 | 10.889 | 9.329 | 15.255 | 2.344 |

### Token Histograms

- `proxy_selector`: `{"evasive_left": 83, "lane_recover": 20, "maintain": 98, "nudge_left": 6, "nudge_right": 55, "slow_yield": 38}`
- `safety_value_baseline`: `{"crawl": 72, "evasive_left": 46, "evasive_right": 8, "lane_recover": 8, "maintain": 17, "nudge_left": 8, "nudge_right": 16, "slow_yield": 95, "stop": 30}`
- `regime_value`: `{"crawl": 7, "evasive_left": 16, "evasive_right": 13, "lane_recover": 6, "maintain": 48, "nudge_left": 15, "slow_yield": 104, "stop": 91}`
- `regime_objective_oracle`: `{"crawl": 75, "evasive_left": 40, "evasive_right": 13, "lane_recover": 5, "maintain": 69, "nudge_right": 9, "slow_yield": 89}`
- `replay_oracle`: `{"crawl": 82, "evasive_left": 45, "evasive_right": 8, "lane_recover": 4, "maintain": 10, "nudge_left": 3, "slow_yield": 54, "stop": 94}`

