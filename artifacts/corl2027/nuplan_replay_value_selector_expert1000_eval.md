# nuPlan Replay-Value Selector Evaluation

- Input replay JSON: `artifacts/corl2027/nuplan_public_replay_study_interaction1000_expert/replay.json`
- Split unit: `source_db_file`
- Train scenes: `700`
- Holdout scenes: `300`
- Near-miss threshold: `1.00 m`
- Objective: `-250.0 * replay_failure + 1.0 * progress - 2.0 * ADE3`

## Train

| Selector | Replay-infeasible | Rate | Recovery | Progress | ADE3 | FDE3 | Entropy |
|---|---:|---:|---:|---:|---:|---:|---:|
| proxy_selector | 418 | 0.597 | 0/186 | 27.353 | 2.566 | 4.651 | 2.142 |
| replay_value | 277 | 0.396 | 155/186 | 20.577 | 4.977 | 8.372 | 2.445 |
| replay_oracle | 232 | 0.331 | 186/186 | 13.239 | 7.340 | 12.214 | 2.703 |
| baseline_selector | 245 | 0.350 | 173/186 | 21.432 | 4.231 | 7.051 | 2.906 |

### Token Histograms

- `proxy_selector`: `{"evasive_left": 16, "evasive_right": 112, "lane_recover": 109, "maintain": 337, "nudge_left": 17, "nudge_right": 25, "slow_yield": 84}`
- `replay_value`: `{"crawl": 18, "evasive_left": 64, "evasive_right": 139, "lane_recover": 5, "maintain": 258, "nudge_left": 9, "nudge_right": 12, "slow_yield": 64, "stop": 131}`
- `replay_oracle`: `{"crawl": 73, "evasive_left": 94, "evasive_right": 158, "lane_recover": 21, "maintain": 53, "nudge_left": 7, "nudge_right": 18, "slow_yield": 71, "stop": 205}`
- `baseline_selector`: `{"crawl": 29, "evasive_left": 75, "evasive_right": 137, "lane_recover": 83, "maintain": 132, "nudge_left": 12, "nudge_right": 40, "slow_yield": 134, "stop": 58}`

## Holdout

| Selector | Replay-infeasible | Rate | Recovery | Progress | ADE3 | FDE3 | Entropy |
|---|---:|---:|---:|---:|---:|---:|---:|
| proxy_selector | 209 | 0.697 | 0/160 | 29.383 | 2.251 | 3.479 | 2.240 |
| replay_value | 50 | 0.167 | 160/160 | 15.004 | 7.884 | 12.878 | 2.260 |
| replay_oracle | 49 | 0.163 | 160/160 | 10.889 | 9.329 | 15.255 | 2.344 |
| baseline_selector | 49 | 0.163 | 160/160 | 19.474 | 5.933 | 9.457 | 2.179 |

### Token Histograms

- `proxy_selector`: `{"evasive_left": 83, "lane_recover": 20, "maintain": 98, "nudge_left": 6, "nudge_right": 55, "slow_yield": 38}`
- `replay_value`: `{"crawl": 16, "evasive_left": 38, "evasive_right": 8, "maintain": 52, "nudge_left": 2, "slow_yield": 75, "stop": 109}`
- `replay_oracle`: `{"crawl": 82, "evasive_left": 45, "evasive_right": 8, "lane_recover": 4, "maintain": 10, "nudge_left": 3, "slow_yield": 54, "stop": 94}`
- `baseline_selector`: `{"crawl": 86, "evasive_left": 56, "evasive_right": 13, "lane_recover": 3, "maintain": 48, "slow_yield": 94}`

