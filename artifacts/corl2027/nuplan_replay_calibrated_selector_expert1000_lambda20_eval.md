# nuPlan Replay-Calibrated Selector Evaluation

- Input replay JSON: `artifacts/corl2027/nuplan_public_replay_study_interaction1000_expert/replay.json`
- Split unit: `source_db_file`
- Train scenes: `700`
- Holdout scenes: `300`
- Near-miss threshold: `1.00 m`

## Train

| Selector | Replay-infeasible | Rate | Recovery | Progress | ADE3 | FDE3 | Entropy |
|---|---:|---:|---:|---:|---:|---:|---:|
| proxy_selector | 418 | 0.597 | 0/186 | 27.353 | 2.566 | 4.651 | 2.142 |
| replay_calibrated | 302 | 0.431 | 116/186 | 25.554 | 2.891 | 5.028 | 2.485 |
| replay_oracle | 232 | 0.331 | 186/186 | 13.239 | 7.340 | 12.214 | 2.703 |

### Token Histograms

- `proxy_selector`: `{"evasive_left": 16, "evasive_right": 112, "lane_recover": 109, "maintain": 337, "nudge_left": 17, "nudge_right": 25, "slow_yield": 84}`
- `replay_calibrated`: `{"crawl": 11, "evasive_left": 7, "evasive_right": 105, "lane_recover": 132, "maintain": 246, "nudge_left": 14, "nudge_right": 47, "slow_yield": 124, "stop": 14}`
- `replay_oracle`: `{"crawl": 73, "evasive_left": 94, "evasive_right": 158, "lane_recover": 21, "maintain": 53, "nudge_left": 7, "nudge_right": 18, "slow_yield": 71, "stop": 205}`

## Holdout

| Selector | Replay-infeasible | Rate | Recovery | Progress | ADE3 | FDE3 | Entropy |
|---|---:|---:|---:|---:|---:|---:|---:|
| proxy_selector | 209 | 0.697 | 0/160 | 29.383 | 2.251 | 3.479 | 2.240 |
| replay_calibrated | 115 | 0.383 | 94/160 | 25.429 | 3.563 | 5.481 | 2.531 |
| replay_oracle | 49 | 0.163 | 160/160 | 10.889 | 9.329 | 15.255 | 2.344 |

### Token Histograms

- `proxy_selector`: `{"evasive_left": 83, "lane_recover": 20, "maintain": 98, "nudge_left": 6, "nudge_right": 55, "slow_yield": 38}`
- `replay_calibrated`: `{"crawl": 26, "evasive_left": 43, "evasive_right": 13, "lane_recover": 7, "maintain": 80, "nudge_left": 4, "nudge_right": 41, "slow_yield": 86}`
- `replay_oracle`: `{"crawl": 82, "evasive_left": 45, "evasive_right": 8, "lane_recover": 4, "maintain": 10, "nudge_left": 3, "slow_yield": 54, "stop": 94}`

