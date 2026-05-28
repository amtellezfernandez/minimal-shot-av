# nuPlan Replay-Calibrated Selector Evaluation

- Input replay JSON: `artifacts/corl2027/nuplan_public_replay_study_interaction250/replay.json`
- Split unit: `source_db_file`
- Train scenes: `200`
- Holdout scenes: `50`
- Near-miss threshold: `1.00 m`

## Train

| Selector | Replay-infeasible | Rate | Recoverable fixed | Recovery rate | Progress | Entropy |
|---|---:|---:|---:|---:|---:|---:|
| proxy_selector | 129 | 0.645 | 0/88 | 0.000 | 31.468 | 2.131 |
| replay_calibrated | 44 | 0.220 | 85/88 | 0.966 | 23.119 | 2.279 |
| replay_oracle | 41 | 0.205 | 88/88 | 1.000 | 9.935 | 2.154 |

### Token Histograms

- `proxy_selector`: `{"evasive_left": 40, "lane_recover": 7, "maintain": 91, "nudge_left": 10, "nudge_right": 24, "slow_yield": 28}`
- `replay_calibrated`: `{"crawl": 24, "evasive_left": 2, "lane_recover": 18, "maintain": 58, "nudge_left": 3, "nudge_right": 4, "slow_yield": 76, "stop": 15}`
- `replay_oracle`: `{"crawl": 34, "evasive_left": 14, "lane_recover": 8, "maintain": 16, "slow_yield": 38, "stop": 90}`

## Holdout

| Selector | Replay-infeasible | Rate | Recoverable fixed | Recovery rate | Progress | Entropy |
|---|---:|---:|---:|---:|---:|---:|
| proxy_selector | 10 | 0.200 | 0/7 | 0.000 | 15.539 | 1.697 |
| replay_calibrated | 3 | 0.060 | 7/7 | 1.000 | 12.595 | 1.697 |
| replay_oracle | 3 | 0.060 | 7/7 | 1.000 | 11.055 | 0.981 |

### Token Histograms

- `proxy_selector`: `{"lane_recover": 21, "maintain": 19, "nudge_left": 7, "slow_yield": 3}`
- `replay_calibrated`: `{"crawl": 3, "evasive_left": 21, "lane_recover": 19, "slow_yield": 7}`
- `replay_oracle`: `{"evasive_left": 40, "maintain": 3, "slow_yield": 6, "stop": 1}`

