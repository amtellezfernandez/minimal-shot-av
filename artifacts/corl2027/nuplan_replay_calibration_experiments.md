# nuPlan Replay Calibration Experiments

- Input replay JSON: `artifacts/corl2027/nuplan_public_replay_study_interaction250/replay.json`
- Scene count: `250`
- DB-level split count: `5`
- Holdout fraction: `0.30`
- Near-miss threshold: `1.00 m`
- Expert deviation: `The replay artifact does not include expert_trajectory, so expert deviation is n/a.`

## Risk Weight Sweep

| lambda | Replay-infeasible rate | Progress | Recovery rate | Token entropy |
|---:|---:|---:|---:|---:|
| 0 | 0.621 +/- 0.150 | 28.672 +/- 2.795 | 0.000 +/- 0.000 | 1.485 +/- 0.175 |
| 5 | 0.621 +/- 0.150 | 28.619 +/- 2.793 | 0.000 +/- 0.000 | 1.546 +/- 0.216 |
| 10 | 0.589 +/- 0.164 | 28.045 +/- 2.630 | 0.102 +/- 0.144 | 1.732 +/- 0.292 |
| 20 | 0.477 +/- 0.163 | 26.391 +/- 2.603 | 0.448 +/- 0.300 | 1.869 +/- 0.348 |
| 40 | 0.430 +/- 0.208 | 24.080 +/- 3.288 | 0.595 +/- 0.356 | 1.692 +/- 0.167 |
| 80 | 0.423 +/- 0.226 | 22.590 +/- 3.879 | 0.646 +/- 0.383 | 1.578 +/- 0.227 |

## Ablations

| Model | Replay-infeasible rate | Progress | Recovery rate | Token entropy |
|---|---:|---:|---:|---:|
| proxy_selector | 0.621 +/- 0.150 | 28.645 +/- 2.777 | 0.000 +/- 0.000 | 1.485 +/- 0.166 |
| risk_only | 0.258 +/- 0.260 | 10.817 +/- 5.215 | 1.000 +/- 0.000 | 1.399 +/- 0.362 |
| proxy_plus_progress | 0.621 +/- 0.150 | 28.672 +/- 2.795 | 0.000 +/- 0.000 | 1.485 +/- 0.175 |
| proxy_plus_risk | 0.428 +/- 0.209 | 24.066 +/- 3.300 | 0.600 +/- 0.352 | 1.693 +/- 0.169 |
| proxy_plus_progress_plus_risk | 0.430 +/- 0.208 | 24.080 +/- 3.288 | 0.595 +/- 0.356 | 1.692 +/- 0.167 |
| replay_oracle | 0.201 +/- 0.261 | 11.076 +/- 2.954 | 1.000 +/- 0.000 | 2.089 +/- 0.258 |

## Token Shift

| Model | Token histogram across holdout splits |
|---|---|
| proxy_plus_progress | `{"evasive_left": 80, "lane_recover": 34, "maintain": 186, "nudge_left": 31, "nudge_right": 24, "slow_yield": 15}` |
| proxy_plus_progress_plus_risk | `{"crawl": 3, "evasive_left": 64, "lane_recover": 62, "maintain": 81, "nudge_left": 19, "nudge_right": 25, "slow_yield": 102, "stop": 14}` |
| proxy_plus_risk | `{"crawl": 3, "evasive_left": 65, "lane_recover": 60, "maintain": 81, "nudge_left": 19, "nudge_right": 26, "slow_yield": 102, "stop": 14}` |
| proxy_selector | `{"evasive_left": 80, "lane_recover": 33, "maintain": 185, "nudge_left": 33, "nudge_right": 24, "slow_yield": 15}` |
| replay_oracle | `{"crawl": 50, "evasive_left": 56, "lane_recover": 24, "maintain": 43, "slow_yield": 70, "stop": 127}` |
| risk_only | `{"crawl": 142, "evasive_left": 36, "maintain": 17, "slow_yield": 103, "stop": 72}` |
