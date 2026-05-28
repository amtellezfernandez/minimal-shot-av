# nuPlan Replay Calibration Experiments

- Input replay JSON: `artifacts/corl2027/nuplan_public_replay_study_interaction1000_expert/replay.json`
- Scene count: `1000`
- DB-level split count: `5`
- Holdout fraction: `0.30`
- Near-miss threshold: `1.00 m`
- Expert deviation: `ADE_3s/FDE_3s are computed against logged ego future trajectory preserved in replay artifacts.`

## Risk Weight Sweep

| lambda | Replay-infeasible rate | Progress | ADE3 | FDE3 | Recovery rate | Token entropy |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.650 +/- 0.109 | 28.013 +/- 3.444 | 2.456 +/- 0.291 | 4.399 +/- 0.965 | 0.004 +/- 0.005 | 2.009 +/- 0.252 |
| 5 | 0.595 +/- 0.142 | 27.122 +/- 4.068 | 2.652 +/- 0.232 | 4.594 +/- 0.759 | 0.118 +/- 0.093 | 2.131 +/- 0.313 |
| 10 | 0.539 +/- 0.143 | 26.207 +/- 4.108 | 2.869 +/- 0.216 | 4.914 +/- 0.428 | 0.269 +/- 0.084 | 2.324 +/- 0.158 |
| 20 | 0.456 +/- 0.158 | 24.897 +/- 4.094 | 3.140 +/- 0.283 | 5.306 +/- 0.381 | 0.498 +/- 0.064 | 2.454 +/- 0.101 |
| 40 | 0.404 +/- 0.143 | 22.992 +/- 3.184 | 3.742 +/- 0.291 | 6.228 +/- 0.460 | 0.659 +/- 0.104 | 2.515 +/- 0.159 |
| 80 | 0.361 +/- 0.171 | 19.704 +/- 3.082 | 4.964 +/- 0.652 | 8.111 +/- 1.173 | 0.771 +/- 0.149 | 2.414 +/- 0.163 |

## Ablations

| Model | Replay-infeasible rate | Progress | ADE3 | FDE3 | Recovery rate | Token entropy |
|---|---:|---:|---:|---:|---:|---:|
| proxy_selector | 0.652 +/- 0.108 | 28.043 +/- 3.407 | 2.443 +/- 0.300 | 4.383 +/- 0.975 | 0.000 +/- 0.000 | 2.010 +/- 0.247 |
| risk_only | 0.279 +/- 0.210 | 10.647 +/- 1.817 | 8.345 +/- 1.273 | 13.708 +/- 2.221 | 0.989 +/- 0.017 | 2.125 +/- 0.026 |
| proxy_plus_progress | 0.650 +/- 0.109 | 28.013 +/- 3.444 | 2.456 +/- 0.291 | 4.399 +/- 0.965 | 0.004 +/- 0.005 | 2.009 +/- 0.252 |
| proxy_plus_risk | 0.404 +/- 0.143 | 22.890 +/- 3.137 | 3.780 +/- 0.297 | 6.287 +/- 0.494 | 0.659 +/- 0.104 | 2.507 +/- 0.164 |
| proxy_plus_progress_plus_risk | 0.404 +/- 0.143 | 22.992 +/- 3.184 | 3.742 +/- 0.291 | 6.228 +/- 0.460 | 0.659 +/- 0.104 | 2.515 +/- 0.159 |
| replay_oracle | 0.259 +/- 0.204 | 12.286 +/- 2.390 | 7.956 +/- 1.210 | 13.206 +/- 1.934 | 1.000 +/- 0.000 | 2.444 +/- 0.250 |

## Token Shift

| Model | Token histogram across holdout splits |
|---|---|
| proxy_plus_progress | `{"crawl": 2, "evasive_left": 103, "evasive_right": 73, "lane_recover": 124, "maintain": 550, "nudge_left": 44, "nudge_right": 146, "slow_yield": 168}` |
| proxy_plus_progress_plus_risk | `{"crawl": 98, "evasive_left": 84, "evasive_right": 144, "lane_recover": 133, "maintain": 248, "nudge_left": 51, "nudge_right": 142, "slow_yield": 271, "stop": 39}` |
| proxy_plus_risk | `{"crawl": 101, "evasive_left": 86, "evasive_right": 144, "lane_recover": 129, "maintain": 246, "nudge_left": 48, "nudge_right": 143, "slow_yield": 274, "stop": 39}` |
| proxy_selector | `{"evasive_left": 103, "evasive_right": 73, "lane_recover": 129, "maintain": 545, "nudge_left": 46, "nudge_right": 146, "slow_yield": 168}` |
| replay_oracle | `{"crawl": 236, "evasive_left": 116, "evasive_right": 146, "lane_recover": 32, "maintain": 81, "nudge_left": 12, "nudge_right": 35, "slow_yield": 163, "stop": 389}` |
| risk_only | `{"crawl": 354, "evasive_left": 119, "evasive_right": 139, "lane_recover": 4, "maintain": 8, "nudge_left": 31, "nudge_right": 1, "slow_yield": 211, "stop": 343}` |
