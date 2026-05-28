# nuPlan Replay Calibration Experiments

- Input replay JSON: `artifacts/corl2027/nuplan_public_replay_study_interaction1000_expert/replay.json`
- Scene count: `1000`
- DB-level split count: `5`
- Holdout fraction: `0.30`
- Near-miss threshold: `1.00 m`
- Expert deviation: `ADE_3s/FDE_3s are computed against logged ego future trajectory preserved in replay artifacts.`
- Meta objective: `-250.0 * replay_failure + 1.0 * progress - 2.0 * ADE3`
- Meta lambda values: `[0.0, 20.0, 40.0, 80.0]`

## Risk Weight Sweep

| lambda | Replay-infeasible rate | Progress | ADE3 | FDE3 | Recovery rate | Token entropy |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.650 +/- 0.109 | 28.013 +/- 3.444 | 2.456 +/- 0.291 | 4.399 +/- 0.965 | 0.004 +/- 0.005 | 2.009 +/- 0.252 |
| 20 | 0.456 +/- 0.158 | 24.897 +/- 4.094 | 3.140 +/- 0.283 | 5.306 +/- 0.381 | 0.498 +/- 0.064 | 2.454 +/- 0.101 |
| 40 | 0.404 +/- 0.143 | 22.992 +/- 3.184 | 3.742 +/- 0.291 | 6.228 +/- 0.460 | 0.659 +/- 0.104 | 2.515 +/- 0.159 |
| 80 | 0.361 +/- 0.171 | 19.704 +/- 3.082 | 4.964 +/- 0.652 | 8.111 +/- 1.173 | 0.771 +/- 0.149 | 2.414 +/- 0.163 |

## Constrained Risk Threshold Sweep

| risk threshold | Replay-infeasible rate | Progress | ADE3 | FDE3 | Recovery rate | Token entropy |
|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.279 +/- 0.210 | 11.626 +/- 1.495 | 7.942 +/- 0.959 | 13.025 +/- 1.746 | 0.989 +/- 0.017 | 2.172 +/- 0.059 |
| 0.1 | 0.289 +/- 0.217 | 12.529 +/- 0.911 | 7.602 +/- 1.158 | 12.515 +/- 1.993 | 0.957 +/- 0.067 | 2.255 +/- 0.072 |
| 0.2 | 0.315 +/- 0.221 | 13.437 +/- 0.910 | 7.321 +/- 1.229 | 12.113 +/- 2.048 | 0.871 +/- 0.115 | 2.333 +/- 0.099 |
| 0.3 | 0.302 +/- 0.218 | 14.677 +/- 1.809 | 6.879 +/- 1.457 | 11.430 +/- 2.308 | 0.889 +/- 0.114 | 2.545 +/- 0.180 |
| 0.4 | 0.317 +/- 0.218 | 15.466 +/- 2.258 | 6.652 +/- 1.433 | 11.121 +/- 2.169 | 0.833 +/- 0.114 | 2.629 +/- 0.246 |
| 0.5 | 0.326 +/- 0.217 | 16.199 +/- 2.628 | 6.488 +/- 1.362 | 10.889 +/- 1.992 | 0.804 +/- 0.115 | 2.603 +/- 0.218 |

## Exploratory Meta-Calibrated Lambda Policy

| Model | Replay-infeasible rate | Progress | ADE3 | FDE3 | Recovery rate | Token entropy |
|---|---:|---:|---:|---:|---:|---:|
| meta_lambda | 0.461 +/- 0.158 | 25.112 +/- 3.891 | 3.224 +/- 0.166 | 5.454 +/- 0.460 | 0.476 +/- 0.060 | 2.319 +/- 0.148 |
| meta_lambda_value | 0.368 +/- 0.167 | 22.057 +/- 3.041 | 4.255 +/- 0.709 | 7.129 +/- 0.986 | 0.730 +/- 0.156 | 2.465 +/- 0.186 |

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
