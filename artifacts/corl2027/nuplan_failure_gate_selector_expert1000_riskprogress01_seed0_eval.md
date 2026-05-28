# nuPlan Failure-Gated Selector

- Input replay JSON: `artifacts/corl2027/nuplan_public_replay_study_interaction1000_expert/replay.json`
- Safe selector: `fit_replay_calibrated_on_train_split`
- Fallback mode: `risk_constrained_progress`
- Fallback risk threshold: `0.100`
- Split unit: `source_db_file`
- Train scenes: `700`
- Holdout scenes: `300`
- Gate threshold: `0.500`
- Train positive rate: `0.261`
- Train gate accuracy: `0.930`

## Train

| Selector | Replay-infeasible | Rate | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|
| proxy_selector | 418 | 0.597 | 27.353 | 2.566 | 2.142 |
| safe_selector | 239 | 0.341 | 11.786 | 7.733 | 2.719 |
| oracle_failure_gate | 235 | 0.336 | 23.011 | 3.903 | 2.589 |
| progress_oracle_failure_gate | 232 | 0.331 | 23.796 | 3.618 | 2.562 |
| failure_gate | 236 | 0.337 | 21.971 | 4.188 | 2.630 |
| replay_oracle | 232 | 0.331 | 13.239 | 7.340 | 2.703 |

### Gate Diagnostics

| Metric | Value |
|---|---:|
| switch count | 232 |
| switch rate | 0.331 |
| true positive switches | 183 |
| missed avoidable failures | 0 |
| false positive switches | 49 |
| introduced failures | 1 |
| switch precision | 0.789 |
| switch recall | 1.000 |
| oracle switch count | 183 |

### Threshold Sweep

| Threshold | Replay-infeasible Rate | Progress | ADE3 | Switch Rate | Missed Avoidable |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.339 | 20.020 | 4.863 | 0.394 | 0 |
| 0.30 | 0.339 | 21.116 | 4.472 | 0.364 | 0 |
| 0.40 | 0.339 | 21.490 | 4.339 | 0.350 | 0 |
| 0.50 | 0.337 | 21.971 | 4.188 | 0.331 | 0 |
| 0.60 | 0.336 | 22.373 | 4.084 | 0.311 | 0 |
| 0.70 | 0.339 | 22.723 | 3.962 | 0.291 | 2 |
| 0.80 | 0.361 | 23.417 | 3.733 | 0.260 | 18 |

## Holdout

| Selector | Replay-infeasible | Rate | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|
| proxy_selector | 209 | 0.697 | 29.383 | 2.251 | 2.240 |
| safe_selector | 49 | 0.163 | 12.028 | 8.814 | 2.195 |
| oracle_failure_gate | 49 | 0.163 | 19.458 | 6.027 | 2.418 |
| progress_oracle_failure_gate | 49 | 0.163 | 21.114 | 5.297 | 2.354 |
| failure_gate | 116 | 0.387 | 23.364 | 4.360 | 2.468 |
| replay_oracle | 49 | 0.163 | 10.889 | 9.329 | 2.344 |

### Gate Diagnostics

| Metric | Value |
|---|---:|
| switch count | 132 |
| switch rate | 0.440 |
| true positive switches | 93 |
| missed avoidable failures | 67 |
| false positive switches | 39 |
| introduced failures | 0 |
| switch precision | 0.705 |
| switch recall | 0.581 |
| oracle switch count | 160 |

### Threshold Sweep

| Threshold | Replay-infeasible Rate | Progress | ADE3 | Switch Rate | Missed Avoidable |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.200 | 18.518 | 6.397 | 0.663 | 11 |
| 0.30 | 0.263 | 20.224 | 5.690 | 0.580 | 30 |
| 0.40 | 0.360 | 22.873 | 4.551 | 0.473 | 59 |
| 0.50 | 0.387 | 23.364 | 4.360 | 0.440 | 67 |
| 0.60 | 0.390 | 23.414 | 4.347 | 0.433 | 68 |
| 0.70 | 0.390 | 23.414 | 4.347 | 0.433 | 68 |
| 0.80 | 0.390 | 23.728 | 4.210 | 0.410 | 68 |

## Train-Selected Operating Points

| Policy | Threshold | Holdout Replay-infeasible Rate | Holdout Progress | Holdout ADE3 |
|---|---:|---:|---:|---:|
| safety_first | 0.60 | 0.390 | 23.414 | 4.347 |
| balanced_80pct_proxy_progress | 0.60 | 0.390 | 23.414 | 4.347 |
| utility_90pct_proxy_progress | 0.60 | 0.390 | 23.414 | 4.347 |

