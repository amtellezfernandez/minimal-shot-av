# nuPlan Failure-Gated Selector

- Input replay JSON: `artifacts/corl2027/nuplan_public_replay_study_interaction1000_expert/replay.json`
- Safe selector: `fit_on_train_split`
- Fallback mode: `score`
- Fallback risk threshold: `0.500`
- Split unit: `source_db_file`
- Train scenes: `700`
- Holdout scenes: `300`
- Gate threshold: `0.500`
- Train positive rate: `0.246`
- Train gate accuracy: `0.934`

## Train

| Selector | Replay-infeasible | Rate | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|
| proxy_selector | 418 | 0.597 | 27.353 | 2.566 | 2.142 |
| safe_selector | 246 | 0.351 | 20.110 | 4.751 | 2.962 |
| oracle_failure_gate | 246 | 0.351 | 24.036 | 3.525 | 2.533 |
| progress_oracle_failure_gate | 232 | 0.331 | 23.796 | 3.618 | 2.562 |
| failure_gate | 247 | 0.353 | 23.264 | 3.752 | 2.581 |
| replay_oracle | 232 | 0.331 | 13.239 | 7.340 | 2.703 |

### Gate Diagnostics

| Metric | Value |
|---|---:|
| switch count | 216 |
| switch rate | 0.309 |
| true positive switches | 171 |
| missed avoidable failures | 1 |
| false positive switches | 45 |
| introduced failures | 0 |
| switch precision | 0.792 |
| switch recall | 0.994 |
| oracle switch count | 172 |

### Threshold Sweep

| Threshold | Replay-infeasible Rate | Progress | ADE3 | Switch Rate | Missed Avoidable |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.351 | 22.474 | 3.977 | 0.353 | 0 |
| 0.30 | 0.351 | 22.553 | 3.964 | 0.344 | 0 |
| 0.40 | 0.351 | 22.778 | 3.908 | 0.329 | 0 |
| 0.50 | 0.353 | 23.264 | 3.752 | 0.309 | 1 |
| 0.60 | 0.359 | 23.470 | 3.697 | 0.290 | 5 |
| 0.70 | 0.361 | 23.651 | 3.638 | 0.274 | 7 |
| 0.80 | 0.373 | 24.086 | 3.493 | 0.243 | 15 |

## Holdout

| Selector | Replay-infeasible | Rate | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|
| proxy_selector | 209 | 0.697 | 29.383 | 2.251 | 2.240 |
| safe_selector | 49 | 0.163 | 18.342 | 6.334 | 2.106 |
| oracle_failure_gate | 49 | 0.163 | 20.786 | 5.450 | 2.292 |
| progress_oracle_failure_gate | 49 | 0.163 | 21.114 | 5.297 | 2.354 |
| failure_gate | 72 | 0.240 | 21.883 | 5.032 | 2.300 |
| replay_oracle | 49 | 0.163 | 10.889 | 9.329 | 2.344 |

### Gate Diagnostics

| Metric | Value |
|---|---:|
| switch count | 158 |
| switch rate | 0.527 |
| true positive switches | 137 |
| missed avoidable failures | 23 |
| false positive switches | 21 |
| introduced failures | 0 |
| switch precision | 0.867 |
| switch recall | 0.856 |
| oracle switch count | 160 |

### Threshold Sweep

| Threshold | Replay-infeasible Rate | Progress | ADE3 | Switch Rate | Missed Avoidable |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.187 | 20.155 | 5.677 | 0.687 | 7 |
| 0.30 | 0.197 | 20.887 | 5.408 | 0.603 | 10 |
| 0.40 | 0.220 | 21.494 | 5.180 | 0.567 | 17 |
| 0.50 | 0.240 | 21.883 | 5.032 | 0.527 | 23 |
| 0.60 | 0.243 | 21.932 | 5.020 | 0.487 | 24 |
| 0.70 | 0.243 | 21.932 | 5.020 | 0.480 | 24 |
| 0.80 | 0.250 | 22.085 | 4.953 | 0.473 | 26 |

## Train-Selected Operating Points

| Policy | Threshold | Holdout Replay-infeasible Rate | Holdout Progress | Holdout ADE3 |
|---|---:|---:|---:|---:|
| safety_first | 0.40 | 0.220 | 21.494 | 5.180 |
| balanced_80pct_proxy_progress | 0.40 | 0.220 | 21.494 | 5.180 |
| utility_90pct_proxy_progress | 0.40 | 0.220 | 21.494 | 5.180 |

