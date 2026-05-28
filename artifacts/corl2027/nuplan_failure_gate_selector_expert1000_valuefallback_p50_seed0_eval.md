# nuPlan Failure-Gated Selector

- Input replay JSON: `artifacts/corl2027/nuplan_public_replay_study_interaction1000_expert/replay.json`
- Safe selector: `fit_replay_value_on_train_split`
- Fallback mode: `score`
- Fallback risk threshold: `0.500`
- Split unit: `source_db_file`
- Train scenes: `700`
- Holdout scenes: `300`
- Gate threshold: `0.500`
- Train positive rate: `0.160`
- Train gate accuracy: `0.980`

## Train

| Selector | Replay-infeasible | Rate | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|
| proxy_selector | 418 | 0.597 | 27.353 | 2.566 | 2.142 |
| safe_selector | 338 | 0.483 | 27.216 | 2.942 | 2.032 |
| oracle_failure_gate | 306 | 0.437 | 25.456 | 3.108 | 2.417 |
| progress_oracle_failure_gate | 232 | 0.331 | 23.796 | 3.618 | 2.562 |
| failure_gate | 306 | 0.437 | 25.260 | 3.160 | 2.419 |
| replay_oracle | 232 | 0.331 | 13.239 | 7.340 | 2.703 |

### Gate Diagnostics

| Metric | Value |
|---|---:|
| switch count | 126 |
| switch rate | 0.180 |
| true positive switches | 112 |
| missed avoidable failures | 0 |
| false positive switches | 14 |
| introduced failures | 0 |
| switch precision | 0.889 |
| switch recall | 1.000 |
| oracle switch count | 112 |

### Threshold Sweep

| Threshold | Replay-infeasible Rate | Progress | ADE3 | Switch Rate | Missed Avoidable |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.437 | 25.167 | 3.187 | 0.193 | 0 |
| 0.30 | 0.437 | 25.220 | 3.164 | 0.189 | 0 |
| 0.40 | 0.437 | 25.235 | 3.169 | 0.184 | 0 |
| 0.50 | 0.437 | 25.260 | 3.160 | 0.180 | 0 |
| 0.60 | 0.437 | 25.274 | 3.153 | 0.177 | 0 |
| 0.70 | 0.437 | 25.349 | 3.135 | 0.171 | 0 |
| 0.80 | 0.447 | 25.435 | 3.115 | 0.159 | 7 |

## Holdout

| Selector | Replay-infeasible | Rate | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|
| proxy_selector | 209 | 0.697 | 29.383 | 2.251 | 2.240 |
| safe_selector | 140 | 0.467 | 27.166 | 3.568 | 2.109 |
| oracle_failure_gate | 130 | 0.433 | 26.393 | 3.396 | 2.479 |
| progress_oracle_failure_gate | 49 | 0.163 | 21.114 | 5.297 | 2.354 |
| failure_gate | 134 | 0.447 | 26.086 | 3.481 | 2.416 |
| replay_oracle | 49 | 0.163 | 10.889 | 9.329 | 2.344 |

### Gate Diagnostics

| Metric | Value |
|---|---:|
| switch count | 101 |
| switch rate | 0.337 |
| true positive switches | 75 |
| missed avoidable failures | 4 |
| false positive switches | 26 |
| introduced failures | 0 |
| switch precision | 0.743 |
| switch recall | 0.949 |
| oracle switch count | 79 |

### Threshold Sweep

| Threshold | Replay-infeasible Rate | Progress | ADE3 | Switch Rate | Missed Avoidable |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.447 | 26.086 | 3.481 | 0.357 | 4 |
| 0.30 | 0.447 | 26.086 | 3.481 | 0.337 | 4 |
| 0.40 | 0.447 | 26.086 | 3.481 | 0.337 | 4 |
| 0.50 | 0.447 | 26.086 | 3.481 | 0.337 | 4 |
| 0.60 | 0.447 | 26.086 | 3.481 | 0.337 | 4 |
| 0.70 | 0.463 | 26.194 | 3.428 | 0.313 | 9 |
| 0.80 | 0.463 | 26.194 | 3.428 | 0.280 | 9 |

## Train-Selected Operating Points

| Policy | Threshold | Holdout Replay-infeasible Rate | Holdout Progress | Holdout ADE3 |
|---|---:|---:|---:|---:|
| safety_first | 0.70 | 0.463 | 26.194 | 3.428 |
| balanced_80pct_proxy_progress | 0.70 | 0.463 | 26.194 | 3.428 |
| utility_90pct_proxy_progress | 0.70 | 0.463 | 26.194 | 3.428 |

