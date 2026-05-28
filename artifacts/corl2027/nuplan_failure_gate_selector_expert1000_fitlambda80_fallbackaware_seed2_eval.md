# nuPlan Failure-Gated Selector

- Input replay JSON: `artifacts/corl2027/nuplan_public_replay_study_interaction1000_expert/replay.json`
- Safe selector: `fit_on_train_split`
- Fallback mode: `score`
- Fallback risk threshold: `0.500`
- Split unit: `source_db_file`
- Train scenes: `700`
- Holdout scenes: `300`
- Gate threshold: `0.500`
- Train positive rate: `0.367`
- Train gate accuracy: `0.906`

## Train

| Selector | Replay-infeasible | Rate | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|
| proxy_selector | 428 | 0.611 | 27.518 | 2.392 | 2.472 |
| safe_selector | 171 | 0.244 | 17.642 | 5.699 | 2.742 |
| oracle_failure_gate | 171 | 0.244 | 21.902 | 4.326 | 2.633 |
| progress_oracle_failure_gate | 163 | 0.233 | 22.041 | 4.279 | 2.656 |
| failure_gate | 171 | 0.244 | 20.097 | 4.959 | 2.651 |
| replay_oracle | 163 | 0.233 | 13.043 | 7.658 | 2.571 |

### Gate Diagnostics

| Metric | Value |
|---|---:|
| switch count | 323 |
| switch rate | 0.461 |
| true positive switches | 257 |
| missed avoidable failures | 0 |
| false positive switches | 66 |
| introduced failures | 0 |
| switch precision | 0.796 |
| switch recall | 1.000 |
| oracle switch count | 257 |

### Threshold Sweep

| Threshold | Replay-infeasible Rate | Progress | ADE3 | Switch Rate | Missed Avoidable |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.244 | 19.055 | 5.354 | 0.519 | 0 |
| 0.30 | 0.244 | 19.200 | 5.290 | 0.506 | 0 |
| 0.40 | 0.244 | 19.550 | 5.169 | 0.487 | 0 |
| 0.50 | 0.244 | 20.097 | 4.959 | 0.461 | 0 |
| 0.60 | 0.253 | 21.462 | 4.445 | 0.419 | 6 |
| 0.70 | 0.260 | 21.784 | 4.322 | 0.397 | 11 |
| 0.80 | 0.283 | 22.361 | 4.122 | 0.360 | 27 |

## Holdout

| Selector | Replay-infeasible | Rate | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|
| proxy_selector | 199 | 0.663 | 28.998 | 2.658 | 1.882 |
| safe_selector | 154 | 0.513 | 19.109 | 5.465 | 2.575 |
| oracle_failure_gate | 143 | 0.477 | 26.232 | 3.567 | 2.234 |
| progress_oracle_failure_gate | 118 | 0.393 | 25.209 | 3.753 | 2.459 |
| failure_gate | 168 | 0.560 | 25.535 | 3.904 | 2.039 |
| replay_oracle | 118 | 0.393 | 11.346 | 8.587 | 2.613 |

### Gate Diagnostics

| Metric | Value |
|---|---:|
| switch count | 55 |
| switch rate | 0.183 |
| true positive switches | 31 |
| missed avoidable failures | 25 |
| false positive switches | 24 |
| introduced failures | 0 |
| switch precision | 0.564 |
| switch recall | 0.554 |
| oracle switch count | 56 |

### Threshold Sweep

| Threshold | Replay-infeasible Rate | Progress | ADE3 | Switch Rate | Missed Avoidable |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.503 | 22.731 | 4.880 | 0.347 | 7 |
| 0.30 | 0.527 | 24.110 | 4.460 | 0.257 | 15 |
| 0.40 | 0.547 | 24.976 | 4.108 | 0.210 | 21 |
| 0.50 | 0.560 | 25.535 | 3.904 | 0.183 | 25 |
| 0.60 | 0.577 | 26.309 | 3.608 | 0.153 | 30 |
| 0.70 | 0.580 | 26.855 | 3.394 | 0.133 | 31 |
| 0.80 | 0.580 | 27.227 | 3.238 | 0.103 | 31 |

## Train-Selected Operating Points

| Policy | Threshold | Holdout Replay-infeasible Rate | Holdout Progress | Holdout ADE3 |
|---|---:|---:|---:|---:|
| safety_first | 0.50 | 0.560 | 25.535 | 3.904 |
| balanced_80pct_proxy_progress | 0.80 | 0.580 | 27.227 | 3.238 |
| utility_90pct_proxy_progress | 0.50 | 0.560 | 25.535 | 3.904 |

