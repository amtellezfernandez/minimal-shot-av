# nuPlan Failure-Gated Selector

- Input replay JSON: `artifacts/corl2027/nuplan_public_replay_study_interaction1000_expert/replay.json`
- Safe selector: `fit_replay_calibrated_on_train_split`
- Fallback mode: `risk_constrained_progress`
- Fallback risk threshold: `0.300`
- Split unit: `source_db_file`
- Train scenes: `700`
- Holdout scenes: `300`
- Gate threshold: `0.500`
- Train positive rate: `0.261`
- Train gate accuracy: `0.940`

## Train

| Selector | Replay-infeasible | Rate | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|
| proxy_selector | 418 | 0.597 | 27.353 | 2.566 | 2.142 |
| safe_selector | 257 | 0.367 | 14.618 | 6.851 | 2.703 |
| oracle_failure_gate | 235 | 0.336 | 23.493 | 3.698 | 2.593 |
| progress_oracle_failure_gate | 232 | 0.331 | 23.796 | 3.618 | 2.562 |
| failure_gate | 236 | 0.337 | 22.705 | 3.910 | 2.636 |
| replay_oracle | 232 | 0.331 | 13.239 | 7.340 | 2.703 |

### Gate Diagnostics

| Metric | Value |
|---|---:|
| switch count | 225 |
| switch rate | 0.321 |
| true positive switches | 183 |
| missed avoidable failures | 0 |
| false positive switches | 42 |
| introduced failures | 1 |
| switch precision | 0.813 |
| switch recall | 1.000 |
| oracle switch count | 183 |

### Threshold Sweep

| Threshold | Replay-infeasible Rate | Progress | ADE3 | Switch Rate | Missed Avoidable |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.340 | 21.238 | 4.380 | 0.384 | 0 |
| 0.30 | 0.339 | 22.030 | 4.102 | 0.354 | 0 |
| 0.40 | 0.339 | 22.317 | 4.027 | 0.339 | 0 |
| 0.50 | 0.337 | 22.705 | 3.910 | 0.321 | 0 |
| 0.60 | 0.337 | 22.849 | 3.864 | 0.310 | 1 |
| 0.70 | 0.340 | 23.194 | 3.760 | 0.290 | 3 |
| 0.80 | 0.356 | 23.616 | 3.635 | 0.266 | 14 |

## Holdout

| Selector | Replay-infeasible | Rate | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|
| proxy_selector | 209 | 0.697 | 29.383 | 2.251 | 2.240 |
| safe_selector | 49 | 0.163 | 12.882 | 8.472 | 2.456 |
| oracle_failure_gate | 49 | 0.163 | 19.701 | 5.906 | 2.405 |
| progress_oracle_failure_gate | 49 | 0.163 | 21.114 | 5.297 | 2.354 |
| failure_gate | 110 | 0.367 | 23.588 | 4.279 | 2.543 |
| replay_oracle | 49 | 0.163 | 10.889 | 9.329 | 2.344 |

### Gate Diagnostics

| Metric | Value |
|---|---:|
| switch count | 147 |
| switch rate | 0.490 |
| true positive switches | 99 |
| missed avoidable failures | 61 |
| false positive switches | 48 |
| introduced failures | 0 |
| switch precision | 0.673 |
| switch recall | 0.619 |
| oracle switch count | 160 |

### Threshold Sweep

| Threshold | Replay-infeasible Rate | Progress | ADE3 | Switch Rate | Missed Avoidable |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.197 | 18.805 | 6.286 | 0.673 | 10 |
| 0.30 | 0.220 | 19.629 | 5.957 | 0.643 | 17 |
| 0.40 | 0.347 | 23.058 | 4.499 | 0.510 | 55 |
| 0.50 | 0.367 | 23.588 | 4.279 | 0.490 | 61 |
| 0.60 | 0.373 | 23.652 | 4.259 | 0.450 | 63 |
| 0.70 | 0.387 | 23.957 | 4.141 | 0.387 | 67 |
| 0.80 | 0.390 | 24.394 | 3.959 | 0.357 | 68 |

## Train-Selected Operating Points

| Policy | Threshold | Holdout Replay-infeasible Rate | Holdout Progress | Holdout ADE3 |
|---|---:|---:|---:|---:|
| safety_first | 0.60 | 0.373 | 23.652 | 4.259 |
| balanced_80pct_proxy_progress | 0.60 | 0.373 | 23.652 | 4.259 |
| utility_90pct_proxy_progress | 0.60 | 0.373 | 23.652 | 4.259 |

