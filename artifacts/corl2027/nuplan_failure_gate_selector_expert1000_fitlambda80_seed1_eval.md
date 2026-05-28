# nuPlan Failure-Gated Selector

- Input replay JSON: `artifacts/corl2027/nuplan_public_replay_study_interaction1000_expert/replay.json`
- Safe selector: `fit_on_train_split`
- Split unit: `source_db_file`
- Train scenes: `845`
- Holdout scenes: `155`
- Gate threshold: `0.500`
- Train positive rate: `0.317`
- Train gate accuracy: `0.930`

## Train

| Selector | Replay-infeasible | Rate | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|
| proxy_selector | 547 | 0.647 | 28.472 | 2.388 | 2.411 |
| safe_selector | 281 | 0.333 | 18.971 | 5.649 | 2.888 |
| failure_gate | 280 | 0.331 | 22.783 | 4.370 | 2.628 |
| replay_oracle | 263 | 0.311 | 12.179 | 8.327 | 2.591 |

### Gate Diagnostics

| Metric | Value |
|---|---:|
| switch count | 325 |
| switch rate | 0.385 |
| true positive switches | 267 |
| missed avoidable failures | 1 |
| false positive switches | 58 |
| introduced failures | 0 |

### Threshold Sweep

| Threshold | Replay-infeasible Rate | Progress | ADE3 | Switch Rate | Missed Avoidable |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.330 | 20.806 | 5.126 | 0.488 | 0 |
| 0.30 | 0.330 | 21.364 | 4.917 | 0.447 | 0 |
| 0.40 | 0.330 | 22.190 | 4.603 | 0.415 | 0 |
| 0.50 | 0.331 | 22.783 | 4.370 | 0.385 | 1 |
| 0.60 | 0.340 | 23.152 | 4.218 | 0.362 | 8 |
| 0.70 | 0.346 | 23.419 | 4.129 | 0.343 | 13 |
| 0.80 | 0.366 | 23.933 | 3.981 | 0.303 | 30 |

## Holdout

| Selector | Replay-infeasible | Rate | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|
| proxy_selector | 80 | 0.516 | 25.180 | 2.929 | 1.595 |
| safe_selector | 38 | 0.245 | 17.253 | 4.372 | 2.551 |
| failure_gate | 57 | 0.368 | 23.086 | 3.587 | 1.727 |
| replay_oracle | 18 | 0.116 | 14.471 | 5.808 | 2.847 |

### Gate Diagnostics

| Metric | Value |
|---|---:|
| switch count | 45 |
| switch rate | 0.290 |
| true positive switches | 23 |
| missed avoidable failures | 21 |
| false positive switches | 22 |
| introduced failures | 0 |

### Threshold Sweep

| Threshold | Replay-infeasible Rate | Progress | ADE3 | Switch Rate | Missed Avoidable |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.316 | 22.431 | 3.809 | 0.348 | 13 |
| 0.30 | 0.342 | 22.818 | 3.705 | 0.323 | 17 |
| 0.40 | 0.355 | 22.952 | 3.646 | 0.303 | 19 |
| 0.50 | 0.368 | 23.086 | 3.587 | 0.290 | 21 |
| 0.60 | 0.381 | 23.219 | 3.529 | 0.265 | 23 |
| 0.70 | 0.400 | 23.723 | 3.314 | 0.226 | 26 |
| 0.80 | 0.400 | 24.052 | 3.178 | 0.200 | 26 |

## Train-Selected Operating Points

| Policy | Threshold | Holdout Replay-infeasible Rate | Holdout Progress | Holdout ADE3 |
|---|---:|---:|---:|---:|
| safety_first | 0.40 | 0.355 | 22.952 | 3.646 |
| balanced_80pct_proxy_progress | 0.50 | 0.368 | 23.086 | 3.587 |
| utility_90pct_proxy_progress | 0.40 | 0.355 | 22.952 | 3.646 |

