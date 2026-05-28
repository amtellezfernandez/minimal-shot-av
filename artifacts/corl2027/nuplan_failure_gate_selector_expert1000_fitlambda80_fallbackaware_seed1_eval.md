# nuPlan Failure-Gated Selector

- Input replay JSON: `artifacts/corl2027/nuplan_public_replay_study_interaction1000_expert/replay.json`
- Safe selector: `fit_on_train_split`
- Fallback mode: `score`
- Fallback risk threshold: `0.500`
- Split unit: `source_db_file`
- Train scenes: `845`
- Holdout scenes: `155`
- Gate threshold: `0.500`
- Train positive rate: `0.317`
- Train gate accuracy: `0.940`

## Train

| Selector | Replay-infeasible | Rate | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|
| proxy_selector | 547 | 0.647 | 28.472 | 2.388 | 2.411 |
| safe_selector | 281 | 0.333 | 18.971 | 5.649 | 2.888 |
| oracle_failure_gate | 279 | 0.330 | 23.445 | 4.167 | 2.619 |
| progress_oracle_failure_gate | 263 | 0.311 | 23.449 | 4.146 | 2.677 |
| failure_gate | 280 | 0.331 | 22.835 | 4.345 | 2.626 |
| replay_oracle | 263 | 0.311 | 12.179 | 8.327 | 2.591 |

### Gate Diagnostics

| Metric | Value |
|---|---:|
| switch count | 317 |
| switch rate | 0.375 |
| true positive switches | 267 |
| missed avoidable failures | 1 |
| false positive switches | 50 |
| introduced failures | 0 |
| switch precision | 0.842 |
| switch recall | 0.996 |
| oracle switch count | 268 |

### Threshold Sweep

| Threshold | Replay-infeasible Rate | Progress | ADE3 | Switch Rate | Missed Avoidable |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.330 | 21.018 | 5.037 | 0.451 | 0 |
| 0.30 | 0.330 | 21.505 | 4.864 | 0.428 | 0 |
| 0.40 | 0.330 | 22.194 | 4.594 | 0.398 | 0 |
| 0.50 | 0.331 | 22.835 | 4.345 | 0.375 | 1 |
| 0.60 | 0.337 | 23.167 | 4.222 | 0.355 | 6 |
| 0.70 | 0.344 | 23.357 | 4.161 | 0.337 | 12 |
| 0.80 | 0.355 | 23.716 | 4.036 | 0.315 | 21 |

## Holdout

| Selector | Replay-infeasible | Rate | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|
| proxy_selector | 80 | 0.516 | 25.180 | 2.929 | 1.595 |
| safe_selector | 38 | 0.245 | 17.253 | 4.372 | 2.551 |
| oracle_failure_gate | 36 | 0.232 | 22.330 | 3.639 | 1.888 |
| progress_oracle_failure_gate | 18 | 0.116 | 20.497 | 3.988 | 2.424 |
| failure_gate | 52 | 0.335 | 22.670 | 3.712 | 1.833 |
| replay_oracle | 18 | 0.116 | 14.471 | 5.808 | 2.847 |

### Gate Diagnostics

| Metric | Value |
|---|---:|
| switch count | 47 |
| switch rate | 0.303 |
| true positive switches | 28 |
| missed avoidable failures | 16 |
| false positive switches | 19 |
| introduced failures | 0 |
| switch precision | 0.596 |
| switch recall | 0.636 |
| oracle switch count | 44 |

### Threshold Sweep

| Threshold | Replay-infeasible Rate | Progress | ADE3 | Switch Rate | Missed Avoidable |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.290 | 21.597 | 4.120 | 0.387 | 9 |
| 0.30 | 0.323 | 22.536 | 3.771 | 0.329 | 14 |
| 0.40 | 0.329 | 22.603 | 3.741 | 0.323 | 15 |
| 0.50 | 0.335 | 22.670 | 3.712 | 0.303 | 16 |
| 0.60 | 0.348 | 22.817 | 3.652 | 0.290 | 18 |
| 0.70 | 0.361 | 22.950 | 3.594 | 0.245 | 20 |
| 0.80 | 0.400 | 23.438 | 3.420 | 0.194 | 26 |

## Train-Selected Operating Points

| Policy | Threshold | Holdout Replay-infeasible Rate | Holdout Progress | Holdout ADE3 |
|---|---:|---:|---:|---:|
| safety_first | 0.40 | 0.329 | 22.603 | 3.741 |
| balanced_80pct_proxy_progress | 0.50 | 0.335 | 22.670 | 3.712 |
| utility_90pct_proxy_progress | 0.40 | 0.329 | 22.603 | 3.741 |

