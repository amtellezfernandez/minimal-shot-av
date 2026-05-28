# nuPlan Failure-Gated Selector

- Input replay JSON: `artifacts/corl2027/nuplan_public_replay_study_interaction1000_expert/replay.json`
- Safe selector: `fit_on_train_split`
- Split unit: `source_db_file`
- Train scenes: `845`
- Holdout scenes: `155`
- Gate threshold: `0.500`
- Train positive rate: `0.288`
- Train gate accuracy: `0.909`

## Train

| Selector | Replay-infeasible | Rate | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|
| proxy_selector | 540 | 0.639 | 28.784 | 2.529 | 2.224 |
| safe_selector | 298 | 0.353 | 20.815 | 5.038 | 2.961 |
| failure_gate | 299 | 0.354 | 24.395 | 3.874 | 2.584 |
| replay_oracle | 276 | 0.327 | 13.159 | 8.004 | 2.710 |

### Gate Diagnostics

| Metric | Value |
|---|---:|
| switch count | 316 |
| switch rate | 0.374 |
| true positive switches | 241 |
| missed avoidable failures | 2 |
| false positive switches | 75 |
| introduced failures | 0 |

### Threshold Sweep

| Threshold | Replay-infeasible Rate | Progress | ADE3 | Switch Rate | Missed Avoidable |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.353 | 22.807 | 4.378 | 0.515 | 0 |
| 0.30 | 0.353 | 23.412 | 4.194 | 0.445 | 0 |
| 0.40 | 0.353 | 23.906 | 4.025 | 0.404 | 0 |
| 0.50 | 0.354 | 24.395 | 3.874 | 0.374 | 2 |
| 0.60 | 0.362 | 24.826 | 3.760 | 0.344 | 9 |
| 0.70 | 0.376 | 25.220 | 3.630 | 0.308 | 21 |
| 0.80 | 0.402 | 25.697 | 3.485 | 0.260 | 43 |

## Holdout

| Selector | Replay-infeasible | Rate | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|
| proxy_selector | 87 | 0.561 | 23.482 | 2.159 | 2.253 |
| safe_selector | 47 | 0.303 | 18.140 | 3.707 | 2.249 |
| failure_gate | 53 | 0.342 | 19.410 | 3.259 | 2.255 |
| replay_oracle | 5 | 0.032 | 9.125 | 7.570 | 2.218 |

### Gate Diagnostics

| Metric | Value |
|---|---:|
| switch count | 64 |
| switch rate | 0.413 |
| true positive switches | 34 |
| missed avoidable failures | 6 |
| false positive switches | 30 |
| introduced failures | 0 |

### Threshold Sweep

| Threshold | Replay-infeasible Rate | Progress | ADE3 | Switch Rate | Missed Avoidable |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.329 | 19.129 | 3.379 | 0.703 | 4 |
| 0.30 | 0.329 | 19.129 | 3.379 | 0.671 | 4 |
| 0.40 | 0.329 | 19.129 | 3.379 | 0.445 | 4 |
| 0.50 | 0.342 | 19.410 | 3.259 | 0.413 | 6 |
| 0.60 | 0.374 | 19.889 | 3.054 | 0.368 | 11 |
| 0.70 | 0.381 | 19.955 | 3.025 | 0.232 | 12 |
| 0.80 | 0.406 | 20.579 | 2.826 | 0.187 | 16 |

## Train-Selected Operating Points

| Policy | Threshold | Holdout Replay-infeasible Rate | Holdout Progress | Holdout ADE3 |
|---|---:|---:|---:|---:|
| safety_first | 0.40 | 0.329 | 19.129 | 3.379 |
| balanced_80pct_proxy_progress | 0.40 | 0.329 | 19.129 | 3.379 |
| utility_90pct_proxy_progress | 0.40 | 0.329 | 19.129 | 3.379 |

