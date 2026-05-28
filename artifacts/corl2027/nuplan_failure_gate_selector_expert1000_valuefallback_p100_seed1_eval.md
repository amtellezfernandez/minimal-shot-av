# nuPlan Failure-Gated Selector

- Input replay JSON: `artifacts/corl2027/nuplan_public_replay_study_interaction1000_expert/replay.json`
- Safe selector: `fit_replay_value_on_train_split`
- Fallback mode: `score`
- Fallback risk threshold: `0.500`
- Split unit: `source_db_file`
- Train scenes: `845`
- Holdout scenes: `155`
- Gate threshold: `0.500`
- Train positive rate: `0.311`
- Train gate accuracy: `0.949`

## Train

| Selector | Replay-infeasible | Rate | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|
| proxy_selector | 547 | 0.647 | 28.472 | 2.388 | 2.411 |
| safe_selector | 299 | 0.354 | 19.316 | 5.837 | 2.507 |
| oracle_failure_gate | 284 | 0.336 | 22.740 | 4.489 | 2.559 |
| progress_oracle_failure_gate | 263 | 0.311 | 23.449 | 4.146 | 2.677 |
| failure_gate | 285 | 0.337 | 22.064 | 4.730 | 2.532 |
| replay_oracle | 263 | 0.311 | 12.179 | 8.327 | 2.591 |

### Gate Diagnostics

| Metric | Value |
|---|---:|
| switch count | 304 |
| switch rate | 0.360 |
| true positive switches | 262 |
| missed avoidable failures | 1 |
| false positive switches | 42 |
| introduced failures | 0 |
| switch precision | 0.862 |
| switch recall | 0.996 |
| oracle switch count | 263 |

### Threshold Sweep

| Threshold | Replay-infeasible Rate | Progress | ADE3 | Switch Rate | Missed Avoidable |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.336 | 20.222 | 5.422 | 0.427 | 0 |
| 0.30 | 0.336 | 20.783 | 5.222 | 0.404 | 0 |
| 0.40 | 0.336 | 21.179 | 5.075 | 0.389 | 0 |
| 0.50 | 0.337 | 22.064 | 4.730 | 0.360 | 1 |
| 0.60 | 0.343 | 22.337 | 4.618 | 0.343 | 6 |
| 0.70 | 0.347 | 22.610 | 4.511 | 0.331 | 9 |
| 0.80 | 0.359 | 23.037 | 4.343 | 0.305 | 19 |

## Holdout

| Selector | Replay-infeasible | Rate | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|
| proxy_selector | 80 | 0.516 | 25.180 | 2.929 | 1.595 |
| safe_selector | 31 | 0.200 | 20.295 | 4.345 | 2.329 |
| oracle_failure_gate | 31 | 0.200 | 21.243 | 3.933 | 2.213 |
| progress_oracle_failure_gate | 18 | 0.116 | 20.497 | 3.988 | 2.424 |
| failure_gate | 50 | 0.323 | 22.699 | 3.644 | 2.061 |
| replay_oracle | 18 | 0.116 | 14.471 | 5.808 | 2.847 |

### Gate Diagnostics

| Metric | Value |
|---|---:|
| switch count | 48 |
| switch rate | 0.310 |
| true positive switches | 30 |
| missed avoidable failures | 19 |
| false positive switches | 18 |
| introduced failures | 0 |
| switch precision | 0.625 |
| switch recall | 0.612 |
| oracle switch count | 49 |

### Threshold Sweep

| Threshold | Replay-infeasible Rate | Progress | ADE3 | Switch Rate | Missed Avoidable |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.226 | 21.031 | 4.139 | 0.413 | 4 |
| 0.30 | 0.258 | 21.724 | 3.941 | 0.374 | 9 |
| 0.40 | 0.297 | 22.359 | 3.756 | 0.335 | 15 |
| 0.50 | 0.323 | 22.699 | 3.644 | 0.310 | 19 |
| 0.60 | 0.368 | 23.226 | 3.446 | 0.252 | 26 |
| 0.70 | 0.381 | 23.359 | 3.388 | 0.239 | 28 |
| 0.80 | 0.419 | 23.632 | 3.321 | 0.155 | 34 |

## Train-Selected Operating Points

| Policy | Threshold | Holdout Replay-infeasible Rate | Holdout Progress | Holdout ADE3 |
|---|---:|---:|---:|---:|
| safety_first | 0.40 | 0.297 | 22.359 | 3.756 |
| balanced_80pct_proxy_progress | 0.80 | 0.419 | 23.632 | 3.321 |
| utility_90pct_proxy_progress | 0.40 | 0.297 | 22.359 | 3.756 |

