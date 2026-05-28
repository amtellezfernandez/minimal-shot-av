# nuPlan Failure-Gated Selector

- Input replay JSON: `artifacts/corl2027/nuplan_public_replay_study_interaction1000_expert/replay.json`
- Safe selector: `fit_replay_value_on_train_split`
- Fallback mode: `score`
- Fallback risk threshold: `0.500`
- Split unit: `source_db_file`
- Train scenes: `845`
- Holdout scenes: `155`
- Gate threshold: `0.500`
- Train positive rate: `0.260`
- Train gate accuracy: `0.948`

## Train

| Selector | Replay-infeasible | Rate | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|
| proxy_selector | 540 | 0.639 | 28.784 | 2.529 | 2.224 |
| safe_selector | 337 | 0.399 | 24.033 | 4.312 | 2.395 |
| oracle_failure_gate | 320 | 0.379 | 25.004 | 3.809 | 2.541 |
| progress_oracle_failure_gate | 276 | 0.327 | 24.816 | 3.775 | 2.622 |
| failure_gate | 320 | 0.379 | 24.257 | 4.064 | 2.555 |
| replay_oracle | 276 | 0.327 | 13.159 | 8.004 | 2.710 |

### Gate Diagnostics

| Metric | Value |
|---|---:|
| switch count | 264 |
| switch rate | 0.312 |
| true positive switches | 220 |
| missed avoidable failures | 0 |
| false positive switches | 44 |
| introduced failures | 0 |
| switch precision | 0.833 |
| switch recall | 1.000 |
| oracle switch count | 220 |

### Threshold Sweep

| Threshold | Replay-infeasible Rate | Progress | ADE3 | Switch Rate | Missed Avoidable |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.379 | 24.132 | 4.106 | 0.335 | 0 |
| 0.30 | 0.379 | 24.168 | 4.092 | 0.327 | 0 |
| 0.40 | 0.379 | 24.217 | 4.076 | 0.320 | 0 |
| 0.50 | 0.379 | 24.257 | 4.064 | 0.312 | 0 |
| 0.60 | 0.379 | 24.299 | 4.052 | 0.309 | 0 |
| 0.70 | 0.380 | 24.413 | 4.017 | 0.299 | 1 |
| 0.80 | 0.391 | 24.814 | 3.880 | 0.267 | 10 |

## Holdout

| Selector | Replay-infeasible | Rate | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|
| proxy_selector | 87 | 0.561 | 23.482 | 2.159 | 2.253 |
| safe_selector | 7 | 0.045 | 11.209 | 6.963 | 2.308 |
| oracle_failure_gate | 5 | 0.032 | 11.504 | 6.713 | 2.430 |
| progress_oracle_failure_gate | 5 | 0.032 | 13.046 | 6.008 | 2.003 |
| failure_gate | 9 | 0.058 | 11.409 | 6.734 | 2.422 |
| replay_oracle | 5 | 0.032 | 9.125 | 7.570 | 2.218 |

### Gate Diagnostics

| Metric | Value |
|---|---:|
| switch count | 81 |
| switch rate | 0.523 |
| true positive switches | 78 |
| missed avoidable failures | 4 |
| false positive switches | 3 |
| introduced failures | 0 |
| switch precision | 0.963 |
| switch recall | 0.951 |
| oracle switch count | 82 |

### Threshold Sweep

| Threshold | Replay-infeasible Rate | Progress | ADE3 | Switch Rate | Missed Avoidable |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.052 | 11.396 | 6.750 | 0.645 | 3 |
| 0.30 | 0.058 | 11.409 | 6.734 | 0.581 | 4 |
| 0.40 | 0.058 | 11.409 | 6.734 | 0.529 | 4 |
| 0.50 | 0.058 | 11.409 | 6.734 | 0.523 | 4 |
| 0.60 | 0.058 | 11.409 | 6.734 | 0.516 | 4 |
| 0.70 | 0.058 | 11.409 | 6.734 | 0.516 | 4 |
| 0.80 | 0.077 | 11.843 | 6.544 | 0.484 | 7 |

## Train-Selected Operating Points

| Policy | Threshold | Holdout Replay-infeasible Rate | Holdout Progress | Holdout ADE3 |
|---|---:|---:|---:|---:|
| safety_first | 0.60 | 0.058 | 11.409 | 6.734 |
| balanced_80pct_proxy_progress | 0.60 | 0.058 | 11.409 | 6.734 |
| utility_90pct_proxy_progress | 0.60 | 0.058 | 11.409 | 6.734 |

