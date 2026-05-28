# nuPlan Failure-Gated Selector

- Input replay JSON: `artifacts/corl2027/nuplan_public_replay_study_interaction1000_expert/replay.json`
- Safe selector: `fit_progress_oracle_student_on_train_split`
- Fallback mode: `score`
- Fallback risk threshold: `0.500`
- Split unit: `source_db_file`
- Train scenes: `700`
- Holdout scenes: `300`
- Gate threshold: `0.500`
- Train positive rate: `0.110`
- Train gate accuracy: `0.981`

## Train

| Selector | Replay-infeasible | Rate | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|
| proxy_selector | 418 | 0.597 | 27.353 | 2.566 | 2.142 |
| safe_selector | 359 | 0.513 | 24.846 | 3.409 | 2.203 |
| oracle_failure_gate | 341 | 0.487 | 25.539 | 3.142 | 2.359 |
| progress_oracle_failure_gate | 232 | 0.331 | 23.796 | 3.618 | 2.562 |
| failure_gate | 341 | 0.487 | 25.287 | 3.226 | 2.354 |
| replay_oracle | 232 | 0.331 | 13.239 | 7.340 | 2.703 |

### Gate Diagnostics

| Metric | Value |
|---|---:|
| switch count | 90 |
| switch rate | 0.129 |
| true positive switches | 77 |
| missed avoidable failures | 0 |
| false positive switches | 13 |
| introduced failures | 0 |
| switch precision | 0.856 |
| switch recall | 1.000 |
| oracle switch count | 77 |

### Threshold Sweep

| Threshold | Replay-infeasible Rate | Progress | ADE3 | Switch Rate | Missed Avoidable |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.487 | 25.022 | 3.311 | 0.137 | 0 |
| 0.30 | 0.487 | 25.069 | 3.295 | 0.136 | 0 |
| 0.40 | 0.487 | 25.069 | 3.295 | 0.136 | 0 |
| 0.50 | 0.487 | 25.287 | 3.226 | 0.129 | 0 |
| 0.60 | 0.487 | 25.369 | 3.193 | 0.123 | 0 |
| 0.70 | 0.489 | 25.439 | 3.166 | 0.119 | 1 |
| 0.80 | 0.496 | 25.524 | 3.139 | 0.109 | 6 |

## Holdout

| Selector | Replay-infeasible | Rate | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|
| proxy_selector | 209 | 0.697 | 29.383 | 2.251 | 2.240 |
| safe_selector | 159 | 0.530 | 22.707 | 4.548 | 2.426 |
| oracle_failure_gate | 133 | 0.443 | 25.734 | 3.411 | 2.518 |
| progress_oracle_failure_gate | 49 | 0.163 | 21.114 | 5.297 | 2.354 |
| failure_gate | 171 | 0.570 | 23.834 | 4.276 | 2.549 |
| replay_oracle | 49 | 0.163 | 10.889 | 9.329 | 2.344 |

### Gate Diagnostics

| Metric | Value |
|---|---:|
| switch count | 130 |
| switch rate | 0.433 |
| true positive switches | 59 |
| missed avoidable failures | 17 |
| false positive switches | 71 |
| introduced failures | 21 |
| switch precision | 0.454 |
| switch recall | 0.776 |
| oracle switch count | 76 |

### Threshold Sweep

| Threshold | Replay-infeasible Rate | Progress | ADE3 | Switch Rate | Missed Avoidable |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.533 | 22.693 | 4.545 | 0.493 | 3 |
| 0.30 | 0.530 | 22.753 | 4.530 | 0.483 | 4 |
| 0.40 | 0.540 | 23.072 | 4.450 | 0.463 | 8 |
| 0.50 | 0.570 | 23.834 | 4.276 | 0.433 | 17 |
| 0.60 | 0.580 | 24.924 | 3.917 | 0.400 | 20 |
| 0.70 | 0.580 | 26.221 | 3.432 | 0.363 | 20 |
| 0.80 | 0.583 | 27.618 | 2.875 | 0.287 | 28 |

## Train-Selected Operating Points

| Policy | Threshold | Holdout Replay-infeasible Rate | Holdout Progress | Holdout ADE3 |
|---|---:|---:|---:|---:|
| safety_first | 0.60 | 0.580 | 24.924 | 3.917 |
| balanced_80pct_proxy_progress | 0.60 | 0.580 | 24.924 | 3.917 |
| utility_90pct_proxy_progress | 0.60 | 0.580 | 24.924 | 3.917 |

