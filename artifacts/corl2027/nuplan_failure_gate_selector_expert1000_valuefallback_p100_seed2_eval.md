# nuPlan Failure-Gated Selector

- Input replay JSON: `artifacts/corl2027/nuplan_public_replay_study_interaction1000_expert/replay.json`
- Safe selector: `fit_replay_value_on_train_split`
- Fallback mode: `score`
- Fallback risk threshold: `0.500`
- Split unit: `source_db_file`
- Train scenes: `700`
- Holdout scenes: `300`
- Gate threshold: `0.500`
- Train positive rate: `0.361`
- Train gate accuracy: `0.933`

## Train

| Selector | Replay-infeasible | Rate | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|
| proxy_selector | 428 | 0.611 | 27.518 | 2.392 | 2.472 |
| safe_selector | 175 | 0.250 | 15.967 | 6.630 | 2.570 |
| oracle_failure_gate | 175 | 0.250 | 20.905 | 4.800 | 2.598 |
| progress_oracle_failure_gate | 163 | 0.233 | 22.041 | 4.279 | 2.656 |
| failure_gate | 175 | 0.250 | 19.725 | 5.233 | 2.584 |
| replay_oracle | 163 | 0.233 | 13.043 | 7.658 | 2.571 |

### Gate Diagnostics

| Metric | Value |
|---|---:|
| switch count | 300 |
| switch rate | 0.429 |
| true positive switches | 253 |
| missed avoidable failures | 0 |
| false positive switches | 47 |
| introduced failures | 0 |
| switch precision | 0.843 |
| switch recall | 1.000 |
| oracle switch count | 253 |

### Threshold Sweep

| Threshold | Replay-infeasible Rate | Progress | ADE3 | Switch Rate | Missed Avoidable |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.250 | 17.955 | 5.903 | 0.501 | 0 |
| 0.30 | 0.250 | 18.484 | 5.694 | 0.474 | 0 |
| 0.40 | 0.250 | 18.824 | 5.577 | 0.459 | 0 |
| 0.50 | 0.250 | 19.725 | 5.233 | 0.429 | 0 |
| 0.60 | 0.259 | 20.550 | 4.908 | 0.400 | 6 |
| 0.70 | 0.261 | 20.757 | 4.834 | 0.386 | 8 |
| 0.80 | 0.279 | 21.359 | 4.591 | 0.354 | 20 |

## Holdout

| Selector | Replay-infeasible | Rate | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|
| proxy_selector | 199 | 0.663 | 28.998 | 2.658 | 1.882 |
| safe_selector | 142 | 0.473 | 24.191 | 4.339 | 2.086 |
| oracle_failure_gate | 140 | 0.467 | 25.874 | 3.637 | 2.265 |
| progress_oracle_failure_gate | 118 | 0.393 | 25.209 | 3.753 | 2.459 |
| failure_gate | 167 | 0.557 | 26.265 | 3.524 | 2.073 |
| replay_oracle | 118 | 0.393 | 11.346 | 8.587 | 2.613 |

### Gate Diagnostics

| Metric | Value |
|---|---:|
| switch count | 47 |
| switch rate | 0.157 |
| true positive switches | 32 |
| missed avoidable failures | 27 |
| false positive switches | 15 |
| introduced failures | 0 |
| switch precision | 0.681 |
| switch recall | 0.542 |
| oracle switch count | 59 |

### Threshold Sweep

| Threshold | Replay-infeasible Rate | Progress | ADE3 | Switch Rate | Missed Avoidable |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.520 | 25.646 | 3.730 | 0.233 | 14 |
| 0.30 | 0.530 | 25.757 | 3.685 | 0.197 | 19 |
| 0.40 | 0.543 | 26.003 | 3.598 | 0.183 | 23 |
| 0.50 | 0.557 | 26.265 | 3.524 | 0.157 | 27 |
| 0.60 | 0.587 | 27.179 | 3.272 | 0.113 | 36 |
| 0.70 | 0.587 | 27.212 | 3.255 | 0.107 | 36 |
| 0.80 | 0.590 | 27.578 | 3.108 | 0.083 | 37 |

## Train-Selected Operating Points

| Policy | Threshold | Holdout Replay-infeasible Rate | Holdout Progress | Holdout ADE3 |
|---|---:|---:|---:|---:|
| safety_first | 0.50 | 0.557 | 26.265 | 3.524 |
| balanced_80pct_proxy_progress | 0.50 | 0.557 | 26.265 | 3.524 |
| utility_90pct_proxy_progress | 0.50 | 0.557 | 26.265 | 3.524 |

