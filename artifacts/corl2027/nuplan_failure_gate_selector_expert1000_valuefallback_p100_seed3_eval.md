# nuPlan Failure-Gated Selector

- Input replay JSON: `artifacts/corl2027/nuplan_public_replay_study_interaction1000_expert/replay.json`
- Safe selector: `fit_replay_value_on_train_split`
- Fallback mode: `score`
- Fallback risk threshold: `0.500`
- Split unit: `source_db_file`
- Train scenes: `700`
- Holdout scenes: `300`
- Gate threshold: `0.500`
- Train positive rate: `0.383`
- Train gate accuracy: `0.923`

## Train

| Selector | Replay-infeasible | Rate | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|
| proxy_selector | 380 | 0.543 | 25.728 | 2.581 | 2.376 |
| safe_selector | 112 | 0.160 | 14.634 | 6.572 | 2.608 |
| oracle_failure_gate | 112 | 0.160 | 18.824 | 5.022 | 2.647 |
| progress_oracle_failure_gate | 104 | 0.149 | 20.155 | 4.433 | 2.667 |
| failure_gate | 112 | 0.160 | 16.991 | 5.685 | 2.645 |
| replay_oracle | 104 | 0.149 | 11.221 | 7.702 | 2.677 |

### Gate Diagnostics

| Metric | Value |
|---|---:|
| switch count | 322 |
| switch rate | 0.460 |
| true positive switches | 268 |
| missed avoidable failures | 0 |
| false positive switches | 54 |
| introduced failures | 0 |
| switch precision | 0.832 |
| switch recall | 1.000 |
| oracle switch count | 268 |

### Threshold Sweep

| Threshold | Replay-infeasible Rate | Progress | ADE3 | Switch Rate | Missed Avoidable |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.160 | 16.057 | 6.061 | 0.513 | 0 |
| 0.30 | 0.160 | 16.101 | 6.041 | 0.504 | 0 |
| 0.40 | 0.160 | 16.444 | 5.895 | 0.483 | 0 |
| 0.50 | 0.160 | 16.991 | 5.685 | 0.460 | 0 |
| 0.60 | 0.164 | 17.891 | 5.343 | 0.433 | 3 |
| 0.70 | 0.173 | 18.641 | 5.070 | 0.401 | 9 |
| 0.80 | 0.191 | 19.179 | 4.875 | 0.373 | 22 |

## Holdout

| Selector | Replay-infeasible | Rate | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|
| proxy_selector | 247 | 0.823 | 33.173 | 2.217 | 2.082 |
| safe_selector | 192 | 0.640 | 27.869 | 4.147 | 2.025 |
| oracle_failure_gate | 191 | 0.637 | 30.384 | 3.146 | 2.210 |
| progress_oracle_failure_gate | 177 | 0.590 | 29.610 | 3.394 | 2.285 |
| failure_gate | 191 | 0.637 | 29.483 | 3.484 | 2.159 |
| replay_oracle | 177 | 0.590 | 15.598 | 8.484 | 2.197 |

### Gate Diagnostics

| Metric | Value |
|---|---:|
| switch count | 74 |
| switch rate | 0.247 |
| true positive switches | 56 |
| missed avoidable failures | 0 |
| false positive switches | 18 |
| introduced failures | 0 |
| switch precision | 0.757 |
| switch recall | 1.000 |
| oracle switch count | 56 |

### Threshold Sweep

| Threshold | Replay-infeasible Rate | Progress | ADE3 | Switch Rate | Missed Avoidable |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.637 | 27.518 | 4.159 | 0.360 | 0 |
| 0.30 | 0.637 | 28.419 | 3.861 | 0.317 | 0 |
| 0.40 | 0.637 | 28.838 | 3.696 | 0.270 | 0 |
| 0.50 | 0.637 | 29.483 | 3.484 | 0.247 | 0 |
| 0.60 | 0.637 | 29.798 | 3.358 | 0.237 | 0 |
| 0.70 | 0.647 | 30.016 | 3.273 | 0.203 | 3 |
| 0.80 | 0.680 | 30.832 | 2.984 | 0.163 | 13 |

## Train-Selected Operating Points

| Policy | Threshold | Holdout Replay-infeasible Rate | Holdout Progress | Holdout ADE3 |
|---|---:|---:|---:|---:|
| safety_first | 0.50 | 0.637 | 29.483 | 3.484 |
| balanced_80pct_proxy_progress | 0.50 | 0.637 | 29.483 | 3.484 |
| utility_90pct_proxy_progress | 0.50 | 0.637 | 29.483 | 3.484 |

