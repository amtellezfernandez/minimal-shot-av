# nuPlan Failure-Gated Selector

- Input replay JSON: `artifacts/corl2027/nuplan_public_replay_study_interaction1000_expert/replay.json`
- Safe selector: `fit_on_train_split`
- Split unit: `source_db_file`
- Train scenes: `700`
- Holdout scenes: `300`
- Gate threshold: `0.500`
- Train positive rate: `0.387`
- Train gate accuracy: `0.904`

## Train

| Selector | Replay-infeasible | Rate | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|
| proxy_selector | 380 | 0.543 | 25.728 | 2.581 | 2.376 |
| safe_selector | 109 | 0.156 | 15.974 | 5.741 | 2.791 |
| failure_gate | 109 | 0.156 | 18.394 | 5.048 | 2.640 |
| replay_oracle | 104 | 0.149 | 11.221 | 7.702 | 2.677 |

### Gate Diagnostics

| Metric | Value |
|---|---:|
| switch count | 338 |
| switch rate | 0.483 |
| true positive switches | 271 |
| missed avoidable failures | 0 |
| false positive switches | 67 |
| introduced failures | 0 |

### Threshold Sweep

| Threshold | Replay-infeasible Rate | Progress | ADE3 | Switch Rate | Missed Avoidable |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.156 | 16.885 | 5.568 | 0.646 | 0 |
| 0.30 | 0.156 | 17.405 | 5.394 | 0.563 | 0 |
| 0.40 | 0.156 | 17.715 | 5.295 | 0.521 | 0 |
| 0.50 | 0.156 | 18.394 | 5.048 | 0.483 | 0 |
| 0.60 | 0.160 | 19.264 | 4.731 | 0.431 | 3 |
| 0.70 | 0.187 | 20.178 | 4.436 | 0.373 | 22 |
| 0.80 | 0.219 | 20.896 | 4.220 | 0.340 | 44 |

## Holdout

| Selector | Replay-infeasible | Rate | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|
| proxy_selector | 247 | 0.823 | 33.173 | 2.217 | 2.082 |
| safe_selector | 193 | 0.643 | 25.487 | 4.921 | 2.536 |
| failure_gate | 197 | 0.657 | 28.848 | 3.724 | 2.173 |
| replay_oracle | 177 | 0.590 | 15.598 | 8.484 | 2.197 |

### Gate Diagnostics

| Metric | Value |
|---|---:|
| switch count | 107 |
| switch rate | 0.357 |
| true positive switches | 50 |
| missed avoidable failures | 4 |
| false positive switches | 57 |
| introduced failures | 0 |

### Threshold Sweep

| Threshold | Replay-infeasible Rate | Progress | ADE3 | Switch Rate | Missed Avoidable |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.657 | 26.895 | 4.375 | 0.513 | 4 |
| 0.30 | 0.657 | 27.851 | 4.059 | 0.443 | 4 |
| 0.40 | 0.657 | 28.380 | 3.858 | 0.393 | 4 |
| 0.50 | 0.657 | 28.848 | 3.724 | 0.357 | 4 |
| 0.60 | 0.673 | 29.053 | 3.704 | 0.327 | 9 |
| 0.70 | 0.680 | 29.613 | 3.537 | 0.300 | 11 |
| 0.80 | 0.683 | 30.349 | 3.251 | 0.243 | 12 |

## Train-Selected Operating Points

| Policy | Threshold | Holdout Replay-infeasible Rate | Holdout Progress | Holdout ADE3 |
|---|---:|---:|---:|---:|
| safety_first | 0.50 | 0.657 | 28.848 | 3.724 |
| balanced_80pct_proxy_progress | 0.80 | 0.683 | 30.349 | 3.251 |
| utility_90pct_proxy_progress | 0.50 | 0.657 | 28.848 | 3.724 |

