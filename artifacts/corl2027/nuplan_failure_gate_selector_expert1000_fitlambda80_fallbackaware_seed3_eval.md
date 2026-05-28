# nuPlan Failure-Gated Selector

- Input replay JSON: `artifacts/corl2027/nuplan_public_replay_study_interaction1000_expert/replay.json`
- Safe selector: `fit_on_train_split`
- Fallback mode: `score`
- Fallback risk threshold: `0.500`
- Split unit: `source_db_file`
- Train scenes: `700`
- Holdout scenes: `300`
- Gate threshold: `0.500`
- Train positive rate: `0.387`
- Train gate accuracy: `0.934`

## Train

| Selector | Replay-infeasible | Rate | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|
| proxy_selector | 380 | 0.543 | 25.728 | 2.581 | 2.376 |
| safe_selector | 109 | 0.156 | 15.974 | 5.741 | 2.791 |
| oracle_failure_gate | 109 | 0.156 | 19.815 | 4.570 | 2.618 |
| progress_oracle_failure_gate | 104 | 0.149 | 20.155 | 4.433 | 2.667 |
| failure_gate | 109 | 0.156 | 18.645 | 4.952 | 2.622 |
| replay_oracle | 104 | 0.149 | 11.221 | 7.702 | 2.677 |

### Gate Diagnostics

| Metric | Value |
|---|---:|
| switch count | 317 |
| switch rate | 0.453 |
| true positive switches | 271 |
| missed avoidable failures | 0 |
| false positive switches | 46 |
| introduced failures | 0 |
| switch precision | 0.855 |
| switch recall | 1.000 |
| oracle switch count | 271 |

### Threshold Sweep

| Threshold | Replay-infeasible Rate | Progress | ADE3 | Switch Rate | Missed Avoidable |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.156 | 17.053 | 5.563 | 0.523 | 0 |
| 0.30 | 0.156 | 17.398 | 5.423 | 0.499 | 0 |
| 0.40 | 0.156 | 17.896 | 5.223 | 0.479 | 0 |
| 0.50 | 0.156 | 18.645 | 4.952 | 0.453 | 0 |
| 0.60 | 0.163 | 19.349 | 4.685 | 0.430 | 5 |
| 0.70 | 0.171 | 19.732 | 4.554 | 0.406 | 11 |
| 0.80 | 0.183 | 20.112 | 4.427 | 0.383 | 19 |

## Holdout

| Selector | Replay-infeasible | Rate | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|
| proxy_selector | 247 | 0.823 | 33.173 | 2.217 | 2.082 |
| safe_selector | 193 | 0.643 | 25.487 | 4.921 | 2.536 |
| oracle_failure_gate | 193 | 0.643 | 30.628 | 3.059 | 2.179 |
| progress_oracle_failure_gate | 177 | 0.590 | 29.610 | 3.394 | 2.285 |
| failure_gate | 196 | 0.653 | 29.675 | 3.371 | 2.208 |
| replay_oracle | 177 | 0.590 | 15.598 | 8.484 | 2.197 |

### Gate Diagnostics

| Metric | Value |
|---|---:|
| switch count | 72 |
| switch rate | 0.240 |
| true positive switches | 51 |
| missed avoidable failures | 3 |
| false positive switches | 21 |
| introduced failures | 0 |
| switch precision | 0.708 |
| switch recall | 0.944 |
| oracle switch count | 54 |

### Threshold Sweep

| Threshold | Replay-infeasible Rate | Progress | ADE3 | Switch Rate | Missed Avoidable |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.647 | 26.967 | 4.405 | 0.347 | 1 |
| 0.30 | 0.653 | 28.293 | 3.886 | 0.293 | 3 |
| 0.40 | 0.653 | 29.101 | 3.610 | 0.263 | 3 |
| 0.50 | 0.653 | 29.675 | 3.371 | 0.240 | 3 |
| 0.60 | 0.657 | 30.024 | 3.315 | 0.207 | 4 |
| 0.70 | 0.660 | 30.144 | 3.272 | 0.200 | 5 |
| 0.80 | 0.673 | 30.625 | 3.101 | 0.180 | 9 |

## Train-Selected Operating Points

| Policy | Threshold | Holdout Replay-infeasible Rate | Holdout Progress | Holdout ADE3 |
|---|---:|---:|---:|---:|
| safety_first | 0.50 | 0.653 | 29.675 | 3.371 |
| balanced_80pct_proxy_progress | 0.50 | 0.653 | 29.675 | 3.371 |
| utility_90pct_proxy_progress | 0.50 | 0.653 | 29.675 | 3.371 |

