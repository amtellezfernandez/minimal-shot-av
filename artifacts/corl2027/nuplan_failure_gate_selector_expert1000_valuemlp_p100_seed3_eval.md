# nuPlan Failure-Gated Selector

- Input replay JSON: `artifacts/corl2027/nuplan_public_replay_study_interaction1000_expert/replay.json`
- Safe selector: `fit_replay_value_mlp_on_train_split`
- Fallback mode: `score`
- Fallback risk threshold: `0.500`
- Split unit: `source_db_file`
- Train scenes: `700`
- Holdout scenes: `300`
- Gate threshold: `0.500`
- Train positive rate: `0.367`
- Train gate accuracy: `0.943`

## Train

| Selector | Replay-infeasible | Rate | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|
| proxy_selector | 380 | 0.543 | 25.728 | 2.581 | 2.376 |
| safe_selector | 123 | 0.176 | 17.402 | 5.096 | 2.873 |
| oracle_failure_gate | 123 | 0.176 | 20.158 | 4.471 | 2.614 |
| progress_oracle_failure_gate | 104 | 0.149 | 20.155 | 4.433 | 2.667 |
| failure_gate | 123 | 0.176 | 19.414 | 4.699 | 2.589 |
| replay_oracle | 104 | 0.149 | 11.221 | 7.702 | 2.677 |

### Gate Diagnostics

| Metric | Value |
|---|---:|
| switch count | 297 |
| switch rate | 0.424 |
| true positive switches | 257 |
| missed avoidable failures | 0 |
| false positive switches | 40 |
| introduced failures | 0 |
| switch precision | 0.865 |
| switch recall | 1.000 |
| oracle switch count | 257 |

### Threshold Sweep

| Threshold | Replay-infeasible Rate | Progress | ADE3 | Switch Rate | Missed Avoidable |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.176 | 18.833 | 4.881 | 0.484 | 0 |
| 0.30 | 0.176 | 19.146 | 4.810 | 0.457 | 0 |
| 0.40 | 0.176 | 19.268 | 4.764 | 0.443 | 0 |
| 0.50 | 0.176 | 19.414 | 4.699 | 0.424 | 0 |
| 0.60 | 0.179 | 19.523 | 4.669 | 0.411 | 2 |
| 0.70 | 0.183 | 19.828 | 4.573 | 0.389 | 5 |
| 0.80 | 0.197 | 20.169 | 4.465 | 0.363 | 15 |

## Holdout

| Selector | Replay-infeasible | Rate | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|
| proxy_selector | 247 | 0.823 | 33.173 | 2.217 | 2.082 |
| safe_selector | 184 | 0.613 | 27.027 | 4.255 | 2.531 |
| oracle_failure_gate | 184 | 0.613 | 30.090 | 3.235 | 2.216 |
| progress_oracle_failure_gate | 177 | 0.590 | 29.610 | 3.394 | 2.285 |
| failure_gate | 187 | 0.623 | 28.133 | 4.009 | 2.307 |
| replay_oracle | 177 | 0.590 | 15.598 | 8.484 | 2.197 |

### Gate Diagnostics

| Metric | Value |
|---|---:|
| switch count | 90 |
| switch rate | 0.300 |
| true positive switches | 60 |
| missed avoidable failures | 3 |
| false positive switches | 30 |
| introduced failures | 0 |
| switch precision | 0.667 |
| switch recall | 0.952 |
| oracle switch count | 63 |

### Threshold Sweep

| Threshold | Replay-infeasible Rate | Progress | ADE3 | Switch Rate | Missed Avoidable |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.613 | 27.254 | 4.270 | 0.370 | 0 |
| 0.30 | 0.617 | 27.694 | 4.146 | 0.323 | 1 |
| 0.40 | 0.617 | 28.024 | 4.048 | 0.307 | 1 |
| 0.50 | 0.623 | 28.133 | 4.009 | 0.300 | 3 |
| 0.60 | 0.647 | 28.556 | 3.855 | 0.270 | 10 |
| 0.70 | 0.653 | 28.730 | 3.867 | 0.243 | 12 |
| 0.80 | 0.667 | 28.909 | 3.813 | 0.230 | 16 |

## Train-Selected Operating Points

| Policy | Threshold | Holdout Replay-infeasible Rate | Holdout Progress | Holdout ADE3 |
|---|---:|---:|---:|---:|
| safety_first | 0.50 | 0.623 | 28.133 | 4.009 |
| balanced_80pct_proxy_progress | 0.50 | 0.623 | 28.133 | 4.009 |
| utility_90pct_proxy_progress | 0.50 | 0.623 | 28.133 | 4.009 |

