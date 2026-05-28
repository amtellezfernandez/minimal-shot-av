# nuPlan Failure-Gated Selector

- Input replay JSON: `artifacts/corl2027/nuplan_public_replay_study_interaction1000_expert/replay.json`
- Safe selector: `fit_replay_value_mlp_on_train_split`
- Fallback mode: `score`
- Fallback risk threshold: `0.500`
- Split unit: `source_db_file`
- Train scenes: `700`
- Holdout scenes: `300`
- Gate threshold: `0.500`
- Train positive rate: `0.364`
- Train gate accuracy: `0.921`

## Train

| Selector | Replay-infeasible | Rate | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|
| proxy_selector | 428 | 0.611 | 27.518 | 2.392 | 2.472 |
| safe_selector | 173 | 0.247 | 19.374 | 5.103 | 2.881 |
| oracle_failure_gate | 173 | 0.247 | 22.007 | 4.314 | 2.658 |
| progress_oracle_failure_gate | 163 | 0.233 | 22.041 | 4.279 | 2.656 |
| failure_gate | 173 | 0.247 | 21.147 | 4.620 | 2.644 |
| replay_oracle | 163 | 0.233 | 13.043 | 7.658 | 2.571 |

### Gate Diagnostics

| Metric | Value |
|---|---:|
| switch count | 310 |
| switch rate | 0.443 |
| true positive switches | 255 |
| missed avoidable failures | 0 |
| false positive switches | 55 |
| introduced failures | 0 |
| switch precision | 0.823 |
| switch recall | 1.000 |
| oracle switch count | 255 |

### Threshold Sweep

| Threshold | Replay-infeasible Rate | Progress | ADE3 | Switch Rate | Missed Avoidable |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.247 | 20.980 | 4.682 | 0.477 | 0 |
| 0.30 | 0.247 | 21.049 | 4.657 | 0.464 | 0 |
| 0.40 | 0.247 | 21.084 | 4.642 | 0.457 | 0 |
| 0.50 | 0.247 | 21.147 | 4.620 | 0.443 | 0 |
| 0.60 | 0.250 | 21.267 | 4.564 | 0.430 | 2 |
| 0.70 | 0.253 | 21.508 | 4.482 | 0.404 | 4 |
| 0.80 | 0.264 | 21.866 | 4.331 | 0.379 | 12 |

## Holdout

| Selector | Replay-infeasible | Rate | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|
| proxy_selector | 199 | 0.663 | 28.998 | 2.658 | 1.882 |
| safe_selector | 127 | 0.423 | 22.097 | 4.565 | 2.774 |
| oracle_failure_gate | 127 | 0.423 | 24.975 | 3.806 | 2.324 |
| progress_oracle_failure_gate | 118 | 0.393 | 25.209 | 3.753 | 2.459 |
| failure_gate | 147 | 0.490 | 24.768 | 3.913 | 2.268 |
| replay_oracle | 118 | 0.393 | 11.346 | 8.587 | 2.613 |

### Gate Diagnostics

| Metric | Value |
|---|---:|
| switch count | 68 |
| switch rate | 0.227 |
| true positive switches | 52 |
| missed avoidable failures | 20 |
| false positive switches | 16 |
| introduced failures | 0 |
| switch precision | 0.765 |
| switch recall | 0.722 |
| oracle switch count | 72 |

### Threshold Sweep

| Threshold | Replay-infeasible Rate | Progress | ADE3 | Switch Rate | Missed Avoidable |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.447 | 23.089 | 4.309 | 0.387 | 7 |
| 0.30 | 0.463 | 23.941 | 4.138 | 0.320 | 12 |
| 0.40 | 0.483 | 24.422 | 3.992 | 0.257 | 18 |
| 0.50 | 0.490 | 24.768 | 3.913 | 0.227 | 20 |
| 0.60 | 0.507 | 25.115 | 3.833 | 0.200 | 25 |
| 0.70 | 0.530 | 25.618 | 3.697 | 0.163 | 32 |
| 0.80 | 0.567 | 27.206 | 3.187 | 0.100 | 43 |

## Train-Selected Operating Points

| Policy | Threshold | Holdout Replay-infeasible Rate | Holdout Progress | Holdout ADE3 |
|---|---:|---:|---:|---:|
| safety_first | 0.50 | 0.490 | 24.768 | 3.913 |
| balanced_80pct_proxy_progress | 0.50 | 0.490 | 24.768 | 3.913 |
| utility_90pct_proxy_progress | 0.50 | 0.490 | 24.768 | 3.913 |

