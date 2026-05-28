# nuPlan Failure-Gated Selector

- Input replay JSON: `artifacts/corl2027/nuplan_public_replay_study_interaction1000_expert/replay.json`
- Safe selector: `fit_replay_value_mlp_on_train_split`
- Fallback mode: `score`
- Fallback risk threshold: `0.500`
- Split unit: `source_db_file`
- Train scenes: `845`
- Holdout scenes: `155`
- Gate threshold: `0.500`
- Train positive rate: `0.284`
- Train gate accuracy: `0.929`

## Train

| Selector | Replay-infeasible | Rate | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|
| proxy_selector | 540 | 0.639 | 28.784 | 2.529 | 2.224 |
| safe_selector | 300 | 0.355 | 23.459 | 4.174 | 2.803 |
| oracle_failure_gate | 300 | 0.355 | 25.129 | 3.657 | 2.575 |
| progress_oracle_failure_gate | 276 | 0.327 | 24.816 | 3.775 | 2.622 |
| failure_gate | 302 | 0.357 | 24.459 | 3.873 | 2.600 |
| replay_oracle | 276 | 0.327 | 13.159 | 8.004 | 2.710 |

### Gate Diagnostics

| Metric | Value |
|---|---:|
| switch count | 296 |
| switch rate | 0.350 |
| true positive switches | 238 |
| missed avoidable failures | 2 |
| false positive switches | 58 |
| introduced failures | 0 |
| switch precision | 0.804 |
| switch recall | 0.992 |
| oracle switch count | 240 |

### Threshold Sweep

| Threshold | Replay-infeasible Rate | Progress | ADE3 | Switch Rate | Missed Avoidable |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.355 | 24.049 | 3.998 | 0.387 | 0 |
| 0.30 | 0.355 | 24.234 | 3.949 | 0.368 | 0 |
| 0.40 | 0.355 | 24.350 | 3.904 | 0.361 | 0 |
| 0.50 | 0.357 | 24.459 | 3.873 | 0.350 | 2 |
| 0.60 | 0.361 | 24.517 | 3.862 | 0.338 | 5 |
| 0.70 | 0.362 | 24.618 | 3.836 | 0.321 | 6 |
| 0.80 | 0.368 | 24.923 | 3.719 | 0.295 | 11 |

## Holdout

| Selector | Replay-infeasible | Rate | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|
| proxy_selector | 87 | 0.561 | 23.482 | 2.159 | 2.253 |
| safe_selector | 5 | 0.032 | 12.145 | 6.434 | 1.895 |
| oracle_failure_gate | 5 | 0.032 | 13.009 | 6.055 | 2.007 |
| progress_oracle_failure_gate | 5 | 0.032 | 13.046 | 6.008 | 2.003 |
| failure_gate | 9 | 0.058 | 13.031 | 6.024 | 1.998 |
| replay_oracle | 5 | 0.032 | 9.125 | 7.570 | 2.218 |

### Gate Diagnostics

| Metric | Value |
|---|---:|
| switch count | 83 |
| switch rate | 0.535 |
| true positive switches | 78 |
| missed avoidable failures | 4 |
| false positive switches | 5 |
| introduced failures | 0 |
| switch precision | 0.940 |
| switch recall | 0.951 |
| oracle switch count | 82 |

### Threshold Sweep

| Threshold | Replay-infeasible Rate | Progress | ADE3 | Switch Rate | Missed Avoidable |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.039 | 12.932 | 6.114 | 0.716 | 1 |
| 0.30 | 0.052 | 13.019 | 6.040 | 0.684 | 3 |
| 0.40 | 0.058 | 13.031 | 6.024 | 0.619 | 4 |
| 0.50 | 0.058 | 13.031 | 6.024 | 0.535 | 4 |
| 0.60 | 0.058 | 13.031 | 6.024 | 0.516 | 4 |
| 0.70 | 0.058 | 13.031 | 6.024 | 0.510 | 4 |
| 0.80 | 0.058 | 13.031 | 6.024 | 0.510 | 4 |

## Train-Selected Operating Points

| Policy | Threshold | Holdout Replay-infeasible Rate | Holdout Progress | Holdout ADE3 |
|---|---:|---:|---:|---:|
| safety_first | 0.40 | 0.058 | 13.031 | 6.024 |
| balanced_80pct_proxy_progress | 0.40 | 0.058 | 13.031 | 6.024 |
| utility_90pct_proxy_progress | 0.40 | 0.058 | 13.031 | 6.024 |

