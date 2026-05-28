# nuPlan Failure-Gated Selector

- Input replay JSON: `artifacts/corl2027/nuplan_public_replay_study_interaction1000_expert/replay.json`
- Safe selector: `fit_replay_calibrated_on_train_split`
- Fallback mode: `risk_constrained_progress`
- Fallback risk threshold: `0.500`
- Split unit: `source_db_file`
- Train scenes: `700`
- Holdout scenes: `300`
- Gate threshold: `0.500`
- Train positive rate: `0.219`
- Train gate accuracy: `0.969`

## Train

| Selector | Replay-infeasible | Rate | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|
| proxy_selector | 418 | 0.597 | 27.353 | 2.566 | 2.142 |
| safe_selector | 295 | 0.421 | 15.792 | 6.668 | 2.622 |
| oracle_failure_gate | 265 | 0.379 | 24.172 | 3.583 | 2.512 |
| progress_oracle_failure_gate | 232 | 0.331 | 23.796 | 3.618 | 2.562 |
| failure_gate | 265 | 0.379 | 23.768 | 3.717 | 2.530 |
| replay_oracle | 232 | 0.331 | 13.239 | 7.340 | 2.703 |

### Gate Diagnostics

| Metric | Value |
|---|---:|
| switch count | 175 |
| switch rate | 0.250 |
| true positive switches | 153 |
| missed avoidable failures | 0 |
| false positive switches | 22 |
| introduced failures | 0 |
| switch precision | 0.874 |
| switch recall | 1.000 |
| oracle switch count | 153 |

### Threshold Sweep

| Threshold | Replay-infeasible Rate | Progress | ADE3 | Switch Rate | Missed Avoidable |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.380 | 22.784 | 4.046 | 0.290 | 0 |
| 0.30 | 0.380 | 23.400 | 3.858 | 0.263 | 0 |
| 0.40 | 0.380 | 23.554 | 3.800 | 0.257 | 0 |
| 0.50 | 0.379 | 23.768 | 3.717 | 0.250 | 0 |
| 0.60 | 0.379 | 23.891 | 3.671 | 0.240 | 0 |
| 0.70 | 0.379 | 23.990 | 3.632 | 0.236 | 0 |
| 0.80 | 0.389 | 24.196 | 3.556 | 0.226 | 7 |

## Holdout

| Selector | Replay-infeasible | Rate | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|
| proxy_selector | 209 | 0.697 | 29.383 | 2.251 | 2.240 |
| safe_selector | 85 | 0.283 | 14.201 | 7.955 | 2.493 |
| oracle_failure_gate | 84 | 0.280 | 20.798 | 5.493 | 2.480 |
| progress_oracle_failure_gate | 49 | 0.163 | 21.114 | 5.297 | 2.354 |
| failure_gate | 139 | 0.463 | 24.643 | 3.868 | 2.625 |
| replay_oracle | 49 | 0.163 | 10.889 | 9.329 | 2.344 |

### Gate Diagnostics

| Metric | Value |
|---|---:|
| switch count | 149 |
| switch rate | 0.497 |
| true positive switches | 70 |
| missed avoidable failures | 55 |
| false positive switches | 79 |
| introduced failures | 0 |
| switch precision | 0.470 |
| switch recall | 0.560 |
| oracle switch count | 125 |

### Threshold Sweep

| Threshold | Replay-infeasible Rate | Progress | ADE3 | Switch Rate | Missed Avoidable |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.307 | 20.061 | 5.802 | 0.673 | 8 |
| 0.30 | 0.420 | 23.553 | 4.298 | 0.547 | 42 |
| 0.40 | 0.447 | 24.332 | 3.966 | 0.520 | 50 |
| 0.50 | 0.463 | 24.643 | 3.868 | 0.497 | 55 |
| 0.60 | 0.467 | 24.757 | 3.821 | 0.493 | 56 |
| 0.70 | 0.470 | 24.772 | 3.813 | 0.490 | 57 |
| 0.80 | 0.483 | 25.256 | 3.617 | 0.410 | 61 |

## Train-Selected Operating Points

| Policy | Threshold | Holdout Replay-infeasible Rate | Holdout Progress | Holdout ADE3 |
|---|---:|---:|---:|---:|
| safety_first | 0.70 | 0.470 | 24.772 | 3.813 |
| balanced_80pct_proxy_progress | 0.70 | 0.470 | 24.772 | 3.813 |
| utility_90pct_proxy_progress | 0.70 | 0.470 | 24.772 | 3.813 |

