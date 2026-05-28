# nuPlan Failure-Gated Selector

- Input replay JSON: `artifacts/corl2027/nuplan_public_replay_study_interaction1000_expert/replay.json`
- Safe selector: `fit_replay_value_mlp_on_train_split`
- Fallback mode: `score`
- Fallback risk threshold: `0.500`
- Split unit: `source_db_file`
- Train scenes: `700`
- Holdout scenes: `300`
- Gate threshold: `0.500`
- Train positive rate: `0.244`
- Train gate accuracy: `0.934`

## Train

| Selector | Replay-infeasible | Rate | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|
| proxy_selector | 418 | 0.597 | 27.353 | 2.566 | 2.142 |
| safe_selector | 247 | 0.353 | 22.060 | 3.997 | 2.906 |
| oracle_failure_gate | 247 | 0.353 | 24.103 | 3.487 | 2.505 |
| progress_oracle_failure_gate | 232 | 0.331 | 23.796 | 3.618 | 2.562 |
| failure_gate | 249 | 0.356 | 23.413 | 3.713 | 2.530 |
| replay_oracle | 232 | 0.331 | 13.239 | 7.340 | 2.703 |

### Gate Diagnostics

| Metric | Value |
|---|---:|
| switch count | 213 |
| switch rate | 0.304 |
| true positive switches | 169 |
| missed avoidable failures | 2 |
| false positive switches | 44 |
| introduced failures | 0 |
| switch precision | 0.793 |
| switch recall | 0.988 |
| oracle switch count | 171 |

### Threshold Sweep

| Threshold | Replay-infeasible Rate | Progress | ADE3 | Switch Rate | Missed Avoidable |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.353 | 23.011 | 3.801 | 0.350 | 0 |
| 0.30 | 0.353 | 23.136 | 3.775 | 0.333 | 0 |
| 0.40 | 0.353 | 23.269 | 3.737 | 0.323 | 0 |
| 0.50 | 0.356 | 23.413 | 3.713 | 0.304 | 2 |
| 0.60 | 0.360 | 23.529 | 3.685 | 0.291 | 5 |
| 0.70 | 0.361 | 23.609 | 3.665 | 0.281 | 6 |
| 0.80 | 0.371 | 23.878 | 3.570 | 0.254 | 13 |

## Holdout

| Selector | Replay-infeasible | Rate | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|
| proxy_selector | 209 | 0.697 | 29.383 | 2.251 | 2.240 |
| safe_selector | 53 | 0.177 | 19.765 | 5.768 | 2.253 |
| oracle_failure_gate | 53 | 0.177 | 20.941 | 5.392 | 2.346 |
| progress_oracle_failure_gate | 49 | 0.163 | 21.114 | 5.297 | 2.354 |
| failure_gate | 66 | 0.220 | 20.805 | 5.450 | 2.248 |
| replay_oracle | 49 | 0.163 | 10.889 | 9.329 | 2.344 |

### Gate Diagnostics

| Metric | Value |
|---|---:|
| switch count | 171 |
| switch rate | 0.570 |
| true positive switches | 143 |
| missed avoidable failures | 13 |
| false positive switches | 28 |
| introduced failures | 0 |
| switch precision | 0.836 |
| switch recall | 0.917 |
| oracle switch count | 156 |

### Threshold Sweep

| Threshold | Replay-infeasible Rate | Progress | ADE3 | Switch Rate | Missed Avoidable |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.190 | 20.397 | 5.568 | 0.670 | 4 |
| 0.30 | 0.197 | 20.462 | 5.537 | 0.660 | 6 |
| 0.40 | 0.210 | 20.658 | 5.488 | 0.600 | 10 |
| 0.50 | 0.220 | 20.805 | 5.450 | 0.570 | 13 |
| 0.60 | 0.223 | 20.855 | 5.437 | 0.550 | 14 |
| 0.70 | 0.237 | 20.981 | 5.397 | 0.537 | 18 |
| 0.80 | 0.240 | 20.995 | 5.389 | 0.523 | 19 |

## Train-Selected Operating Points

| Policy | Threshold | Holdout Replay-infeasible Rate | Holdout Progress | Holdout ADE3 |
|---|---:|---:|---:|---:|
| safety_first | 0.40 | 0.210 | 20.658 | 5.488 |
| balanced_80pct_proxy_progress | 0.40 | 0.210 | 20.658 | 5.488 |
| utility_90pct_proxy_progress | 0.40 | 0.210 | 20.658 | 5.488 |

