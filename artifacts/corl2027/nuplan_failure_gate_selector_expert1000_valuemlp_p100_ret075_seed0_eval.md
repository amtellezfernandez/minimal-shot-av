# nuPlan Failure-Gated Selector

- Input replay JSON: `artifacts/corl2027/nuplan_public_replay_study_interaction1000_expert/replay.json`
- Safe selector: `fit_replay_value_mlp_on_train_split`
- Fallback mode: `score`
- Fallback risk threshold: `0.500`
- Split unit: `source_db_file`
- Train scenes: `700`
- Holdout scenes: `300`
- Gate threshold: `0.500`
- Train positive rate: `0.043`
- Train gate accuracy: `0.979`

## Train

| Selector | Replay-infeasible | Rate | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|
| proxy_selector | 418 | 0.597 | 27.353 | 2.566 | 2.142 |
| safe_selector | 247 | 0.353 | 22.060 | 3.997 | 2.906 |
| oracle_failure_gate | 388 | 0.554 | 27.019 | 2.664 | 2.191 |
| progress_oracle_failure_gate | 232 | 0.331 | 23.796 | 3.618 | 2.562 |
| failure_gate | 392 | 0.560 | 26.933 | 2.725 | 2.160 |
| replay_oracle | 232 | 0.331 | 13.239 | 7.340 | 2.703 |

### Gate Diagnostics

| Metric | Value |
|---|---:|
| switch count | 37 |
| switch rate | 0.053 |
| true positive switches | 26 |
| missed avoidable failures | 4 |
| false positive switches | 11 |
| introduced failures | 0 |
| switch precision | 0.703 |
| switch recall | 0.867 |
| oracle switch count | 30 |

### Threshold Sweep

| Threshold | Replay-infeasible Rate | Progress | ADE3 | Switch Rate | Missed Avoidable |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.554 | 26.838 | 2.697 | 0.079 | 0 |
| 0.30 | 0.556 | 26.899 | 2.718 | 0.061 | 1 |
| 0.40 | 0.560 | 26.930 | 2.726 | 0.054 | 4 |
| 0.50 | 0.560 | 26.933 | 2.725 | 0.053 | 4 |
| 0.60 | 0.563 | 26.940 | 2.725 | 0.049 | 6 |
| 0.70 | 0.564 | 26.982 | 2.705 | 0.041 | 7 |
| 0.80 | 0.569 | 27.038 | 2.678 | 0.034 | 10 |

## Holdout

| Selector | Replay-infeasible | Rate | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|
| proxy_selector | 209 | 0.697 | 29.383 | 2.251 | 2.240 |
| safe_selector | 53 | 0.177 | 19.765 | 5.768 | 2.253 |
| oracle_failure_gate | 195 | 0.650 | 29.088 | 2.402 | 2.275 |
| progress_oracle_failure_gate | 49 | 0.163 | 21.114 | 5.297 | 2.354 |
| failure_gate | 202 | 0.673 | 29.212 | 2.334 | 2.315 |
| replay_oracle | 49 | 0.163 | 10.889 | 9.329 | 2.344 |

### Gate Diagnostics

| Metric | Value |
|---|---:|
| switch count | 7 |
| switch rate | 0.023 |
| true positive switches | 7 |
| missed avoidable failures | 7 |
| false positive switches | 0 |
| introduced failures | 0 |
| switch precision | 1.000 |
| switch recall | 0.500 |
| oracle switch count | 14 |

### Threshold Sweep

| Threshold | Replay-infeasible Rate | Progress | ADE3 | Switch Rate | Missed Avoidable |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.663 | 29.133 | 2.372 | 0.033 | 4 |
| 0.30 | 0.663 | 29.133 | 2.372 | 0.033 | 4 |
| 0.40 | 0.663 | 29.133 | 2.372 | 0.033 | 4 |
| 0.50 | 0.673 | 29.212 | 2.334 | 0.023 | 7 |
| 0.60 | 0.680 | 29.241 | 2.319 | 0.017 | 9 |
| 0.70 | 0.680 | 29.241 | 2.319 | 0.017 | 9 |
| 0.80 | 0.680 | 29.241 | 2.319 | 0.017 | 9 |

## Train-Selected Operating Points

| Policy | Threshold | Holdout Replay-infeasible Rate | Holdout Progress | Holdout ADE3 |
|---|---:|---:|---:|---:|
| safety_first | 0.20 | 0.663 | 29.133 | 2.372 |
| balanced_80pct_proxy_progress | 0.20 | 0.663 | 29.133 | 2.372 |
| utility_90pct_proxy_progress | 0.20 | 0.663 | 29.133 | 2.372 |

