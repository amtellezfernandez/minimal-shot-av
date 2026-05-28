# nuPlan Failure-Gated Selector

- Input replay JSON: `artifacts/corl2027/nuplan_public_replay_study_interaction1000_expert/replay.json`
- Safe selector: `fit_replay_value_on_train_split`
- Fallback mode: `score`
- Fallback risk threshold: `0.500`
- Split unit: `source_db_file`
- Train scenes: `700`
- Holdout scenes: `300`
- Gate threshold: `0.500`
- Train positive rate: `0.251`
- Train gate accuracy: `0.951`

## Train

| Selector | Replay-infeasible | Rate | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|
| proxy_selector | 418 | 0.597 | 27.353 | 2.566 | 2.142 |
| safe_selector | 249 | 0.356 | 19.271 | 5.366 | 2.498 |
| oracle_failure_gate | 242 | 0.346 | 23.178 | 3.868 | 2.534 |
| progress_oracle_failure_gate | 232 | 0.331 | 23.796 | 3.618 | 2.562 |
| failure_gate | 242 | 0.346 | 22.478 | 4.116 | 2.542 |
| replay_oracle | 232 | 0.331 | 13.239 | 7.340 | 2.703 |

### Gate Diagnostics

| Metric | Value |
|---|---:|
| switch count | 210 |
| switch rate | 0.300 |
| true positive switches | 176 |
| missed avoidable failures | 0 |
| false positive switches | 34 |
| introduced failures | 0 |
| switch precision | 0.838 |
| switch recall | 1.000 |
| oracle switch count | 176 |

### Threshold Sweep

| Threshold | Replay-infeasible Rate | Progress | ADE3 | Switch Rate | Missed Avoidable |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.346 | 21.182 | 4.587 | 0.347 | 0 |
| 0.30 | 0.346 | 21.458 | 4.492 | 0.330 | 0 |
| 0.40 | 0.346 | 22.271 | 4.193 | 0.310 | 0 |
| 0.50 | 0.346 | 22.478 | 4.116 | 0.300 | 0 |
| 0.60 | 0.346 | 22.712 | 4.034 | 0.287 | 0 |
| 0.70 | 0.347 | 23.041 | 3.907 | 0.267 | 1 |
| 0.80 | 0.361 | 23.293 | 3.820 | 0.250 | 11 |

## Holdout

| Selector | Replay-infeasible | Rate | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|
| proxy_selector | 209 | 0.697 | 29.383 | 2.251 | 2.240 |
| safe_selector | 49 | 0.163 | 13.476 | 8.477 | 2.244 |
| oracle_failure_gate | 49 | 0.163 | 17.703 | 6.795 | 2.342 |
| progress_oracle_failure_gate | 49 | 0.163 | 21.114 | 5.297 | 2.354 |
| failure_gate | 66 | 0.220 | 17.391 | 6.924 | 2.295 |
| replay_oracle | 49 | 0.163 | 10.889 | 9.329 | 2.344 |

### Gate Diagnostics

| Metric | Value |
|---|---:|
| switch count | 180 |
| switch rate | 0.600 |
| true positive switches | 143 |
| missed avoidable failures | 17 |
| false positive switches | 37 |
| introduced failures | 0 |
| switch precision | 0.794 |
| switch recall | 0.894 |
| oracle switch count | 160 |

### Threshold Sweep

| Threshold | Replay-infeasible Rate | Progress | ADE3 | Switch Rate | Missed Avoidable |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.177 | 14.231 | 8.140 | 0.700 | 4 |
| 0.30 | 0.183 | 15.226 | 7.753 | 0.670 | 6 |
| 0.40 | 0.197 | 16.702 | 7.202 | 0.623 | 10 |
| 0.50 | 0.220 | 17.391 | 6.924 | 0.600 | 17 |
| 0.60 | 0.237 | 17.716 | 6.810 | 0.577 | 22 |
| 0.70 | 0.250 | 18.136 | 6.644 | 0.537 | 26 |
| 0.80 | 0.287 | 19.033 | 6.358 | 0.470 | 37 |

## Train-Selected Operating Points

| Policy | Threshold | Holdout Replay-infeasible Rate | Holdout Progress | Holdout ADE3 |
|---|---:|---:|---:|---:|
| safety_first | 0.60 | 0.237 | 17.716 | 6.810 |
| balanced_80pct_proxy_progress | 0.60 | 0.237 | 17.716 | 6.810 |
| utility_90pct_proxy_progress | 0.60 | 0.237 | 17.716 | 6.810 |

