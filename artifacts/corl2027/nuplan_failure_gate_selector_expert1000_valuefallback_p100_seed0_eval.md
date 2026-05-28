# nuPlan Failure-Gated Selector

- Input replay JSON: `artifacts/corl2027/nuplan_public_replay_study_interaction1000_expert/replay.json`
- Safe selector: `fit_replay_value_on_train_split`
- Fallback mode: `score`
- Fallback risk threshold: `0.500`
- Split unit: `source_db_file`
- Train scenes: `700`
- Holdout scenes: `300`
- Gate threshold: `0.500`
- Train positive rate: `0.209`
- Train gate accuracy: `0.954`

## Train

| Selector | Replay-infeasible | Rate | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|
| proxy_selector | 418 | 0.597 | 27.353 | 2.566 | 2.142 |
| safe_selector | 288 | 0.411 | 23.187 | 4.111 | 2.371 |
| oracle_failure_gate | 272 | 0.389 | 24.260 | 3.551 | 2.463 |
| progress_oracle_failure_gate | 232 | 0.331 | 23.796 | 3.618 | 2.562 |
| failure_gate | 272 | 0.389 | 23.554 | 3.791 | 2.482 |
| replay_oracle | 232 | 0.331 | 13.239 | 7.340 | 2.703 |

### Gate Diagnostics

| Metric | Value |
|---|---:|
| switch count | 178 |
| switch rate | 0.254 |
| true positive switches | 146 |
| missed avoidable failures | 0 |
| false positive switches | 32 |
| introduced failures | 0 |
| switch precision | 0.820 |
| switch recall | 1.000 |
| oracle switch count | 146 |

### Threshold Sweep

| Threshold | Replay-infeasible Rate | Progress | ADE3 | Switch Rate | Missed Avoidable |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.389 | 23.432 | 3.831 | 0.271 | 0 |
| 0.30 | 0.389 | 23.501 | 3.809 | 0.261 | 0 |
| 0.40 | 0.389 | 23.516 | 3.804 | 0.259 | 0 |
| 0.50 | 0.389 | 23.554 | 3.791 | 0.254 | 0 |
| 0.60 | 0.390 | 23.638 | 3.774 | 0.246 | 1 |
| 0.70 | 0.390 | 23.697 | 3.754 | 0.239 | 1 |
| 0.80 | 0.400 | 23.962 | 3.664 | 0.219 | 8 |

## Holdout

| Selector | Replay-infeasible | Rate | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|
| proxy_selector | 209 | 0.697 | 29.383 | 2.251 | 2.240 |
| safe_selector | 52 | 0.173 | 18.690 | 6.397 | 2.371 |
| oracle_failure_gate | 50 | 0.167 | 19.371 | 6.076 | 2.529 |
| progress_oracle_failure_gate | 49 | 0.163 | 21.114 | 5.297 | 2.354 |
| failure_gate | 57 | 0.190 | 18.893 | 6.233 | 2.505 |
| replay_oracle | 49 | 0.163 | 10.889 | 9.329 | 2.344 |

### Gate Diagnostics

| Metric | Value |
|---|---:|
| switch count | 179 |
| switch rate | 0.597 |
| true positive switches | 152 |
| missed avoidable failures | 7 |
| false positive switches | 27 |
| introduced failures | 0 |
| switch precision | 0.849 |
| switch recall | 0.956 |
| oracle switch count | 159 |

### Threshold Sweep

| Threshold | Replay-infeasible Rate | Progress | ADE3 | Switch Rate | Missed Avoidable |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.180 | 18.780 | 6.276 | 0.610 | 4 |
| 0.30 | 0.180 | 18.780 | 6.276 | 0.610 | 4 |
| 0.40 | 0.187 | 18.844 | 6.245 | 0.603 | 6 |
| 0.50 | 0.190 | 18.893 | 6.233 | 0.597 | 7 |
| 0.60 | 0.197 | 19.015 | 6.190 | 0.587 | 9 |
| 0.70 | 0.223 | 19.621 | 5.963 | 0.527 | 17 |
| 0.80 | 0.263 | 20.223 | 5.729 | 0.487 | 29 |

## Train-Selected Operating Points

| Policy | Threshold | Holdout Replay-infeasible Rate | Holdout Progress | Holdout ADE3 |
|---|---:|---:|---:|---:|
| safety_first | 0.50 | 0.190 | 18.893 | 6.233 |
| balanced_80pct_proxy_progress | 0.50 | 0.190 | 18.893 | 6.233 |
| utility_90pct_proxy_progress | 0.50 | 0.190 | 18.893 | 6.233 |

