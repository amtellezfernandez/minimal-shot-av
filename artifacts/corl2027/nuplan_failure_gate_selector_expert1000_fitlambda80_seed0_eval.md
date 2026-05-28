# nuPlan Failure-Gated Selector

- Input replay JSON: `artifacts/corl2027/nuplan_public_replay_study_interaction1000_expert/replay.json`
- Safe selector: `fit_on_train_split`
- Split unit: `source_db_file`
- Train scenes: `700`
- Holdout scenes: `300`
- Gate threshold: `0.500`
- Train positive rate: `0.246`
- Train gate accuracy: `0.916`

## Train

| Selector | Replay-infeasible | Rate | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|
| proxy_selector | 418 | 0.597 | 27.353 | 2.566 | 2.142 |
| safe_selector | 246 | 0.351 | 20.110 | 4.751 | 2.962 |
| failure_gate | 246 | 0.351 | 23.263 | 3.744 | 2.597 |
| replay_oracle | 232 | 0.331 | 13.239 | 7.340 | 2.703 |

### Gate Diagnostics

| Metric | Value |
|---|---:|
| switch count | 231 |
| switch rate | 0.330 |
| true positive switches | 172 |
| missed avoidable failures | 0 |
| false positive switches | 59 |
| introduced failures | 0 |

### Threshold Sweep

| Threshold | Replay-infeasible Rate | Progress | ADE3 | Switch Rate | Missed Avoidable |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.351 | 22.492 | 3.959 | 0.443 | 0 |
| 0.30 | 0.351 | 22.714 | 3.888 | 0.411 | 0 |
| 0.40 | 0.351 | 23.175 | 3.755 | 0.351 | 0 |
| 0.50 | 0.351 | 23.263 | 3.744 | 0.330 | 0 |
| 0.60 | 0.357 | 23.572 | 3.633 | 0.301 | 4 |
| 0.70 | 0.380 | 24.353 | 3.432 | 0.240 | 20 |
| 0.80 | 0.407 | 24.778 | 3.304 | 0.209 | 39 |

## Holdout

| Selector | Replay-infeasible | Rate | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|
| proxy_selector | 209 | 0.697 | 29.383 | 2.251 | 2.240 |
| safe_selector | 49 | 0.163 | 18.342 | 6.334 | 2.106 |
| failure_gate | 84 | 0.280 | 22.917 | 4.497 | 2.467 |
| replay_oracle | 49 | 0.163 | 10.889 | 9.329 | 2.344 |

### Gate Diagnostics

| Metric | Value |
|---|---:|
| switch count | 176 |
| switch rate | 0.587 |
| true positive switches | 125 |
| missed avoidable failures | 35 |
| false positive switches | 51 |
| introduced failures | 0 |

### Threshold Sweep

| Threshold | Replay-infeasible Rate | Progress | ADE3 | Switch Rate | Missed Avoidable |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.177 | 18.860 | 6.166 | 0.763 | 4 |
| 0.30 | 0.177 | 19.451 | 5.945 | 0.730 | 4 |
| 0.40 | 0.183 | 20.077 | 5.698 | 0.703 | 6 |
| 0.50 | 0.280 | 22.917 | 4.497 | 0.587 | 35 |
| 0.60 | 0.427 | 25.168 | 3.615 | 0.437 | 79 |
| 0.70 | 0.493 | 26.087 | 3.284 | 0.367 | 99 |
| 0.80 | 0.570 | 27.416 | 2.950 | 0.287 | 122 |

## Train-Selected Operating Points

| Policy | Threshold | Holdout Replay-infeasible Rate | Holdout Progress | Holdout ADE3 |
|---|---:|---:|---:|---:|
| safety_first | 0.50 | 0.280 | 22.917 | 4.497 |
| balanced_80pct_proxy_progress | 0.50 | 0.280 | 22.917 | 4.497 |
| utility_90pct_proxy_progress | 0.80 | 0.570 | 27.416 | 2.950 |

