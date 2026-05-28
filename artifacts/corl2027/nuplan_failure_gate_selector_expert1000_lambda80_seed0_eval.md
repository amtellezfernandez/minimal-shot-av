# nuPlan Failure-Gated Selector

- Input replay JSON: `artifacts/corl2027/nuplan_public_replay_study_interaction1000_expert/replay.json`
- Safe selector: `artifacts/corl2027/nuplan_replay_calibrated_selector_expert1000_lambda80.json`
- Split unit: `source_db_file`
- Train scenes: `700`
- Holdout scenes: `300`
- Gate threshold: `0.500`
- Train positive rate: `0.247`
- Train gate accuracy: `0.920`

## Train

| Selector | Replay-infeasible | Rate | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|
| proxy_selector | 418 | 0.597 | 27.353 | 2.566 | 2.142 |
| safe_selector | 245 | 0.350 | 21.432 | 4.231 | 2.906 |
| failure_gate | 245 | 0.350 | 23.328 | 3.714 | 2.554 |
| replay_oracle | 232 | 0.331 | 13.239 | 7.340 | 2.703 |

### Gate Diagnostics

| Metric | Value |
|---|---:|
| switch count | 229 |
| switch rate | 0.327 |
| true positive switches | 173 |
| missed avoidable failures | 0 |
| false positive switches | 56 |
| introduced failures | 0 |

### Threshold Sweep

| Threshold | Replay-infeasible Rate | Progress | ADE3 | Switch Rate | Missed Avoidable |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.350 | 23.008 | 3.749 | 0.449 | 0 |
| 0.30 | 0.350 | 23.084 | 3.731 | 0.417 | 0 |
| 0.40 | 0.350 | 23.170 | 3.731 | 0.359 | 0 |
| 0.50 | 0.350 | 23.328 | 3.714 | 0.327 | 0 |
| 0.60 | 0.359 | 23.545 | 3.645 | 0.301 | 6 |
| 0.70 | 0.379 | 24.267 | 3.453 | 0.243 | 20 |
| 0.80 | 0.407 | 24.792 | 3.301 | 0.209 | 40 |

## Holdout

| Selector | Replay-infeasible | Rate | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|
| proxy_selector | 209 | 0.697 | 29.383 | 2.251 | 2.240 |
| safe_selector | 49 | 0.163 | 19.474 | 5.933 | 2.179 |
| failure_gate | 83 | 0.277 | 23.164 | 4.394 | 2.494 |
| replay_oracle | 49 | 0.163 | 10.889 | 9.329 | 2.344 |

### Gate Diagnostics

| Metric | Value |
|---|---:|
| switch count | 177 |
| switch rate | 0.590 |
| true positive switches | 126 |
| missed avoidable failures | 34 |
| false positive switches | 51 |
| introduced failures | 0 |

### Threshold Sweep

| Threshold | Replay-infeasible Rate | Progress | ADE3 | Switch Rate | Missed Avoidable |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.177 | 19.845 | 5.785 | 0.763 | 4 |
| 0.30 | 0.177 | 19.845 | 5.785 | 0.730 | 4 |
| 0.40 | 0.183 | 20.353 | 5.584 | 0.703 | 6 |
| 0.50 | 0.277 | 23.164 | 4.394 | 0.590 | 34 |
| 0.60 | 0.427 | 25.444 | 3.500 | 0.433 | 79 |
| 0.70 | 0.490 | 26.336 | 3.178 | 0.370 | 98 |
| 0.80 | 0.567 | 27.605 | 2.852 | 0.290 | 121 |

