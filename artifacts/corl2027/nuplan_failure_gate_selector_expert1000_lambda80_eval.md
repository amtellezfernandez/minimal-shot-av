# nuPlan Failure-Gated Selector

- Input replay JSON: `artifacts/corl2027/nuplan_public_replay_study_interaction1000_expert/replay.json`
- Safe selector: `artifacts/corl2027/nuplan_replay_calibrated_selector_expert1000_lambda80.json`
- Split unit: `source_db_file`
- Train scenes: `700`
- Holdout scenes: `300`
- Gate threshold: `0.500`
- Train positive rate: `0.290`
- Train gate accuracy: `0.911`

## Train

| Selector | Replay-infeasible | Rate | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|
| proxy_selector | 470 | 0.671 | 29.360 | 2.423 | 2.251 |
| safe_selector | 267 | 0.381 | 23.133 | 4.431 | 2.684 |
| failure_gate | 269 | 0.384 | 25.207 | 3.740 | 2.406 |
| replay_oracle | 258 | 0.369 | 12.775 | 8.400 | 2.592 |

### Gate Diagnostics

| Metric | Value |
|---|---:|
| switch count | 261 |
| switch rate | 0.373 |
| true positive switches | 201 |
| missed avoidable failures | 2 |
| false positive switches | 60 |
| introduced failures | 0 |

### Threshold Sweep

| Threshold | Replay-infeasible Rate | Progress | ADE3 | Switch Rate | Missed Avoidable |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.381 | 24.198 | 4.110 | 0.480 | 0 |
| 0.30 | 0.381 | 24.319 | 4.069 | 0.429 | 0 |
| 0.40 | 0.381 | 24.964 | 3.835 | 0.391 | 0 |
| 0.50 | 0.384 | 25.207 | 3.740 | 0.373 | 2 |
| 0.60 | 0.387 | 25.548 | 3.604 | 0.350 | 4 |
| 0.70 | 0.396 | 25.910 | 3.494 | 0.314 | 10 |
| 0.80 | 0.427 | 26.385 | 3.352 | 0.263 | 32 |

## Holdout

| Selector | Replay-infeasible | Rate | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|
| proxy_selector | 157 | 0.523 | 24.700 | 2.585 | 2.004 |
| safe_selector | 27 | 0.090 | 15.505 | 5.466 | 2.742 |
| failure_gate | 83 | 0.277 | 19.522 | 4.366 | 2.357 |
| replay_oracle | 23 | 0.077 | 11.972 | 6.855 | 2.684 |

### Gate Diagnostics

| Metric | Value |
|---|---:|
| switch count | 110 |
| switch rate | 0.367 |
| true positive switches | 74 |
| missed avoidable failures | 56 |
| false positive switches | 36 |
| introduced failures | 0 |

### Threshold Sweep

| Threshold | Replay-infeasible Rate | Progress | ADE3 | Switch Rate | Missed Avoidable |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.200 | 18.147 | 4.878 | 0.530 | 33 |
| 0.30 | 0.240 | 18.767 | 4.671 | 0.407 | 45 |
| 0.40 | 0.253 | 19.058 | 4.548 | 0.390 | 49 |
| 0.50 | 0.277 | 19.522 | 4.366 | 0.367 | 56 |
| 0.60 | 0.383 | 22.013 | 3.488 | 0.260 | 88 |
| 0.70 | 0.450 | 23.577 | 2.892 | 0.193 | 108 |
| 0.80 | 0.453 | 23.609 | 2.892 | 0.187 | 109 |

