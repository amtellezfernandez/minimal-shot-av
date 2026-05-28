# nuPlan Failure-Gated Selector

- Input replay JSON: `artifacts/corl2027/nuplan_public_replay_study_interaction1000_expert/replay.json`
- Safe selector: `artifacts/corl2027/nuplan_replay_calibrated_selector_expert1000_lambda80.json`
- Split unit: `source_db_file`
- Train scenes: `845`
- Holdout scenes: `155`
- Gate threshold: `0.500`
- Train positive rate: `0.325`
- Train gate accuracy: `0.927`

## Train

| Selector | Replay-infeasible | Rate | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|
| proxy_selector | 547 | 0.647 | 28.472 | 2.388 | 2.411 |
| safe_selector | 272 | 0.322 | 21.312 | 4.821 | 2.769 |
| failure_gate | 273 | 0.323 | 23.048 | 4.254 | 2.606 |
| replay_oracle | 263 | 0.311 | 12.179 | 8.327 | 2.591 |

### Gate Diagnostics

| Metric | Value |
|---|---:|
| switch count | 335 |
| switch rate | 0.396 |
| true positive switches | 274 |
| missed avoidable failures | 1 |
| false positive switches | 61 |
| introduced failures | 0 |

### Threshold Sweep

| Threshold | Replay-infeasible Rate | Progress | ADE3 | Switch Rate | Missed Avoidable |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.322 | 22.266 | 4.554 | 0.493 | 0 |
| 0.30 | 0.322 | 22.521 | 4.461 | 0.463 | 0 |
| 0.40 | 0.322 | 22.766 | 4.364 | 0.430 | 0 |
| 0.50 | 0.323 | 23.048 | 4.254 | 0.396 | 1 |
| 0.60 | 0.333 | 23.334 | 4.138 | 0.373 | 9 |
| 0.70 | 0.341 | 23.595 | 4.041 | 0.355 | 16 |
| 0.80 | 0.361 | 24.097 | 3.881 | 0.314 | 33 |

## Holdout

| Selector | Replay-infeasible | Rate | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|
| proxy_selector | 80 | 0.516 | 25.180 | 2.929 | 1.595 |
| safe_selector | 22 | 0.142 | 18.294 | 4.305 | 3.017 |
| failure_gate | 56 | 0.361 | 23.451 | 3.352 | 1.994 |
| replay_oracle | 18 | 0.116 | 14.471 | 5.808 | 2.847 |

### Gate Diagnostics

| Metric | Value |
|---|---:|
| switch count | 44 |
| switch rate | 0.284 |
| true positive switches | 24 |
| missed avoidable failures | 34 |
| false positive switches | 20 |
| introduced failures | 0 |

### Threshold Sweep

| Threshold | Replay-infeasible Rate | Progress | ADE3 | Switch Rate | Missed Avoidable |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.213 | 20.716 | 3.916 | 0.497 | 11 |
| 0.30 | 0.297 | 22.360 | 3.627 | 0.348 | 24 |
| 0.40 | 0.329 | 22.979 | 3.462 | 0.316 | 29 |
| 0.50 | 0.361 | 23.451 | 3.352 | 0.284 | 34 |
| 0.60 | 0.387 | 23.926 | 3.202 | 0.239 | 38 |
| 0.70 | 0.400 | 24.071 | 3.142 | 0.213 | 40 |
| 0.80 | 0.400 | 24.071 | 3.142 | 0.206 | 40 |

