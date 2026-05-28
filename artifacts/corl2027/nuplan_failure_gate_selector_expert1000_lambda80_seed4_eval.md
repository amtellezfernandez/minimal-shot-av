# nuPlan Failure-Gated Selector

- Input replay JSON: `artifacts/corl2027/nuplan_public_replay_study_interaction1000_expert/replay.json`
- Safe selector: `artifacts/corl2027/nuplan_replay_calibrated_selector_expert1000_lambda80.json`
- Split unit: `source_db_file`
- Train scenes: `845`
- Holdout scenes: `155`
- Gate threshold: `0.500`
- Train positive rate: `0.297`
- Train gate accuracy: `0.904`

## Train

| Selector | Replay-infeasible | Rate | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|
| proxy_selector | 540 | 0.639 | 28.784 | 2.529 | 2.224 |
| safe_selector | 289 | 0.342 | 22.421 | 4.448 | 2.865 |
| failure_gate | 289 | 0.342 | 24.307 | 3.875 | 2.604 |
| replay_oracle | 276 | 0.327 | 13.159 | 8.004 | 2.710 |

### Gate Diagnostics

| Metric | Value |
|---|---:|
| switch count | 332 |
| switch rate | 0.393 |
| true positive switches | 251 |
| missed avoidable failures | 0 |
| false positive switches | 81 |
| introduced failures | 0 |

### Threshold Sweep

| Threshold | Replay-infeasible Rate | Progress | ADE3 | Switch Rate | Missed Avoidable |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.342 | 23.389 | 4.179 | 0.523 | 0 |
| 0.30 | 0.342 | 23.751 | 4.050 | 0.473 | 0 |
| 0.40 | 0.342 | 23.990 | 3.980 | 0.426 | 0 |
| 0.50 | 0.342 | 24.307 | 3.875 | 0.393 | 0 |
| 0.60 | 0.347 | 24.604 | 3.783 | 0.363 | 4 |
| 0.70 | 0.363 | 25.073 | 3.647 | 0.323 | 18 |
| 0.80 | 0.402 | 25.816 | 3.436 | 0.260 | 51 |

## Holdout

| Selector | Replay-infeasible | Rate | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|
| proxy_selector | 87 | 0.561 | 23.482 | 2.159 | 2.253 |
| safe_selector | 5 | 0.032 | 12.250 | 6.338 | 1.849 |
| failure_gate | 52 | 0.335 | 19.218 | 3.343 | 2.264 |
| replay_oracle | 5 | 0.032 | 9.125 | 7.570 | 2.218 |

### Gate Diagnostics

| Metric | Value |
|---|---:|
| switch count | 63 |
| switch rate | 0.406 |
| true positive switches | 35 |
| missed avoidable failures | 47 |
| false positive switches | 28 |
| introduced failures | 0 |

### Threshold Sweep

| Threshold | Replay-infeasible Rate | Progress | ADE3 | Switch Rate | Missed Avoidable |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.058 | 12.968 | 6.052 | 0.697 | 4 |
| 0.30 | 0.084 | 13.560 | 5.794 | 0.665 | 8 |
| 0.40 | 0.303 | 18.511 | 3.645 | 0.445 | 42 |
| 0.50 | 0.335 | 19.218 | 3.343 | 0.406 | 47 |
| 0.60 | 0.348 | 19.498 | 3.225 | 0.394 | 49 |
| 0.70 | 0.381 | 19.892 | 3.053 | 0.303 | 54 |
| 0.80 | 0.406 | 20.579 | 2.826 | 0.187 | 58 |

