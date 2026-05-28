# nuPlan Failure-Gated Selector

- Input replay JSON: `artifacts/corl2027/nuplan_public_replay_study_interaction1000_expert/replay.json`
- Safe selector: `artifacts/corl2027/nuplan_replay_calibrated_selector_expert1000_lambda80.json`
- Split unit: `source_db_file`
- Train scenes: `700`
- Holdout scenes: `300`
- Gate threshold: `0.500`
- Train positive rate: `0.373`
- Train gate accuracy: `0.887`

## Train

| Selector | Replay-infeasible | Rate | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|
| proxy_selector | 428 | 0.611 | 27.518 | 2.392 | 2.472 |
| safe_selector | 167 | 0.239 | 19.994 | 4.888 | 2.694 |
| failure_gate | 170 | 0.243 | 21.643 | 4.372 | 2.578 |
| replay_oracle | 163 | 0.233 | 13.043 | 7.658 | 2.571 |

### Gate Diagnostics

| Metric | Value |
|---|---:|
| switch count | 334 |
| switch rate | 0.477 |
| true positive switches | 258 |
| missed avoidable failures | 3 |
| false positive switches | 76 |
| introduced failures | 0 |

### Threshold Sweep

| Threshold | Replay-infeasible Rate | Progress | ADE3 | Switch Rate | Missed Avoidable |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.239 | 20.990 | 4.644 | 0.586 | 0 |
| 0.30 | 0.239 | 21.072 | 4.609 | 0.526 | 0 |
| 0.40 | 0.239 | 21.238 | 4.537 | 0.497 | 0 |
| 0.50 | 0.243 | 21.643 | 4.372 | 0.477 | 3 |
| 0.60 | 0.249 | 21.950 | 4.246 | 0.446 | 7 |
| 0.70 | 0.264 | 22.309 | 4.112 | 0.400 | 18 |
| 0.80 | 0.299 | 23.210 | 3.822 | 0.344 | 42 |

## Holdout

| Selector | Replay-infeasible | Rate | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|
| proxy_selector | 199 | 0.663 | 28.998 | 2.658 | 1.882 |
| safe_selector | 127 | 0.423 | 22.830 | 4.400 | 2.835 |
| failure_gate | 175 | 0.583 | 27.686 | 3.047 | 2.072 |
| replay_oracle | 118 | 0.393 | 11.346 | 8.587 | 2.613 |

### Gate Diagnostics

| Metric | Value |
|---|---:|
| switch count | 47 |
| switch rate | 0.157 |
| true positive switches | 24 |
| missed avoidable failures | 48 |
| false positive switches | 23 |
| introduced failures | 0 |

### Threshold Sweep

| Threshold | Replay-infeasible Rate | Progress | ADE3 | Switch Rate | Missed Avoidable |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.483 | 24.901 | 3.975 | 0.360 | 18 |
| 0.30 | 0.540 | 26.906 | 3.301 | 0.220 | 35 |
| 0.40 | 0.570 | 27.505 | 3.109 | 0.170 | 44 |
| 0.50 | 0.583 | 27.686 | 3.047 | 0.157 | 48 |
| 0.60 | 0.583 | 27.686 | 3.047 | 0.157 | 48 |
| 0.70 | 0.583 | 27.686 | 3.047 | 0.147 | 48 |
| 0.80 | 0.587 | 27.750 | 3.017 | 0.123 | 49 |

