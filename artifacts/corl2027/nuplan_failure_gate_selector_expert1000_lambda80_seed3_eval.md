# nuPlan Failure-Gated Selector

- Input replay JSON: `artifacts/corl2027/nuplan_public_replay_study_interaction1000_expert/replay.json`
- Safe selector: `artifacts/corl2027/nuplan_replay_calibrated_selector_expert1000_lambda80.json`
- Split unit: `source_db_file`
- Train scenes: `700`
- Holdout scenes: `300`
- Gate threshold: `0.500`
- Train positive rate: `0.389`
- Train gate accuracy: `0.903`

## Train

| Selector | Replay-infeasible | Rate | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|
| proxy_selector | 380 | 0.543 | 25.728 | 2.581 | 2.376 |
| safe_selector | 108 | 0.154 | 18.003 | 4.994 | 2.786 |
| failure_gate | 108 | 0.154 | 19.220 | 4.728 | 2.646 |
| replay_oracle | 104 | 0.149 | 11.221 | 7.702 | 2.677 |

### Gate Diagnostics

| Metric | Value |
|---|---:|
| switch count | 340 |
| switch rate | 0.486 |
| true positive switches | 272 |
| missed avoidable failures | 0 |
| false positive switches | 68 |
| introduced failures | 0 |

### Threshold Sweep

| Threshold | Replay-infeasible Rate | Progress | ADE3 | Switch Rate | Missed Avoidable |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.154 | 18.726 | 4.866 | 0.653 | 0 |
| 0.30 | 0.154 | 18.934 | 4.805 | 0.571 | 0 |
| 0.40 | 0.154 | 19.022 | 4.788 | 0.527 | 0 |
| 0.50 | 0.154 | 19.220 | 4.728 | 0.486 | 0 |
| 0.60 | 0.160 | 19.726 | 4.555 | 0.431 | 4 |
| 0.70 | 0.187 | 20.424 | 4.334 | 0.373 | 23 |
| 0.80 | 0.219 | 21.135 | 4.119 | 0.340 | 45 |

## Holdout

| Selector | Replay-infeasible | Rate | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|
| proxy_selector | 247 | 0.823 | 33.173 | 2.217 | 2.082 |
| safe_selector | 186 | 0.620 | 27.475 | 4.151 | 2.367 |
| failure_gate | 197 | 0.657 | 29.390 | 3.514 | 2.145 |
| replay_oracle | 177 | 0.590 | 15.598 | 8.484 | 2.197 |

### Gate Diagnostics

| Metric | Value |
|---|---:|
| switch count | 106 |
| switch rate | 0.353 |
| true positive switches | 50 |
| missed avoidable failures | 11 |
| false positive switches | 56 |
| introduced failures | 0 |

### Threshold Sweep

| Threshold | Replay-infeasible Rate | Progress | ADE3 | Switch Rate | Missed Avoidable |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.640 | 28.095 | 3.929 | 0.493 | 6 |
| 0.30 | 0.640 | 28.151 | 3.911 | 0.433 | 6 |
| 0.40 | 0.643 | 28.795 | 3.691 | 0.387 | 7 |
| 0.50 | 0.657 | 29.390 | 3.514 | 0.353 | 11 |
| 0.60 | 0.673 | 29.595 | 3.492 | 0.323 | 16 |
| 0.70 | 0.677 | 30.160 | 3.326 | 0.290 | 17 |
| 0.80 | 0.680 | 30.411 | 3.222 | 0.240 | 18 |

