# nuPlan Failure-Gated Selector

- Input replay JSON: `artifacts/corl2027/nuplan_public_replay_study_interaction1000_expert/replay.json`
- Safe selector: `fit_on_train_split`
- Split unit: `source_db_file`
- Train scenes: `700`
- Holdout scenes: `300`
- Gate threshold: `0.500`
- Train positive rate: `0.367`
- Train gate accuracy: `0.884`

## Train

| Selector | Replay-infeasible | Rate | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|
| proxy_selector | 428 | 0.611 | 27.518 | 2.392 | 2.472 |
| safe_selector | 171 | 0.244 | 17.642 | 5.699 | 2.742 |
| failure_gate | 174 | 0.249 | 20.282 | 4.878 | 2.650 |
| replay_oracle | 163 | 0.233 | 13.043 | 7.658 | 2.571 |

### Gate Diagnostics

| Metric | Value |
|---|---:|
| switch count | 332 |
| switch rate | 0.474 |
| true positive switches | 254 |
| missed avoidable failures | 3 |
| false positive switches | 78 |
| introduced failures | 0 |

### Threshold Sweep

| Threshold | Replay-infeasible Rate | Progress | ADE3 | Switch Rate | Missed Avoidable |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.244 | 19.177 | 5.313 | 0.587 | 0 |
| 0.30 | 0.244 | 19.378 | 5.244 | 0.527 | 0 |
| 0.40 | 0.244 | 19.849 | 5.053 | 0.497 | 0 |
| 0.50 | 0.249 | 20.282 | 4.878 | 0.474 | 3 |
| 0.60 | 0.256 | 20.657 | 4.725 | 0.440 | 8 |
| 0.70 | 0.276 | 22.050 | 4.213 | 0.386 | 22 |
| 0.80 | 0.310 | 23.018 | 3.893 | 0.333 | 46 |

## Holdout

| Selector | Replay-infeasible | Rate | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|
| proxy_selector | 199 | 0.663 | 28.998 | 2.658 | 1.882 |
| safe_selector | 154 | 0.513 | 19.109 | 5.465 | 2.575 |
| failure_gate | 175 | 0.583 | 27.177 | 3.283 | 1.946 |
| replay_oracle | 118 | 0.393 | 11.346 | 8.587 | 2.613 |

### Gate Diagnostics

| Metric | Value |
|---|---:|
| switch count | 47 |
| switch rate | 0.157 |
| true positive switches | 24 |
| missed avoidable failures | 32 |
| false positive switches | 23 |
| introduced failures | 0 |

### Threshold Sweep

| Threshold | Replay-infeasible Rate | Progress | ADE3 | Switch Rate | Missed Avoidable |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.507 | 22.852 | 4.662 | 0.373 | 8 |
| 0.30 | 0.540 | 26.210 | 3.554 | 0.223 | 19 |
| 0.40 | 0.567 | 26.989 | 3.349 | 0.173 | 27 |
| 0.50 | 0.583 | 27.177 | 3.283 | 0.157 | 32 |
| 0.60 | 0.583 | 27.177 | 3.283 | 0.157 | 32 |
| 0.70 | 0.583 | 27.177 | 3.283 | 0.147 | 32 |
| 0.80 | 0.587 | 27.309 | 3.226 | 0.123 | 33 |

## Train-Selected Operating Points

| Policy | Threshold | Holdout Replay-infeasible Rate | Holdout Progress | Holdout ADE3 |
|---|---:|---:|---:|---:|
| safety_first | 0.40 | 0.567 | 26.989 | 3.349 |
| balanced_80pct_proxy_progress | 0.70 | 0.583 | 27.177 | 3.283 |
| utility_90pct_proxy_progress | 0.40 | 0.567 | 26.989 | 3.349 |

