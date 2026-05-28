# nuPlan Failure-Gated Selector

- Input replay JSON: `artifacts/corl2027/nuplan_public_replay_study_interaction1000_expert/replay.json`
- Safe selector: `fit_replay_value_mlp_on_train_split`
- Fallback mode: `score`
- Fallback risk threshold: `0.500`
- Split unit: `source_db_file`
- Train scenes: `700`
- Holdout scenes: `300`
- Gate threshold: `0.500`
- Train positive rate: `0.137`
- Train gate accuracy: `0.947`

## Train

| Selector | Replay-infeasible | Rate | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|
| proxy_selector | 418 | 0.597 | 27.353 | 2.566 | 2.142 |
| safe_selector | 247 | 0.353 | 22.060 | 3.997 | 2.906 |
| oracle_failure_gate | 322 | 0.460 | 26.188 | 2.821 | 2.185 |
| progress_oracle_failure_gate | 232 | 0.331 | 23.796 | 3.618 | 2.562 |
| failure_gate | 325 | 0.464 | 25.852 | 2.928 | 2.169 |
| replay_oracle | 232 | 0.331 | 13.239 | 7.340 | 2.703 |

### Gate Diagnostics

| Metric | Value |
|---|---:|
| switch count | 126 |
| switch rate | 0.180 |
| true positive switches | 93 |
| missed avoidable failures | 3 |
| false positive switches | 33 |
| introduced failures | 0 |
| switch precision | 0.738 |
| switch recall | 0.969 |
| oracle switch count | 96 |

### Threshold Sweep

| Threshold | Replay-infeasible Rate | Progress | ADE3 | Switch Rate | Missed Avoidable |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.460 | 25.647 | 2.928 | 0.229 | 0 |
| 0.30 | 0.461 | 25.715 | 2.944 | 0.199 | 1 |
| 0.40 | 0.464 | 25.811 | 2.938 | 0.183 | 3 |
| 0.50 | 0.464 | 25.852 | 2.928 | 0.180 | 3 |
| 0.60 | 0.466 | 25.911 | 2.921 | 0.171 | 4 |
| 0.70 | 0.470 | 25.999 | 2.890 | 0.157 | 7 |
| 0.80 | 0.481 | 26.232 | 2.807 | 0.130 | 15 |

## Holdout

| Selector | Replay-infeasible | Rate | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|
| proxy_selector | 209 | 0.697 | 29.383 | 2.251 | 2.240 |
| safe_selector | 53 | 0.177 | 19.765 | 5.768 | 2.253 |
| oracle_failure_gate | 132 | 0.440 | 27.114 | 3.058 | 2.306 |
| progress_oracle_failure_gate | 49 | 0.163 | 21.114 | 5.297 | 2.354 |
| failure_gate | 148 | 0.493 | 27.386 | 2.973 | 2.190 |
| replay_oracle | 49 | 0.163 | 10.889 | 9.329 | 2.344 |

### Gate Diagnostics

| Metric | Value |
|---|---:|
| switch count | 84 |
| switch rate | 0.280 |
| true positive switches | 61 |
| missed avoidable failures | 16 |
| false positive switches | 23 |
| introduced failures | 0 |
| switch precision | 0.726 |
| switch recall | 0.792 |
| oracle switch count | 77 |

### Threshold Sweep

| Threshold | Replay-infeasible Rate | Progress | ADE3 | Switch Rate | Missed Avoidable |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.487 | 27.289 | 2.999 | 0.350 | 14 |
| 0.30 | 0.490 | 27.337 | 2.986 | 0.310 | 15 |
| 0.40 | 0.490 | 27.337 | 2.986 | 0.283 | 15 |
| 0.50 | 0.493 | 27.386 | 2.973 | 0.280 | 16 |
| 0.60 | 0.503 | 27.429 | 2.951 | 0.267 | 19 |
| 0.70 | 0.503 | 27.452 | 2.944 | 0.263 | 19 |
| 0.80 | 0.503 | 27.563 | 2.911 | 0.240 | 19 |

## Train-Selected Operating Points

| Policy | Threshold | Holdout Replay-infeasible Rate | Holdout Progress | Holdout ADE3 |
|---|---:|---:|---:|---:|
| safety_first | 0.20 | 0.487 | 27.289 | 2.999 |
| balanced_80pct_proxy_progress | 0.20 | 0.487 | 27.289 | 2.999 |
| utility_90pct_proxy_progress | 0.20 | 0.487 | 27.289 | 2.999 |

