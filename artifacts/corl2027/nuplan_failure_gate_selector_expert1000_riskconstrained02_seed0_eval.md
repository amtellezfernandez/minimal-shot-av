# nuPlan Failure-Gated Selector

- Input replay JSON: `artifacts/corl2027/nuplan_public_replay_study_interaction1000_expert/replay.json`
- Safe selector: `fit_on_train_split`
- Fallback mode: `risk_constrained`
- Fallback risk threshold: `0.200`
- Split unit: `source_db_file`
- Train scenes: `700`
- Holdout scenes: `300`
- Gate threshold: `0.500`
- Train positive rate: `0.261`
- Train gate accuracy: `0.931`

## Train

| Selector | Replay-infeasible | Rate | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|
| proxy_selector | 418 | 0.597 | 27.353 | 2.566 | 2.142 |
| safe_selector | 239 | 0.341 | 13.198 | 7.185 | 2.794 |
| oracle_failure_gate | 235 | 0.336 | 23.350 | 3.759 | 2.577 |
| failure_gate | 237 | 0.339 | 22.408 | 4.006 | 2.647 |
| replay_oracle | 232 | 0.331 | 13.239 | 7.340 | 2.703 |

### Gate Diagnostics

| Metric | Value |
|---|---:|
| switch count | 231 |
| switch rate | 0.330 |
| true positive switches | 183 |
| missed avoidable failures | 0 |
| false positive switches | 48 |
| introduced failures | 2 |
| switch precision | 0.792 |
| switch recall | 1.000 |
| oracle switch count | 183 |

### Threshold Sweep

| Threshold | Replay-infeasible Rate | Progress | ADE3 | Switch Rate | Missed Avoidable |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.339 | 21.047 | 4.417 | 0.386 | 0 |
| 0.30 | 0.339 | 21.927 | 4.155 | 0.351 | 0 |
| 0.40 | 0.339 | 22.173 | 4.075 | 0.341 | 0 |
| 0.50 | 0.339 | 22.408 | 4.006 | 0.330 | 0 |
| 0.60 | 0.336 | 22.791 | 3.901 | 0.311 | 0 |
| 0.70 | 0.340 | 23.120 | 3.808 | 0.287 | 3 |
| 0.80 | 0.357 | 23.593 | 3.658 | 0.263 | 15 |

## Holdout

| Selector | Replay-infeasible | Rate | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|
| proxy_selector | 209 | 0.697 | 29.383 | 2.251 | 2.240 |
| safe_selector | 49 | 0.163 | 12.533 | 8.626 | 2.397 |
| oracle_failure_gate | 49 | 0.163 | 19.477 | 6.011 | 2.470 |
| failure_gate | 112 | 0.373 | 23.469 | 4.330 | 2.570 |
| replay_oracle | 49 | 0.163 | 10.889 | 9.329 | 2.344 |

### Gate Diagnostics

| Metric | Value |
|---|---:|
| switch count | 125 |
| switch rate | 0.417 |
| true positive switches | 97 |
| missed avoidable failures | 63 |
| false positive switches | 28 |
| introduced failures | 0 |
| switch precision | 0.776 |
| switch recall | 0.606 |
| oracle switch count | 160 |

### Threshold Sweep

| Threshold | Replay-infeasible Rate | Progress | ADE3 | Switch Rate | Missed Avoidable |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.207 | 18.933 | 6.247 | 0.623 | 13 |
| 0.30 | 0.293 | 21.307 | 5.238 | 0.500 | 39 |
| 0.40 | 0.363 | 23.313 | 4.390 | 0.427 | 60 |
| 0.50 | 0.373 | 23.469 | 4.330 | 0.417 | 63 |
| 0.60 | 0.387 | 23.596 | 4.289 | 0.403 | 67 |
| 0.70 | 0.390 | 23.960 | 4.139 | 0.377 | 68 |
| 0.80 | 0.397 | 24.535 | 3.890 | 0.330 | 70 |

## Train-Selected Operating Points

| Policy | Threshold | Holdout Replay-infeasible Rate | Holdout Progress | Holdout ADE3 |
|---|---:|---:|---:|---:|
| safety_first | 0.60 | 0.387 | 23.596 | 4.289 |
| balanced_80pct_proxy_progress | 0.60 | 0.387 | 23.596 | 4.289 |
| utility_90pct_proxy_progress | 0.60 | 0.387 | 23.596 | 4.289 |

