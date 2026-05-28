# nuPlan Failure-Gated Selector

- Input replay JSON: `artifacts/corl2027/nuplan_public_replay_study_interaction1000_expert/replay.json`
- Safe selector: `fit_replay_value_mlp_on_train_split`
- Fallback mode: `score`
- Fallback risk threshold: `0.500`
- Split unit: `source_db_file`
- Train scenes: `845`
- Holdout scenes: `155`
- Gate threshold: `0.500`
- Train positive rate: `0.316`
- Train gate accuracy: `0.940`

## Train

| Selector | Replay-infeasible | Rate | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|
| proxy_selector | 547 | 0.647 | 28.472 | 2.388 | 2.411 |
| safe_selector | 282 | 0.334 | 21.439 | 4.686 | 2.880 |
| oracle_failure_gate | 280 | 0.331 | 23.483 | 4.164 | 2.638 |
| progress_oracle_failure_gate | 263 | 0.311 | 23.449 | 4.146 | 2.677 |
| failure_gate | 283 | 0.335 | 22.816 | 4.381 | 2.613 |
| replay_oracle | 263 | 0.311 | 12.179 | 8.327 | 2.591 |

### Gate Diagnostics

| Metric | Value |
|---|---:|
| switch count | 316 |
| switch rate | 0.374 |
| true positive switches | 266 |
| missed avoidable failures | 1 |
| false positive switches | 50 |
| introduced failures | 2 |
| switch precision | 0.842 |
| switch recall | 0.996 |
| oracle switch count | 267 |

### Threshold Sweep

| Threshold | Replay-infeasible Rate | Progress | ADE3 | Switch Rate | Missed Avoidable |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.334 | 22.304 | 4.598 | 0.428 | 0 |
| 0.30 | 0.334 | 22.410 | 4.557 | 0.414 | 0 |
| 0.40 | 0.334 | 22.568 | 4.486 | 0.396 | 0 |
| 0.50 | 0.335 | 22.816 | 4.381 | 0.374 | 1 |
| 0.60 | 0.340 | 22.906 | 4.342 | 0.364 | 6 |
| 0.70 | 0.340 | 23.077 | 4.278 | 0.351 | 7 |
| 0.80 | 0.349 | 23.418 | 4.162 | 0.318 | 15 |

## Holdout

| Selector | Replay-infeasible | Rate | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|
| proxy_selector | 80 | 0.516 | 25.180 | 2.929 | 1.595 |
| safe_selector | 18 | 0.116 | 14.973 | 5.179 | 2.842 |
| oracle_failure_gate | 18 | 0.116 | 19.269 | 4.355 | 2.238 |
| progress_oracle_failure_gate | 18 | 0.116 | 20.497 | 3.988 | 2.424 |
| failure_gate | 34 | 0.219 | 19.944 | 4.270 | 2.315 |
| replay_oracle | 18 | 0.116 | 14.471 | 5.808 | 2.847 |

### Gate Diagnostics

| Metric | Value |
|---|---:|
| switch count | 67 |
| switch rate | 0.432 |
| true positive switches | 46 |
| missed avoidable failures | 16 |
| false positive switches | 21 |
| introduced failures | 0 |
| switch precision | 0.687 |
| switch recall | 0.742 |
| oracle switch count | 62 |

### Threshold Sweep

| Threshold | Replay-infeasible Rate | Progress | ADE3 | Switch Rate | Missed Avoidable |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.161 | 18.190 | 4.779 | 0.561 | 7 |
| 0.30 | 0.168 | 18.561 | 4.640 | 0.529 | 8 |
| 0.40 | 0.200 | 19.066 | 4.473 | 0.484 | 13 |
| 0.50 | 0.219 | 19.944 | 4.270 | 0.432 | 16 |
| 0.60 | 0.310 | 21.178 | 3.952 | 0.329 | 30 |
| 0.70 | 0.348 | 21.620 | 3.810 | 0.290 | 36 |
| 0.80 | 0.439 | 23.476 | 3.342 | 0.135 | 50 |

## Train-Selected Operating Points

| Policy | Threshold | Holdout Replay-infeasible Rate | Holdout Progress | Holdout ADE3 |
|---|---:|---:|---:|---:|
| safety_first | 0.40 | 0.200 | 19.066 | 4.473 |
| balanced_80pct_proxy_progress | 0.50 | 0.219 | 19.944 | 4.270 |
| utility_90pct_proxy_progress | 0.40 | 0.200 | 19.066 | 4.473 |

