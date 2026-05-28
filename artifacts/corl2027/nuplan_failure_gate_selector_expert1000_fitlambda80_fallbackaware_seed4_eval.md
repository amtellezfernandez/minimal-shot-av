# nuPlan Failure-Gated Selector

- Input replay JSON: `artifacts/corl2027/nuplan_public_replay_study_interaction1000_expert/replay.json`
- Safe selector: `fit_on_train_split`
- Fallback mode: `score`
- Fallback risk threshold: `0.500`
- Split unit: `source_db_file`
- Train scenes: `845`
- Holdout scenes: `155`
- Gate threshold: `0.500`
- Train positive rate: `0.288`
- Train gate accuracy: `0.933`

## Train

| Selector | Replay-infeasible | Rate | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|
| proxy_selector | 540 | 0.639 | 28.784 | 2.529 | 2.224 |
| safe_selector | 298 | 0.353 | 20.815 | 5.038 | 2.961 |
| oracle_failure_gate | 297 | 0.351 | 25.095 | 3.694 | 2.555 |
| progress_oracle_failure_gate | 276 | 0.327 | 24.816 | 3.775 | 2.622 |
| failure_gate | 301 | 0.356 | 24.483 | 3.858 | 2.563 |
| replay_oracle | 276 | 0.327 | 13.159 | 8.004 | 2.710 |

### Gate Diagnostics

| Metric | Value |
|---|---:|
| switch count | 292 |
| switch rate | 0.346 |
| true positive switches | 239 |
| missed avoidable failures | 4 |
| false positive switches | 53 |
| introduced failures | 0 |
| switch precision | 0.818 |
| switch recall | 0.984 |
| oracle switch count | 243 |

### Threshold Sweep

| Threshold | Replay-infeasible Rate | Progress | ADE3 | Switch Rate | Missed Avoidable |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.353 | 22.733 | 4.437 | 0.425 | 0 |
| 0.30 | 0.353 | 23.284 | 4.246 | 0.402 | 0 |
| 0.40 | 0.353 | 23.934 | 4.043 | 0.373 | 1 |
| 0.50 | 0.356 | 24.483 | 3.858 | 0.346 | 4 |
| 0.60 | 0.360 | 24.706 | 3.785 | 0.333 | 7 |
| 0.70 | 0.363 | 25.011 | 3.693 | 0.307 | 10 |
| 0.80 | 0.379 | 25.379 | 3.582 | 0.282 | 23 |

## Holdout

| Selector | Replay-infeasible | Rate | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|
| proxy_selector | 87 | 0.561 | 23.482 | 2.159 | 2.253 |
| safe_selector | 47 | 0.303 | 18.140 | 3.707 | 2.249 |
| oracle_failure_gate | 47 | 0.303 | 19.042 | 3.437 | 2.287 |
| progress_oracle_failure_gate | 5 | 0.032 | 13.046 | 6.008 | 2.003 |
| failure_gate | 51 | 0.329 | 19.129 | 3.379 | 2.265 |
| replay_oracle | 5 | 0.032 | 9.125 | 7.570 | 2.218 |

### Gate Diagnostics

| Metric | Value |
|---|---:|
| switch count | 53 |
| switch rate | 0.342 |
| true positive switches | 36 |
| missed avoidable failures | 4 |
| false positive switches | 17 |
| introduced failures | 0 |
| switch precision | 0.679 |
| switch recall | 0.900 |
| oracle switch count | 40 |

### Threshold Sweep

| Threshold | Replay-infeasible Rate | Progress | ADE3 | Switch Rate | Missed Avoidable |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.323 | 19.104 | 3.393 | 0.690 | 3 |
| 0.30 | 0.329 | 19.129 | 3.379 | 0.677 | 4 |
| 0.40 | 0.329 | 19.129 | 3.379 | 0.594 | 4 |
| 0.50 | 0.329 | 19.129 | 3.379 | 0.342 | 4 |
| 0.60 | 0.329 | 19.129 | 3.379 | 0.310 | 4 |
| 0.70 | 0.329 | 19.129 | 3.379 | 0.232 | 4 |
| 0.80 | 0.329 | 19.129 | 3.379 | 0.232 | 4 |

## Train-Selected Operating Points

| Policy | Threshold | Holdout Replay-infeasible Rate | Holdout Progress | Holdout ADE3 |
|---|---:|---:|---:|---:|
| safety_first | 0.40 | 0.329 | 19.129 | 3.379 |
| balanced_80pct_proxy_progress | 0.40 | 0.329 | 19.129 | 3.379 |
| utility_90pct_proxy_progress | 0.40 | 0.329 | 19.129 | 3.379 |

