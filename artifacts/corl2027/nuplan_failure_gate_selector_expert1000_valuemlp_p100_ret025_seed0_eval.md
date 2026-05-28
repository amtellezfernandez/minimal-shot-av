# nuPlan Failure-Gated Selector

- Input replay JSON: `artifacts/corl2027/nuplan_public_replay_study_interaction1000_expert/replay.json`
- Safe selector: `fit_replay_value_mlp_on_train_split`
- Fallback mode: `score`
- Fallback risk threshold: `0.500`
- Split unit: `source_db_file`
- Train scenes: `700`
- Holdout scenes: `300`
- Gate threshold: `0.500`
- Train positive rate: `0.160`
- Train gate accuracy: `0.944`

## Train

| Selector | Replay-infeasible | Rate | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|
| proxy_selector | 418 | 0.597 | 27.353 | 2.566 | 2.142 |
| safe_selector | 247 | 0.353 | 22.060 | 3.997 | 2.906 |
| oracle_failure_gate | 306 | 0.437 | 25.943 | 2.923 | 2.279 |
| progress_oracle_failure_gate | 232 | 0.331 | 23.796 | 3.618 | 2.562 |
| failure_gate | 309 | 0.441 | 25.545 | 3.056 | 2.278 |
| replay_oracle | 232 | 0.331 | 13.239 | 7.340 | 2.703 |

### Gate Diagnostics

| Metric | Value |
|---|---:|
| switch count | 145 |
| switch rate | 0.207 |
| true positive switches | 109 |
| missed avoidable failures | 3 |
| false positive switches | 36 |
| introduced failures | 0 |
| switch precision | 0.752 |
| switch recall | 0.973 |
| oracle switch count | 112 |

### Threshold Sweep

| Threshold | Replay-infeasible Rate | Progress | ADE3 | Switch Rate | Missed Avoidable |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.437 | 25.357 | 3.050 | 0.251 | 0 |
| 0.30 | 0.437 | 25.400 | 3.057 | 0.233 | 0 |
| 0.40 | 0.441 | 25.499 | 3.064 | 0.211 | 3 |
| 0.50 | 0.441 | 25.545 | 3.056 | 0.207 | 3 |
| 0.60 | 0.441 | 25.599 | 3.047 | 0.200 | 3 |
| 0.70 | 0.447 | 25.721 | 3.011 | 0.181 | 7 |
| 0.80 | 0.459 | 25.942 | 2.929 | 0.157 | 15 |

## Holdout

| Selector | Replay-infeasible | Rate | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|
| proxy_selector | 209 | 0.697 | 29.383 | 2.251 | 2.240 |
| safe_selector | 53 | 0.177 | 19.765 | 5.768 | 2.253 |
| oracle_failure_gate | 127 | 0.423 | 26.944 | 3.133 | 2.399 |
| progress_oracle_failure_gate | 49 | 0.163 | 21.114 | 5.297 | 2.354 |
| failure_gate | 137 | 0.457 | 26.856 | 3.152 | 2.360 |
| replay_oracle | 49 | 0.163 | 10.889 | 9.329 | 2.344 |

### Gate Diagnostics

| Metric | Value |
|---|---:|
| switch count | 111 |
| switch rate | 0.370 |
| true positive switches | 72 |
| missed avoidable failures | 10 |
| false positive switches | 39 |
| introduced failures | 0 |
| switch precision | 0.649 |
| switch recall | 0.878 |
| oracle switch count | 82 |

### Threshold Sweep

| Threshold | Replay-infeasible Rate | Progress | ADE3 | Switch Rate | Missed Avoidable |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.437 | 26.595 | 3.233 | 0.413 | 4 |
| 0.30 | 0.443 | 26.693 | 3.208 | 0.407 | 6 |
| 0.40 | 0.450 | 26.758 | 3.177 | 0.400 | 8 |
| 0.50 | 0.457 | 26.856 | 3.152 | 0.370 | 10 |
| 0.60 | 0.477 | 27.116 | 3.082 | 0.313 | 16 |
| 0.70 | 0.487 | 27.193 | 3.054 | 0.283 | 19 |
| 0.80 | 0.487 | 27.193 | 3.054 | 0.277 | 19 |

## Train-Selected Operating Points

| Policy | Threshold | Holdout Replay-infeasible Rate | Holdout Progress | Holdout ADE3 |
|---|---:|---:|---:|---:|
| safety_first | 0.30 | 0.443 | 26.693 | 3.208 |
| balanced_80pct_proxy_progress | 0.30 | 0.443 | 26.693 | 3.208 |
| utility_90pct_proxy_progress | 0.30 | 0.443 | 26.693 | 3.208 |

