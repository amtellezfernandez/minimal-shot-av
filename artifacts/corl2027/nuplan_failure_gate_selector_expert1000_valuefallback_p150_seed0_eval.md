# nuPlan Failure-Gated Selector

- Input replay JSON: `artifacts/corl2027/nuplan_public_replay_study_interaction1000_expert/replay.json`
- Safe selector: `fit_replay_value_on_train_split`
- Fallback mode: `score`
- Fallback risk threshold: `0.500`
- Split unit: `source_db_file`
- Train scenes: `700`
- Holdout scenes: `300`
- Gate threshold: `0.500`
- Train positive rate: `0.229`
- Train gate accuracy: `0.950`

## Train

| Selector | Replay-infeasible | Rate | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|
| proxy_selector | 418 | 0.597 | 27.353 | 2.566 | 2.142 |
| safe_selector | 272 | 0.389 | 20.492 | 4.981 | 2.462 |
| oracle_failure_gate | 258 | 0.369 | 23.807 | 3.654 | 2.497 |
| progress_oracle_failure_gate | 232 | 0.331 | 23.796 | 3.618 | 2.562 |
| failure_gate | 258 | 0.369 | 23.098 | 3.907 | 2.509 |
| replay_oracle | 232 | 0.331 | 13.239 | 7.340 | 2.703 |

### Gate Diagnostics

| Metric | Value |
|---|---:|
| switch count | 195 |
| switch rate | 0.279 |
| true positive switches | 160 |
| missed avoidable failures | 0 |
| false positive switches | 35 |
| introduced failures | 0 |
| switch precision | 0.821 |
| switch recall | 1.000 |
| oracle switch count | 160 |

### Threshold Sweep

| Threshold | Replay-infeasible Rate | Progress | ADE3 | Switch Rate | Missed Avoidable |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.369 | 21.803 | 4.371 | 0.321 | 0 |
| 0.30 | 0.369 | 22.099 | 4.269 | 0.303 | 0 |
| 0.40 | 0.369 | 22.870 | 3.987 | 0.287 | 0 |
| 0.50 | 0.369 | 23.098 | 3.907 | 0.279 | 0 |
| 0.60 | 0.369 | 23.326 | 3.827 | 0.266 | 0 |
| 0.70 | 0.369 | 23.605 | 3.716 | 0.247 | 0 |
| 0.80 | 0.390 | 24.056 | 3.556 | 0.221 | 15 |

## Holdout

| Selector | Replay-infeasible | Rate | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|
| proxy_selector | 209 | 0.697 | 29.383 | 2.251 | 2.240 |
| safe_selector | 50 | 0.167 | 15.489 | 7.696 | 2.274 |
| oracle_failure_gate | 49 | 0.163 | 17.810 | 6.749 | 2.346 |
| progress_oracle_failure_gate | 49 | 0.163 | 21.114 | 5.297 | 2.354 |
| failure_gate | 68 | 0.227 | 18.161 | 6.600 | 2.316 |
| replay_oracle | 49 | 0.163 | 10.889 | 9.329 | 2.344 |

### Gate Diagnostics

| Metric | Value |
|---|---:|
| switch count | 169 |
| switch rate | 0.563 |
| true positive switches | 141 |
| missed avoidable failures | 19 |
| false positive switches | 28 |
| introduced failures | 0 |
| switch precision | 0.834 |
| switch recall | 0.881 |
| oracle switch count | 160 |

### Threshold Sweep

| Threshold | Replay-infeasible Rate | Progress | ADE3 | Switch Rate | Missed Avoidable |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.177 | 15.695 | 7.547 | 0.657 | 4 |
| 0.30 | 0.190 | 16.512 | 7.240 | 0.627 | 8 |
| 0.40 | 0.200 | 17.449 | 6.882 | 0.590 | 11 |
| 0.50 | 0.227 | 18.161 | 6.600 | 0.563 | 19 |
| 0.60 | 0.240 | 18.397 | 6.524 | 0.547 | 23 |
| 0.70 | 0.250 | 18.636 | 6.428 | 0.527 | 26 |
| 0.80 | 0.303 | 19.999 | 6.033 | 0.450 | 42 |

## Train-Selected Operating Points

| Policy | Threshold | Holdout Replay-infeasible Rate | Holdout Progress | Holdout ADE3 |
|---|---:|---:|---:|---:|
| safety_first | 0.70 | 0.250 | 18.636 | 6.428 |
| balanced_80pct_proxy_progress | 0.70 | 0.250 | 18.636 | 6.428 |
| utility_90pct_proxy_progress | 0.70 | 0.250 | 18.636 | 6.428 |

