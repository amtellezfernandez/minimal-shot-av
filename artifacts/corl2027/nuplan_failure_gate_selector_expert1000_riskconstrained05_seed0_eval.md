# nuPlan Failure-Gated Selector

- Input replay JSON: `artifacts/corl2027/nuplan_public_replay_study_interaction1000_expert/replay.json`
- Safe selector: `fit_on_train_split`
- Fallback mode: `risk_constrained`
- Fallback risk threshold: `0.500`
- Split unit: `source_db_file`
- Train scenes: `700`
- Holdout scenes: `300`
- Gate threshold: `0.500`
- Train positive rate: `0.219`
- Train gate accuracy: `0.967`

## Train

| Selector | Replay-infeasible | Rate | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|
| proxy_selector | 418 | 0.597 | 27.353 | 2.566 | 2.142 |
| safe_selector | 268 | 0.383 | 15.695 | 6.631 | 2.806 |
| oracle_failure_gate | 265 | 0.379 | 24.172 | 3.583 | 2.512 |
| failure_gate | 266 | 0.380 | 23.745 | 3.725 | 2.531 |
| replay_oracle | 232 | 0.331 | 13.239 | 7.340 | 2.703 |

### Gate Diagnostics

| Metric | Value |
|---|---:|
| switch count | 176 |
| switch rate | 0.251 |
| true positive switches | 153 |
| missed avoidable failures | 0 |
| false positive switches | 23 |
| introduced failures | 1 |
| switch precision | 0.869 |
| switch recall | 1.000 |
| oracle switch count | 153 |

### Threshold Sweep

| Threshold | Replay-infeasible Rate | Progress | ADE3 | Switch Rate | Missed Avoidable |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.380 | 22.826 | 4.029 | 0.289 | 0 |
| 0.30 | 0.380 | 23.400 | 3.858 | 0.263 | 0 |
| 0.40 | 0.380 | 23.554 | 3.800 | 0.257 | 0 |
| 0.50 | 0.380 | 23.745 | 3.725 | 0.251 | 0 |
| 0.60 | 0.379 | 23.846 | 3.686 | 0.241 | 0 |
| 0.70 | 0.379 | 23.990 | 3.632 | 0.236 | 0 |
| 0.80 | 0.389 | 24.196 | 3.556 | 0.226 | 7 |

## Holdout

| Selector | Replay-infeasible | Rate | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|
| proxy_selector | 209 | 0.697 | 29.383 | 2.251 | 2.240 |
| safe_selector | 84 | 0.280 | 14.194 | 7.954 | 2.500 |
| oracle_failure_gate | 84 | 0.280 | 20.798 | 5.493 | 2.480 |
| failure_gate | 140 | 0.467 | 24.757 | 3.821 | 2.609 |
| replay_oracle | 49 | 0.163 | 10.889 | 9.329 | 2.344 |

### Gate Diagnostics

| Metric | Value |
|---|---:|
| switch count | 148 |
| switch rate | 0.493 |
| true positive switches | 69 |
| missed avoidable failures | 56 |
| false positive switches | 79 |
| introduced failures | 0 |
| switch precision | 0.466 |
| switch recall | 0.552 |
| oracle switch count | 125 |

### Threshold Sweep

| Threshold | Replay-infeasible Rate | Progress | ADE3 | Switch Rate | Missed Avoidable |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.350 | 21.384 | 5.232 | 0.623 | 21 |
| 0.30 | 0.437 | 24.010 | 4.100 | 0.530 | 47 |
| 0.40 | 0.453 | 24.430 | 3.941 | 0.507 | 52 |
| 0.50 | 0.467 | 24.757 | 3.821 | 0.493 | 56 |
| 0.60 | 0.467 | 24.757 | 3.821 | 0.493 | 56 |
| 0.70 | 0.477 | 25.184 | 3.651 | 0.453 | 59 |
| 0.80 | 0.483 | 25.350 | 3.576 | 0.393 | 61 |

## Train-Selected Operating Points

| Policy | Threshold | Holdout Replay-infeasible Rate | Holdout Progress | Holdout ADE3 |
|---|---:|---:|---:|---:|
| safety_first | 0.70 | 0.477 | 25.184 | 3.651 |
| balanced_80pct_proxy_progress | 0.70 | 0.477 | 25.184 | 3.651 |
| utility_90pct_proxy_progress | 0.70 | 0.477 | 25.184 | 3.651 |

