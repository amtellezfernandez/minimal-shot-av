# nuPlan Failure-Gated Selector

- Input replay JSON: `artifacts/corl2027/nuplan_public_replay_study_interaction1000_expert/replay.json`
- Safe selector: `fit_on_train_split`
- Fallback mode: `risk_constrained`
- Fallback risk threshold: `0.350`
- Split unit: `source_db_file`
- Train scenes: `700`
- Holdout scenes: `300`
- Gate threshold: `0.500`
- Train positive rate: `0.261`
- Train gate accuracy: `0.934`

## Train

| Selector | Replay-infeasible | Rate | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|
| proxy_selector | 418 | 0.597 | 27.353 | 2.566 | 2.142 |
| safe_selector | 239 | 0.341 | 14.984 | 6.741 | 2.810 |
| oracle_failure_gate | 235 | 0.336 | 23.623 | 3.670 | 2.587 |
| failure_gate | 236 | 0.337 | 22.849 | 3.863 | 2.646 |
| replay_oracle | 232 | 0.331 | 13.239 | 7.340 | 2.703 |

### Gate Diagnostics

| Metric | Value |
|---|---:|
| switch count | 229 |
| switch rate | 0.327 |
| true positive switches | 183 |
| missed avoidable failures | 0 |
| false positive switches | 46 |
| introduced failures | 1 |
| switch precision | 0.799 |
| switch recall | 1.000 |
| oracle switch count | 183 |

### Threshold Sweep

| Threshold | Replay-infeasible Rate | Progress | ADE3 | Switch Rate | Missed Avoidable |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.339 | 21.426 | 4.372 | 0.376 | 0 |
| 0.30 | 0.339 | 22.356 | 4.024 | 0.351 | 0 |
| 0.40 | 0.339 | 22.655 | 3.920 | 0.340 | 0 |
| 0.50 | 0.337 | 22.849 | 3.863 | 0.327 | 0 |
| 0.60 | 0.339 | 22.984 | 3.833 | 0.316 | 1 |
| 0.70 | 0.341 | 23.280 | 3.751 | 0.293 | 4 |
| 0.80 | 0.354 | 23.639 | 3.636 | 0.273 | 13 |

## Holdout

| Selector | Replay-infeasible | Rate | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|
| proxy_selector | 209 | 0.697 | 29.383 | 2.251 | 2.240 |
| safe_selector | 57 | 0.190 | 13.216 | 8.330 | 2.475 |
| oracle_failure_gate | 57 | 0.190 | 19.960 | 5.809 | 2.415 |
| failure_gate | 118 | 0.393 | 23.938 | 4.152 | 2.544 |
| replay_oracle | 49 | 0.163 | 10.889 | 9.329 | 2.344 |

### Gate Diagnostics

| Metric | Value |
|---|---:|
| switch count | 146 |
| switch rate | 0.487 |
| true positive switches | 91 |
| missed avoidable failures | 61 |
| false positive switches | 55 |
| introduced failures | 0 |
| switch precision | 0.623 |
| switch recall | 0.599 |
| oracle switch count | 152 |

### Threshold Sweep

| Threshold | Replay-infeasible Rate | Progress | ADE3 | Switch Rate | Missed Avoidable |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.233 | 19.277 | 6.111 | 0.663 | 13 |
| 0.30 | 0.360 | 23.054 | 4.500 | 0.527 | 51 |
| 0.40 | 0.380 | 23.611 | 4.273 | 0.507 | 57 |
| 0.50 | 0.393 | 23.938 | 4.152 | 0.487 | 61 |
| 0.60 | 0.400 | 24.135 | 4.074 | 0.460 | 63 |
| 0.70 | 0.410 | 24.437 | 3.948 | 0.393 | 66 |
| 0.80 | 0.420 | 25.204 | 3.626 | 0.300 | 69 |

## Train-Selected Operating Points

| Policy | Threshold | Holdout Replay-infeasible Rate | Holdout Progress | Holdout ADE3 |
|---|---:|---:|---:|---:|
| safety_first | 0.50 | 0.393 | 23.938 | 4.152 |
| balanced_80pct_proxy_progress | 0.50 | 0.393 | 23.938 | 4.152 |
| utility_90pct_proxy_progress | 0.50 | 0.393 | 23.938 | 4.152 |

