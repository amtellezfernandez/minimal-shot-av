# nuPlan Failure-Gate Seed Sweep: Train-Split Safe Fallback

| Seed | Proxy Rate | Gate Rate | Safe Rate | Gate Progress | Safe Progress | Switch Rate |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.697 | 0.280 | 0.163 | 22.917 | 18.342 | 0.587 |
| 1 | 0.516 | 0.368 | 0.245 | 23.086 | 17.253 | 0.290 |
| 2 | 0.663 | 0.583 | 0.513 | 27.177 | 19.109 | 0.157 |
| 3 | 0.823 | 0.657 | 0.643 | 28.848 | 25.487 | 0.357 |
| 4 | 0.561 | 0.342 | 0.303 | 19.410 | 18.140 | 0.413 |

## Mean Std

| Metric | Mean | Std |
|---|---:|---:|
| proxy_rate | 0.652 | 0.108 |
| gate_rate | 0.446 | 0.147 |
| safe_rate | 0.374 | 0.178 |
| oracle_rate | 0.259 | 0.204 |
| proxy_progress | 28.043 | 3.407 |
| gate_progress | 24.288 | 3.354 |
| safe_progress | 19.666 | 2.970 |
| proxy_ade | 2.443 | 0.300 |
| gate_ade | 3.670 | 0.450 |
| safe_ade | 4.960 | 0.901 |
| gate_switch_rate | 0.361 | 0.142 |

## Train-Selected Threshold Policies

| Policy | Threshold | Holdout Rate | Holdout Progress | Holdout ADE3 | Switch Rate |
|---|---:|---:|---:|---:|---:|
| safety_first | 0.440±0.049 | 0.437±0.147 | 24.167±3.415 | 3.719±0.416 | 0.373±0.138 |
| balanced_80pct_proxy_progress | 0.580±0.147 | 0.449±0.157 | 24.532±3.866 | 3.600±0.464 | 0.342±0.156 |
| utility_90pct_proxy_progress | 0.500±0.155 | 0.495±0.130 | 25.067±3.556 | 3.410±0.272 | 0.313±0.089 |
