# nuPlan Failure-Gate Seed Sweep: Fallback-Aware Train-Split Safe Fallback

| Seed | Proxy Rate | Gate Rate | Safe Rate | Gate Progress | Safe Progress | Switch Rate |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.697 | 0.240 | 0.163 | 21.883 | 18.342 | 0.527 |
| 1 | 0.516 | 0.335 | 0.245 | 22.670 | 17.253 | 0.303 |
| 2 | 0.663 | 0.560 | 0.513 | 25.535 | 19.109 | 0.183 |
| 3 | 0.823 | 0.653 | 0.643 | 29.675 | 25.487 | 0.240 |
| 4 | 0.561 | 0.329 | 0.303 | 19.129 | 18.140 | 0.342 |

## Mean Std

| Metric | Mean | Std |
|---|---:|---:|
| proxy_rate | 0.652 | 0.108 |
| gate_rate | 0.424 | 0.156 |
| safe_rate | 0.374 | 0.178 |
| oracle_rate | 0.259 | 0.204 |
| proxy_progress | 28.043 | 3.407 |
| gate_progress | 23.778 | 3.586 |
| safe_progress | 19.666 | 2.970 |
| proxy_ade | 2.443 | 0.300 |
| gate_ade | 3.880 | 0.611 |
| safe_ade | 4.960 | 0.901 |
| gate_switch_rate | 0.319 | 0.117 |

## Train-Selected Threshold Policies

| Policy | Threshold | Holdout Rate | Holdout Progress | Holdout ADE3 | Switch Rate |
|---|---:|---:|---:|---:|---:|
| safety_first | 0.440±0.049 | 0.418±0.162 | 23.687±3.634 | 3.915±0.666 | 0.381±0.169 |
| balanced_80pct_proxy_progress | 0.520±0.147 | 0.424±0.165 | 24.039±3.857 | 3.776±0.719 | 0.361±0.190 |
| utility_90pct_proxy_progress | 0.440±0.049 | 0.418±0.162 | 23.687±3.634 | 3.915±0.666 | 0.381±0.169 |
