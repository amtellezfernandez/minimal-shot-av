# MLP Failure-Gate Bridge Seed Summary

Five DB-level split bridge runs, 40 base scenes per seed. Each base scene is evaluated with proxy, embedded failure gate, and replay oracle.

Absolute nuPlan box-clearance fails for all selectors in these bridge runs. The table below reports the aligned replay-equivalent clearance metric only, so this is directional transfer evidence, not an absolute closed-loop safety claim.

| Seed | Proxy fails | Gate fails | Oracle fails | Recoverable | Closed by gate | Gate progress | Proxy progress | Gate ADE3 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 20/40 | 0/40 | 0/40 | 20 | 20 | 9.988 | 23.295 | 7.680 |
| 1 | 20/40 | 6/40 | 3/40 | 17 | 14 | 20.969 | 26.722 | 3.905 |
| 2 | 20/40 | 3/40 | 3/40 | 17 | 17 | 19.899 | 26.336 | 4.140 |
| 3 | 20/40 | 20/40 | 20/40 | 0 | 0 | 23.183 | 23.183 | 2.365 |
| 4 | 20/40 | 0/40 | 0/40 | 20 | 20 | 9.557 | 21.019 | 6.917 |

## Mean ± Std

| Metric | Mean ± Std |
|---|---:|
| Proxy aligned failures | 20.000 ± 0.000 |
| Failure-gate aligned failures | 5.800 ± 7.440 |
| Replay-oracle aligned failures | 5.200 ± 7.521 |
| Recoverable proxy failures | 14.800 ± 7.521 |
| Recoverable failures closed by gate | 14.200 ± 7.440 |
| Recoverable gap closed rate | 0.956 ± 0.076 |
| Proxy progress | 24.111 ± 2.138 |
| Failure-gate progress | 16.719 ± 5.772 |
| Failure-gate progress retained vs proxy | 0.685 ± 0.216 |
| Proxy ADE3 | 2.250 ± 0.213 |
| Failure-gate ADE3 | 5.001 ± 1.987 |

## Interpretation

- The failure gate directionally transfers on aligned replay-equivalent clearance: mean aligned failures drop from proxy `20.0/40` to gate `5.8/40`.
- Seed 3 is an unrecoverable bridge split under the replay oracle (`20/40` oracle failures), so it should not be interpreted as a gate failure.
- On recoverable bridge failures, the gate closes the full recoverable gap in 3/4 recoverable splits and closes `0.956 ± 0.076` of the recoverable gap over recoverable splits.
- The gain is not free: gate progress is `16.719 ± 5.793 m` versus proxy `24.911 ± 2.104 m`, retaining `0.668 ± 0.205` of proxy progress and increasing ADE3.
- Absolute nuPlan box-clearance remains uncalibrated because every selector has `40/40` box failures in these bridge runs.
