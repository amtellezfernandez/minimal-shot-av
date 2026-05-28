# Linear vs MLP Replay-Value Fallback Summary

Five DB-level splits on `nuplan_public_replay_study_interaction1000_expert/replay.json`.

## Holdout Mean ± Std

| Variant | Selector | Replay fail rate | Progress | ADE3 | Entropy |
|---|---|---:|---:|---:|---:|
| linear_value_p100 | safe_selector | 0.306 ± 0.217 | 20.451 ± 5.613 | 5.238 ± 1.193 | 2.224 ± 0.140 |
| linear_value_p100 | failure_gate | 0.353 ± 0.217 | 21.750 ± 6.264 | 4.724 ± 1.447 | 2.244 ± 0.184 |
| linear_value_p100 | oracle_failure_gate | 0.300 ± 0.219 | 21.675 ± 6.363 | 4.701 ± 1.420 | 2.330 ± 0.128 |
| linear_value_p100 | progress_oracle_failure_gate | 0.259 ± 0.204 | 21.895 ± 5.502 | 4.488 ± 0.995 | 2.305 ± 0.162 |
| mlp_value_p100 | safe_selector | 0.272 ± 0.215 | 19.201 ± 5.246 | 5.240 ± 0.792 | 2.459 ± 0.350 |
| mlp_value_p100 | failure_gate | 0.322 ± 0.205 | 21.336 ± 5.082 | 4.733 ± 0.848 | 2.227 ± 0.117 |
| mlp_value_p100 | oracle_failure_gate | 0.272 ± 0.215 | 21.657 ± 5.712 | 4.568 ± 1.029 | 2.226 ± 0.120 |
| mlp_value_p100 | progress_oracle_failure_gate | 0.259 ± 0.204 | 21.895 ± 5.502 | 4.488 ± 0.995 | 2.305 ± 0.162 |

## Per-Seed Failure-Gate Holdout Rates

| Seed | Linear gate | MLP gate | Linear progress | MLP progress |
|---:|---:|---:|---:|---:|
| 0 | 0.190 | 0.220 | 18.893 | 20.805 |
| 1 | 0.323 | 0.219 | 22.699 | 19.944 |
| 2 | 0.557 | 0.490 | 26.265 | 24.768 |
| 3 | 0.637 | 0.623 | 29.483 | 28.133 |
| 4 | 0.058 | 0.058 | 11.409 | 13.031 |

## Interpretation

- MLP fallback is not uniformly better than the linear value fallback.
- It exposes a different point on the safety-utility frontier: lower average replay-failure rate than linear gating, with similar ADE and slightly lower average progress.
- The method claim should remain failure-aware gating plus fallback-selector frontier, not “MLP solves it.”
