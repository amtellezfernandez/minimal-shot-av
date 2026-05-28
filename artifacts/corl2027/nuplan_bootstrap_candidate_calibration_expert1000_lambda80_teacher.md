# nuPlan Bootstrapped Candidate-Calibration Audit

- Input replay JSON: `artifacts/corl2027/nuplan_public_replay_study_interaction1000_expert/replay.json`
- Scenes: `1000`
- Teacher: `lambda80`

## Candidate Surface

| Metric | Value |
|---|---:|
| mean candidate count | 9.000 |
| candidate token entropy | 3.170 |
| mean pairwise final-pose distance | 13.281 |

## Gap Decomposition

| Gap | Count | Rate |
|---|---:|---:|
| oracle@K safe | 719 | 0.719 |
| generation gap | 281 | 0.281 |
| proxy top-1 safe | 373 | 0.373 |
| proxy selection gap | 346 | 0.346 |

## Bootstrap Iteration Target

| Metric | Value |
|---|---:|
| teacher changes proxy token | 554 |
| teacher change rate | 0.554 |
| replay-safe improvements | 333 |
| replay-safe regressions | 0 |
| mean clearance delta | 0.721 |
| mean progress delta | -7.117 |
| mean ADE3 delta | 2.269 |
| mean final-pose shift | 7.248 |

## Selector Table

| Selector | Replay-infeasible | Rate | Progress | ADE3 | Entropy | Selection Gap Fixed |
|---|---:|---:|---:|---:|---:|---:|
| proxy_selector | 627 | 0.627 | 27.962 | 2.472 | 2.375 | 0 |
| lambda80 | 294 | 0.294 | 20.844 | 4.741 | 2.890 | 333 |
| boundary_clearance | 294 | 0.294 | 20.698 | 4.789 | 2.889 | 333 |
| replay_oracle | 281 | 0.281 | 12.534 | 7.937 | 2.694 | 346 |

## Bootstrap Targets

- Target count: `1000`
- Target token histogram: `{"crawl": 115, "evasive_left": 131, "evasive_right": 150, "lane_recover": 86, "maintain": 180, "nudge_left": 12, "nudge_right": 40, "slow_yield": 228, "stop": 58}`

