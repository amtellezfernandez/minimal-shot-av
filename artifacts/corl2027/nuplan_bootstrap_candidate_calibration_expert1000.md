# nuPlan Bootstrapped Candidate-Calibration Audit

- Input replay JSON: `artifacts/corl2027/nuplan_public_replay_study_interaction1000_expert/replay.json`
- Scenes: `1000`
- Teacher: `replay_oracle`

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
| teacher changes proxy token | 795 |
| teacher change rate | 0.795 |
| replay-safe improvements | 346 |
| replay-safe regressions | 0 |
| mean clearance delta | 1.045 |
| mean progress delta | -15.428 |
| mean ADE3 delta | 5.465 |
| mean final-pose shift | 15.588 |

## Selector Table

| Selector | Replay-infeasible | Rate | Progress | ADE3 | Entropy | Selection Gap Fixed |
|---|---:|---:|---:|---:|---:|---:|
| proxy_selector | 627 | 0.627 | 27.962 | 2.472 | 2.375 | 0 |
| lambda80 | 294 | 0.294 | 20.844 | 4.741 | 2.890 | 333 |
| boundary_clearance | 294 | 0.294 | 20.698 | 4.789 | 2.889 | 333 |
| replay_oracle | 281 | 0.281 | 12.534 | 7.937 | 2.694 | 346 |

## Bootstrap Targets

- Target count: `1000`
- Target token histogram: `{"crawl": 155, "evasive_left": 139, "evasive_right": 166, "lane_recover": 25, "maintain": 63, "nudge_left": 10, "nudge_right": 18, "slow_yield": 125, "stop": 299}`

