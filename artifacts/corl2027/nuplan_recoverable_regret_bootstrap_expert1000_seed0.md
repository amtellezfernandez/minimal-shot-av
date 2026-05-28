# nuPlan Recoverable-Regret Bootstrap

- Input replay JSON: `artifacts/corl2027/nuplan_public_replay_study_interaction1000_expert/replay.json`
- Split unit: `source_db_file`
- Train scenes: `700`
- Holdout scenes: `300`
- Claim boundary: This is a ManeuverToken surrogate for VLA candidate bootstrapping: G1 changes top-1 selection from recovered oracle@K targets, but does not yet regenerate new trajectory candidates.

## Target Summary

- Target count: `1000`
- Teacher changes proxy: `509`
- Mean recoverable regret: `79.649`
- Target tokens: `{"crawl": 92, "evasive_left": 40, "evasive_right": 99, "lane_recover": 18, "maintain": 457, "nudge_left": 4, "nudge_right": 69, "slow_yield": 175, "stop": 46}`

## Train

| Iteration/Policy | Score | Regret | Positive Regret | Replay Fail | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|---:|---:|
| g0_proxy_top1 | -127.066 | 62.791 | 347 | 0.597 | 27.353 | 2.566 | 2.142 |
| g1_student_top1 | -73.987 | 9.712 | 150 | 0.354 | 22.862 | 4.139 | 2.560 |
| teacher_promoted_upper_bound | -64.275 | 0.000 | 0 | 0.331 | 25.510 | 3.464 | 2.064 |
| oracle_at_k | -64.275 | 0.000 | 0 | 0.331 | 25.510 | 3.464 | 2.064 |
| replay_oracle_safe_only | -84.298 | 20.023 | 469 | 0.331 | 13.239 | 7.340 | 2.703 |

## Holdout

| Iteration/Policy | Score | Regret | Positive Regret | Replay Fail | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|---:|---:|
| g0_proxy_top1 | -149.285 | 118.986 | 162 | 0.697 | 29.383 | 2.251 | 2.240 |
| g1_student_top1 | -99.081 | 68.782 | 135 | 0.453 | 23.219 | 4.483 | 1.925 |
| teacher_promoted_upper_bound | -30.299 | 0.000 | 0 | 0.163 | 21.134 | 5.300 | 2.329 |
| oracle_at_k | -30.299 | 0.000 | 0 | 0.163 | 21.134 | 5.300 | 2.329 |
| replay_oracle_safe_only | -48.601 | 18.302 | 189 | 0.163 | 10.889 | 9.329 | 2.344 |

