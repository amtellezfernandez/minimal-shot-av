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
| g0_proxy_top1 | -115.148 | 89.988 | 350 | 0.543 | 25.728 | 2.581 | 2.376 |
| g1_student_top1 | -39.390 | 14.230 | 240 | 0.171 | 15.315 | 5.924 | 2.741 |
| teacher_promoted_upper_bound | -25.160 | 0.000 | 0 | 0.149 | 20.729 | 4.373 | 2.483 |
| oracle_at_k | -25.160 | 0.000 | 0 | 0.149 | 20.729 | 4.373 | 2.483 |
| replay_oracle_safe_only | -41.326 | 16.165 | 471 | 0.149 | 11.221 | 7.702 | 2.677 |

## Holdout

| Iteration/Policy | Score | Regret | Positive Regret | Replay Fail | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|---:|---:|
| g0_proxy_top1 | -177.094 | 55.526 | 159 | 0.823 | 33.173 | 2.217 | 2.082 |
| g1_student_top1 | -169.139 | 47.572 | 122 | 0.747 | 27.564 | 5.018 | 2.108 |
| teacher_promoted_upper_bound | -121.568 | 0.000 | 0 | 0.590 | 32.288 | 3.178 | 1.556 |
| oracle_at_k | -121.568 | 0.000 | 0 | 0.590 | 32.288 | 3.178 | 1.556 |
| replay_oracle_safe_only | -148.870 | 27.302 | 187 | 0.590 | 15.598 | 8.484 | 2.197 |

