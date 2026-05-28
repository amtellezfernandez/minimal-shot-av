# nuPlan Recoverable-Regret Bootstrap

- Input replay JSON: `artifacts/corl2027/nuplan_public_replay_study_interaction1000_expert/replay.json`
- Split unit: `source_db_file`
- Train scenes: `845`
- Holdout scenes: `155`
- Claim boundary: This is a ManeuverToken surrogate for VLA candidate bootstrapping: G1 changes top-1 selection from recovered oracle@K targets, but does not yet regenerate new trajectory candidates.

## Target Summary

- Target count: `1000`
- Teacher changes proxy: `509`
- Mean recoverable regret: `79.649`
- Target tokens: `{"crawl": 92, "evasive_left": 40, "evasive_right": 99, "lane_recover": 18, "maintain": 457, "nudge_left": 4, "nudge_right": 69, "slow_yield": 175, "stop": 46}`

## Train

| Iteration/Policy | Score | Regret | Positive Regret | Replay Fail | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|---:|---:|
| g0_proxy_top1 | -136.038 | 73.324 | 426 | 0.639 | 28.784 | 2.529 | 2.224 |
| g1_student_top1 | -84.601 | 21.886 | 312 | 0.381 | 20.832 | 5.083 | 2.806 |
| teacher_promoted_upper_bound | -62.714 | 0.000 | 0 | 0.327 | 26.240 | 3.649 | 2.237 |
| oracle_at_k | -62.714 | 0.000 | 0 | 0.327 | 26.240 | 3.649 | 2.237 |
| replay_oracle_safe_only | -84.505 | 21.791 | 566 | 0.327 | 13.159 | 8.004 | 2.710 |

## Holdout

| Iteration/Policy | Score | Regret | Positive Regret | Replay Fail | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|---:|---:|
| g0_proxy_top1 | -121.159 | 114.134 | 83 | 0.561 | 23.482 | 2.159 | 2.253 |
| g1_student_top1 | -34.067 | 27.042 | 116 | 0.123 | 10.565 | 6.994 | 2.350 |
| teacher_promoted_upper_bound | -7.025 | 0.000 | 0 | 0.032 | 13.058 | 6.009 | 1.987 |
| oracle_at_k | -7.025 | 0.000 | 0 | 0.032 | 13.058 | 6.009 | 1.987 |
| replay_oracle_safe_only | -14.080 | 7.055 | 92 | 0.032 | 9.125 | 7.570 | 2.218 |

