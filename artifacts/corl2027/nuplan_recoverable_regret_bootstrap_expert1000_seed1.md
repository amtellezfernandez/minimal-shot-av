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
| g0_proxy_top1 | -138.138 | 77.003 | 433 | 0.647 | 28.472 | 2.388 | 2.411 |
| g1_student_top1 | -67.245 | 6.110 | 107 | 0.327 | 23.284 | 4.436 | 2.603 |
| teacher_promoted_upper_bound | -61.135 | 0.000 | 0 | 0.311 | 24.706 | 4.015 | 2.358 |
| oracle_at_k | -61.135 | 0.000 | 0 | 0.311 | 24.706 | 4.015 | 2.358 |
| replay_oracle_safe_only | -82.286 | 21.151 | 576 | 0.311 | 12.179 | 8.327 | 2.591 |

## Holdout

| Iteration/Policy | Score | Regret | Positive Regret | Replay Fail | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|---:|---:|
| g0_proxy_top1 | -109.711 | 94.076 | 76 | 0.516 | 25.180 | 2.929 | 1.595 |
| g1_student_top1 | -113.261 | 97.626 | 92 | 0.510 | 21.942 | 3.892 | 1.970 |
| teacher_promoted_upper_bound | -15.635 | 0.000 | 0 | 0.116 | 21.421 | 4.012 | 2.196 |
| oracle_at_k | -15.635 | 0.000 | 0 | 0.116 | 21.421 | 4.012 | 2.196 |
| replay_oracle_safe_only | -26.178 | 10.543 | 82 | 0.116 | 14.471 | 5.808 | 2.847 |

