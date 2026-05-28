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

## Iteration Summary

### Iteration 1

| Policy | Score | Regret | Replay Fail | Progress | ADE3 |
|---|---:|---:|---:|---:|---:|
| current_proxy | -109.711 | 94.076 | 0.516 | 25.180 | 2.929 |
| student_top1 | -114.388 | 98.753 | 0.458 | 12.583 | 6.228 |
| teacher_upper_bound | -15.635 | 0.000 | 0.116 | 21.421 | 4.012 |

### Iteration 2

| Policy | Score | Regret | Replay Fail | Progress | ADE3 |
|---|---:|---:|---:|---:|---:|
| current_proxy | -114.388 | 98.753 | 0.458 | 12.583 | 6.228 |
| student_top1 | -126.607 | 110.972 | 0.574 | 23.709 | 3.384 |
| teacher_upper_bound | -15.635 | 0.000 | 0.116 | 21.421 | 4.012 |

## Train

| Iteration/Policy | Score | Regret | Positive Regret | Replay Fail | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|---:|---:|
| g0_proxy_top1 | -71.952 | 10.817 | 256 | 0.321 | 19.600 | 5.687 | 2.807 |
| g1_student_top1 | -74.153 | 13.018 | 111 | 0.362 | 24.401 | 4.011 | 2.538 |
| teacher_promoted_upper_bound | -61.135 | 0.000 | 0 | 0.311 | 24.706 | 4.015 | 2.358 |
| oracle_at_k | -61.135 | 0.000 | 0 | 0.311 | 24.706 | 4.015 | 2.358 |
| replay_oracle_safe_only | -82.286 | 21.151 | 576 | 0.311 | 12.179 | 8.327 | 2.591 |

## Holdout

| Iteration/Policy | Score | Regret | Positive Regret | Replay Fail | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|---:|---:|
| g0_proxy_top1 | -114.388 | 98.753 | 141 | 0.458 | 12.583 | 6.228 | 2.405 |
| g1_student_top1 | -126.607 | 110.972 | 105 | 0.574 | 23.709 | 3.384 | 1.993 |
| teacher_promoted_upper_bound | -15.635 | 0.000 | 0 | 0.116 | 21.421 | 4.012 | 2.196 |
| oracle_at_k | -15.635 | 0.000 | 0 | 0.116 | 21.421 | 4.012 | 2.196 |
| replay_oracle_safe_only | -26.178 | 10.543 | 82 | 0.116 | 14.471 | 5.808 | 2.847 |

