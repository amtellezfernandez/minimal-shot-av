# nuPlan Recoverable-Regret Bootstrap

- Input replay JSON: `artifacts/corl2027/nuplan_public_replay_study_interaction1000_expanded/replay.json`
- Split unit: `source_db_file`
- Train scenes: `700`
- Holdout scenes: `300`
- Claim boundary: This is a ManeuverToken surrogate for VLA candidate bootstrapping: G1 changes top-1 selection from recovered oracle@K targets, but does not yet regenerate new trajectory candidates.

## Target Summary

- Target count: `1000`
- Teacher changes proxy: `738`
- Mean recoverable regret: `96.346`
- Target tokens: `{"brake_left": 36, "brake_right": 11, "crawl": 48, "creep_right": 9, "evasive_left": 35, "evasive_right": 77, "fast_left": 86, "fast_right": 265, "lane_recover": 18, "maintain": 137, "micro_creep": 8, "nudge_left": 4, "nudge_right": 18, "reverse_creep": 6, "slow_yield": 105, "stop": 12, "wide_left": 54, "wide_left_crawl": 11, "wide_right": 25, "wide_right_crawl": 35}`

## Train

| Iteration/Policy | Score | Regret | Positive Regret | Replay Fail | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|---:|---:|
| g0_proxy_top1 | -105.829 | 80.753 | 487 | 0.649 | 31.210 | 3.053 | 2.391 |
| g1_student_top1 | -40.407 | 15.331 | 510 | 0.287 | 20.890 | 5.200 | 3.974 |
| teacher_promoted_upper_bound | -25.076 | 0.000 | 0 | 0.277 | 26.316 | 4.212 | 3.093 |
| oracle_at_k | -25.076 | 0.000 | 0 | 0.277 | 26.316 | 4.212 | 3.093 |
| replay_oracle_safe_only | -59.467 | 34.390 | 540 | 0.277 | 12.810 | 7.900 | 3.927 |

## Holdout

| Iteration/Policy | Score | Regret | Positive Regret | Replay Fail | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|---:|---:|
| g0_proxy_top1 | -128.067 | 132.729 | 251 | 0.740 | 31.181 | 2.714 | 2.773 |
| g1_student_top1 | -16.485 | 21.146 | 219 | 0.097 | 12.814 | 8.973 | 3.110 |
| teacher_promoted_upper_bound | 4.662 | 0.000 | 0 | 0.097 | 20.526 | 6.112 | 3.238 |
| oracle_at_k | 4.662 | 0.000 | 0 | 0.097 | 20.526 | 6.112 | 3.238 |
| replay_oracle_safe_only | -20.854 | 25.516 | 232 | 0.097 | 11.258 | 9.601 | 3.053 |

