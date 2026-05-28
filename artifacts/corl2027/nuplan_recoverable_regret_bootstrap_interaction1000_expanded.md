# nuPlan Recoverable-Regret Bootstrap

- Input replay JSON: `artifacts/corl2027/nuplan_public_replay_study_interaction1000_expanded/replay.json`
- Split unit: `source_db_file`
- Train scenes: `700`
- Holdout scenes: `300`
- Claim boundary: This is a ManeuverToken surrogate for VLA candidate bootstrapping: G1 changes top-1 selection from recovered oracle@K targets, but does not yet regenerate new trajectory candidates.

## Target Summary

- Target count: `1000`
- Teacher changes proxy: `734`
- Mean recoverable regret: `103.238`
- Target tokens: `{"brake_left": 36, "brake_right": 11, "crawl": 48, "creep_right": 9, "evasive_left": 35, "evasive_right": 74, "fast_left": 5, "fast_right": 107, "lane_recover": 15, "maintain": 361, "micro_creep": 8, "nudge_left": 4, "nudge_right": 36, "reverse_creep": 6, "slow_yield": 108, "stop": 12, "wide_left": 54, "wide_left_crawl": 11, "wide_right": 25, "wide_right_crawl": 35}`

## Train

| Iteration/Policy | Score | Regret | Positive Regret | Replay Fail | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|---:|---:|
| g0_proxy_top1 | -137.039 | 85.964 | 506 | 0.649 | 31.210 | 3.053 | 2.391 |
| g1_student_top1 | -69.895 | 18.821 | 251 | 0.329 | 21.716 | 4.734 | 3.583 |
| teacher_promoted_upper_bound | -51.075 | 0.000 | 0 | 0.277 | 25.678 | 3.733 | 2.740 |
| oracle_at_k | -51.075 | 0.000 | 0 | 0.277 | 25.678 | 3.733 | 2.740 |
| replay_oracle_safe_only | -72.276 | 21.202 | 540 | 0.277 | 12.810 | 7.900 | 3.927 |

## Holdout

| Iteration/Policy | Score | Regret | Positive Regret | Replay Fail | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|---:|---:|
| g0_proxy_top1 | -159.248 | 143.544 | 228 | 0.740 | 31.181 | 2.714 | 2.773 |
| g1_student_top1 | -109.211 | 93.507 | 225 | 0.467 | 19.263 | 5.904 | 2.712 |
| teacher_promoted_upper_bound | -15.704 | 0.000 | 0 | 0.097 | 20.260 | 5.899 | 3.320 |
| oracle_at_k | -15.704 | 0.000 | 0 | 0.097 | 20.260 | 5.899 | 3.320 |
| replay_oracle_safe_only | -32.112 | 16.408 | 228 | 0.097 | 11.258 | 9.601 | 3.053 |

