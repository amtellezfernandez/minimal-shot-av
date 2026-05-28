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

## Iteration Summary

### Iteration 1

| Policy | Score | Regret | Replay Fail | Progress | ADE3 |
|---|---:|---:|---:|---:|---:|
| current_proxy | -159.248 | 143.544 | 0.740 | 31.181 | 2.714 |
| student_top1 | -109.211 | 93.507 | 0.467 | 19.263 | 5.904 |
| teacher_upper_bound | -15.704 | 0.000 | 0.097 | 20.260 | 5.899 |

### Iteration 2

| Policy | Score | Regret | Replay Fail | Progress | ADE3 |
|---|---:|---:|---:|---:|---:|
| current_proxy | -109.211 | 93.507 | 0.467 | 19.263 | 5.904 |
| student_top1 | -106.990 | 91.286 | 0.483 | 23.235 | 4.696 |
| teacher_upper_bound | -15.704 | 0.000 | 0.097 | 20.260 | 5.899 |

### Iteration 3

| Policy | Score | Regret | Replay Fail | Progress | ADE3 |
|---|---:|---:|---:|---:|---:|
| current_proxy | -106.990 | 91.286 | 0.483 | 23.235 | 4.696 |
| student_top1 | -101.411 | 85.707 | 0.460 | 23.062 | 4.737 |
| teacher_upper_bound | -15.704 | 0.000 | 0.097 | 20.260 | 5.899 |

## Train

| Iteration/Policy | Score | Regret | Positive Regret | Replay Fail | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|---:|---:|
| g0_proxy_top1 | -63.495 | 12.420 | 132 | 0.317 | 24.011 | 4.110 | 3.109 |
| g1_student_top1 | -58.976 | 7.901 | 106 | 0.299 | 24.033 | 4.183 | 3.075 |
| teacher_promoted_upper_bound | -51.075 | 0.000 | 0 | 0.277 | 25.678 | 3.733 | 2.740 |
| oracle_at_k | -51.075 | 0.000 | 0 | 0.277 | 25.678 | 3.733 | 2.740 |
| replay_oracle_safe_only | -72.276 | 21.202 | 540 | 0.277 | 12.810 | 7.900 | 3.927 |

## Holdout

| Iteration/Policy | Score | Regret | Positive Regret | Replay Fail | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|---:|---:|
| g0_proxy_top1 | -106.990 | 91.286 | 219 | 0.483 | 23.235 | 4.696 | 2.592 |
| g1_student_top1 | -101.411 | 85.707 | 207 | 0.460 | 23.062 | 4.737 | 2.471 |
| teacher_promoted_upper_bound | -15.704 | 0.000 | 0 | 0.097 | 20.260 | 5.899 | 3.320 |
| oracle_at_k | -15.704 | 0.000 | 0 | 0.097 | 20.260 | 5.899 | 3.320 |
| replay_oracle_safe_only | -32.112 | 16.408 | 228 | 0.097 | 11.258 | 9.601 | 3.053 |

