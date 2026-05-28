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
| g0_proxy_top1 | -130.124 | 86.628 | 374 | 0.611 | 27.518 | 2.392 | 2.472 |
| g1_student_top1 | -51.592 | 8.096 | 183 | 0.236 | 18.606 | 5.635 | 2.781 |
| teacher_promoted_upper_bound | -43.495 | 0.000 | 0 | 0.233 | 23.053 | 4.167 | 2.444 |
| oracle_at_k | -43.495 | 0.000 | 0 | 0.233 | 23.053 | 4.167 | 2.444 |
| replay_oracle_safe_only | -60.487 | 16.992 | 466 | 0.233 | 13.043 | 7.658 | 2.571 |

## Holdout

| Iteration/Policy | Score | Regret | Positive Regret | Replay Fail | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|---:|---:|
| g0_proxy_top1 | -142.151 | 63.365 | 135 | 0.663 | 28.998 | 2.658 | 1.882 |
| g1_student_top1 | -139.253 | 60.468 | 177 | 0.630 | 25.599 | 3.676 | 2.281 |
| teacher_promoted_upper_bound | -78.786 | 0.000 | 0 | 0.393 | 26.867 | 3.660 | 1.862 |
| oracle_at_k | -78.786 | 0.000 | 0 | 0.393 | 26.867 | 3.660 | 1.862 |
| replay_oracle_safe_only | -104.161 | 25.375 | 192 | 0.393 | 11.346 | 8.587 | 2.613 |

