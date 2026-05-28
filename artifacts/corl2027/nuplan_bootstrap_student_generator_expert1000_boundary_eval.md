# nuPlan Bootstrap Student Generator

- Input replay JSON: `artifacts/corl2027/nuplan_public_replay_study_interaction1000_expert/replay.json`
- Teacher targets: `artifacts/corl2027/nuplan_bootstrap_boundary_teacher_targets_expert1000.jsonl`
- Split unit: `source_db_file`
- Train scenes: `700`
- Holdout scenes: `300`
- Training scene accuracy: `0.871`

## Train

| Selector | Replay-infeasible | Rate | Selection Gap | Teacher Agree | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|---:|---:|
| g0_proxy_top1 | 470 | 0.671 | 212 | 0.503 | 29.360 | 2.423 | 2.251 |
| g1_student | 280 | 0.400 | 22 | 0.871 | 23.505 | 4.316 | 2.630 |
| teacher_promoted_upper_bound | 267 | 0.381 | 9 | 1.000 | 23.070 | 4.452 | 2.691 |
| replay_oracle | 258 | 0.369 | 0 | 0.556 | 12.775 | 8.400 | 2.592 |

### Token Histograms

- `g0_proxy_top1`: `{"evasive_left": 93, "evasive_right": 112, "lane_recover": 113, "maintain": 301, "nudge_left": 11, "nudge_right": 12, "slow_yield": 58}`
- `g1_student`: `{"crawl": 22, "evasive_left": 88, "evasive_right": 165, "lane_recover": 47, "maintain": 164, "nudge_right": 16, "slow_yield": 155, "stop": 43}`
- `teacher_promoted_upper_bound`: `{"crawl": 22, "evasive_left": 89, "evasive_right": 147, "lane_recover": 62, "maintain": 145, "nudge_left": 2, "nudge_right": 17, "slow_yield": 172, "stop": 44}`
- `replay_oracle`: `{"crawl": 106, "evasive_left": 96, "evasive_right": 159, "lane_recover": 13, "maintain": 32, "nudge_left": 6, "nudge_right": 5, "slow_yield": 83, "stop": 200}`

## Holdout

| Selector | Replay-infeasible | Rate | Selection Gap | Teacher Agree | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|---:|---:|
| g0_proxy_top1 | 157 | 0.523 | 134 | 0.293 | 24.700 | 2.585 | 2.004 |
| g1_student | 94 | 0.313 | 71 | 0.617 | 20.268 | 3.622 | 2.733 |
| teacher_promoted_upper_bound | 27 | 0.090 | 4 | 1.000 | 15.163 | 5.575 | 2.734 |
| replay_oracle | 23 | 0.077 | 0 | 0.590 | 11.972 | 6.855 | 2.684 |

### Token Histograms

- `g0_proxy_top1`: `{"evasive_left": 6, "lane_recover": 16, "maintain": 134, "nudge_left": 12, "nudge_right": 68, "slow_yield": 64}`
- `g1_student`: `{"crawl": 32, "evasive_left": 30, "evasive_right": 3, "lane_recover": 23, "maintain": 53, "nudge_left": 8, "nudge_right": 76, "slow_yield": 67, "stop": 8}`
- `teacher_promoted_upper_bound`: `{"crawl": 94, "evasive_left": 44, "evasive_right": 3, "lane_recover": 21, "maintain": 32, "nudge_left": 9, "nudge_right": 24, "slow_yield": 57, "stop": 16}`
- `replay_oracle`: `{"crawl": 49, "evasive_left": 43, "evasive_right": 7, "lane_recover": 12, "maintain": 31, "nudge_left": 4, "nudge_right": 13, "slow_yield": 42, "stop": 99}`

