# nuPlan Bootstrap Student Generator

- Input replay JSON: `artifacts/corl2027/nuplan_public_replay_study_interaction1000_expert/replay.json`
- Teacher targets: `artifacts/corl2027/nuplan_bootstrap_lambda80_teacher_targets_expert1000.jsonl`
- Split unit: `source_db_file`
- Train scenes: `700`
- Holdout scenes: `300`
- Training scene accuracy: `0.869`

## Train

| Selector | Replay-infeasible | Rate | Selection Gap | Teacher Agree | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|---:|---:|
| g0_proxy_top1 | 470 | 0.671 | 212 | 0.506 | 29.360 | 2.423 | 2.251 |
| g1_student | 280 | 0.400 | 22 | 0.869 | 23.518 | 4.310 | 2.633 |
| teacher_promoted_upper_bound | 267 | 0.381 | 9 | 1.000 | 23.133 | 4.431 | 2.684 |
| replay_oracle | 258 | 0.369 | 0 | 0.554 | 12.775 | 8.400 | 2.592 |

### Token Histograms

- `g0_proxy_top1`: `{"evasive_left": 93, "evasive_right": 112, "lane_recover": 113, "maintain": 301, "nudge_left": 11, "nudge_right": 12, "slow_yield": 58}`
- `g1_student`: `{"crawl": 22, "evasive_left": 88, "evasive_right": 164, "lane_recover": 48, "maintain": 164, "nudge_right": 16, "slow_yield": 155, "stop": 43}`
- `teacher_promoted_upper_bound`: `{"crawl": 21, "evasive_left": 89, "evasive_right": 147, "lane_recover": 63, "maintain": 146, "nudge_left": 2, "nudge_right": 16, "slow_yield": 172, "stop": 44}`
- `replay_oracle`: `{"crawl": 106, "evasive_left": 96, "evasive_right": 159, "lane_recover": 13, "maintain": 32, "nudge_left": 6, "nudge_right": 5, "slow_yield": 83, "stop": 200}`

## Holdout

| Selector | Replay-infeasible | Rate | Selection Gap | Teacher Agree | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|---:|---:|
| g0_proxy_top1 | 157 | 0.523 | 134 | 0.307 | 24.700 | 2.585 | 2.004 |
| g1_student | 94 | 0.313 | 71 | 0.610 | 20.293 | 3.631 | 2.740 |
| teacher_promoted_upper_bound | 27 | 0.090 | 4 | 1.000 | 15.505 | 5.466 | 2.742 |
| replay_oracle | 23 | 0.077 | 0 | 0.583 | 11.972 | 6.855 | 2.684 |

### Token Histograms

- `g0_proxy_top1`: `{"evasive_left": 6, "lane_recover": 16, "maintain": 134, "nudge_left": 12, "nudge_right": 68, "slow_yield": 64}`
- `g1_student`: `{"crawl": 32, "evasive_left": 30, "evasive_right": 3, "lane_recover": 24, "maintain": 54, "nudge_left": 8, "nudge_right": 74, "slow_yield": 67, "stop": 8}`
- `teacher_promoted_upper_bound`: `{"crawl": 94, "evasive_left": 42, "evasive_right": 3, "lane_recover": 23, "maintain": 34, "nudge_left": 10, "nudge_right": 24, "slow_yield": 56, "stop": 14}`
- `replay_oracle`: `{"crawl": 49, "evasive_left": 43, "evasive_right": 7, "lane_recover": 12, "maintain": 31, "nudge_left": 4, "nudge_right": 13, "slow_yield": 42, "stop": 99}`

