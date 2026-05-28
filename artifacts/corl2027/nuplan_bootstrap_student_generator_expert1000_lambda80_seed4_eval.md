# nuPlan Bootstrap Student Generator

- Input replay JSON: `artifacts/corl2027/nuplan_public_replay_study_interaction1000_expert/replay.json`
- Teacher targets: `artifacts/corl2027/nuplan_bootstrap_lambda80_teacher_targets_expert1000.jsonl`
- Split unit: `source_db_file`
- Train scenes: `845`
- Holdout scenes: `155`
- Training scene accuracy: `0.636`

## Train

| Selector | Replay-infeasible | Rate | Selection Gap | Teacher Agree | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|---:|---:|
| g0_proxy_top1 | 540 | 0.639 | 264 | 0.470 | 28.784 | 2.529 | 2.224 |
| g1_student | 351 | 0.415 | 75 | 0.636 | 23.851 | 4.101 | 2.610 |
| teacher_promoted_upper_bound | 289 | 0.342 | 13 | 1.000 | 22.421 | 4.448 | 2.865 |
| replay_oracle | 276 | 0.327 | 0 | 0.567 | 13.159 | 8.004 | 2.710 |

### Token Histograms

- `g0_proxy_top1`: `{"evasive_left": 93, "evasive_right": 112, "lane_recover": 117, "maintain": 399, "nudge_left": 12, "nudge_right": 28, "slow_yield": 84}`
- `g1_student`: `{"crawl": 23, "evasive_left": 31, "evasive_right": 133, "lane_recover": 87, "maintain": 285, "nudge_left": 7, "nudge_right": 38, "slow_yield": 180, "stop": 61}`
- `teacher_promoted_upper_bound`: `{"crawl": 42, "evasive_left": 103, "evasive_right": 150, "lane_recover": 84, "maintain": 166, "nudge_left": 12, "nudge_right": 40, "slow_yield": 190, "stop": 58}`
- `replay_oracle`: `{"crawl": 129, "evasive_left": 110, "evasive_right": 166, "lane_recover": 24, "maintain": 57, "nudge_left": 7, "nudge_right": 18, "slow_yield": 94, "stop": 240}`

## Holdout

| Selector | Replay-infeasible | Rate | Selection Gap | Teacher Agree | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|---:|---:|
| g0_proxy_top1 | 87 | 0.561 | 82 | 0.316 | 23.482 | 2.159 | 2.253 |
| g1_student | 50 | 0.323 | 45 | 0.619 | 18.780 | 3.453 | 2.366 |
| teacher_promoted_upper_bound | 5 | 0.032 | 0 | 1.000 | 12.250 | 6.338 | 1.849 |
| replay_oracle | 5 | 0.032 | 0 | 0.542 | 9.125 | 7.570 | 2.218 |

### Token Histograms

- `g0_proxy_top1`: `{"evasive_left": 6, "lane_recover": 12, "maintain": 36, "nudge_left": 11, "nudge_right": 52, "slow_yield": 38}`
- `g1_student`: `{"crawl": 26, "evasive_left": 21, "lane_recover": 5, "maintain": 18, "nudge_right": 41, "slow_yield": 44}`
- `teacher_promoted_upper_bound`: `{"crawl": 73, "evasive_left": 28, "lane_recover": 2, "maintain": 14, "slow_yield": 38}`
- `replay_oracle`: `{"crawl": 26, "evasive_left": 29, "lane_recover": 1, "maintain": 6, "nudge_left": 3, "slow_yield": 31, "stop": 59}`

