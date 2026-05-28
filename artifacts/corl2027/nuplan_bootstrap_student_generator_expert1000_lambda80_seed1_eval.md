# nuPlan Bootstrap Student Generator

- Input replay JSON: `artifacts/corl2027/nuplan_public_replay_study_interaction1000_expert/replay.json`
- Teacher targets: `artifacts/corl2027/nuplan_bootstrap_lambda80_teacher_targets_expert1000.jsonl`
- Split unit: `source_db_file`
- Train scenes: `845`
- Holdout scenes: `155`
- Training scene accuracy: `0.769`

## Train

| Selector | Replay-infeasible | Rate | Selection Gap | Teacher Agree | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|---:|---:|
| g0_proxy_top1 | 547 | 0.647 | 284 | 0.477 | 28.472 | 2.388 | 2.411 |
| g1_student | 328 | 0.388 | 65 | 0.769 | 21.541 | 4.652 | 2.811 |
| teacher_promoted_upper_bound | 272 | 0.322 | 9 | 1.000 | 21.312 | 4.821 | 2.769 |
| replay_oracle | 263 | 0.311 | 0 | 0.549 | 12.179 | 8.327 | 2.591 |

### Token Histograms

- `g0_proxy_top1`: `{"evasive_left": 99, "evasive_right": 112, "lane_recover": 125, "maintain": 337, "nudge_left": 12, "nudge_right": 64, "slow_yield": 96}`
- `g1_student`: `{"crawl": 54, "evasive_left": 156, "evasive_right": 164, "lane_recover": 16, "maintain": 139, "nudge_left": 17, "nudge_right": 52, "slow_yield": 188, "stop": 59}`
- `teacher_promoted_upper_bound`: `{"crawl": 94, "evasive_left": 117, "evasive_right": 147, "lane_recover": 65, "maintain": 160, "nudge_left": 2, "nudge_right": 16, "slow_yield": 200, "stop": 44}`
- `replay_oracle`: `{"crawl": 132, "evasive_left": 125, "evasive_right": 159, "lane_recover": 14, "maintain": 38, "nudge_left": 9, "nudge_right": 5, "slow_yield": 106, "stop": 257}`

## Holdout

| Selector | Replay-infeasible | Rate | Selection Gap | Teacher Agree | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|---:|---:|
| g0_proxy_top1 | 80 | 0.516 | 62 | 0.277 | 25.180 | 2.929 | 1.595 |
| g1_student | 48 | 0.310 | 30 | 0.484 | 21.110 | 3.769 | 2.813 |
| teacher_promoted_upper_bound | 22 | 0.142 | 4 | 1.000 | 18.294 | 4.305 | 3.017 |
| replay_oracle | 18 | 0.116 | 0 | 0.639 | 14.471 | 5.808 | 2.847 |

### Token Histograms

- `g0_proxy_top1`: `{"lane_recover": 4, "maintain": 98, "nudge_left": 11, "nudge_right": 16, "slow_yield": 26}`
- `g1_student`: `{"crawl": 5, "evasive_left": 15, "evasive_right": 14, "lane_recover": 8, "maintain": 45, "nudge_left": 17, "nudge_right": 12, "slow_yield": 34, "stop": 5}`
- `teacher_promoted_upper_bound`: `{"crawl": 21, "evasive_left": 14, "evasive_right": 3, "lane_recover": 21, "maintain": 20, "nudge_left": 10, "nudge_right": 24, "slow_yield": 28, "stop": 14}`
- `replay_oracle`: `{"crawl": 23, "evasive_left": 14, "evasive_right": 7, "lane_recover": 11, "maintain": 25, "nudge_left": 1, "nudge_right": 13, "slow_yield": 19, "stop": 42}`

