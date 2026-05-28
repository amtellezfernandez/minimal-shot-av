# nuPlan Bootstrap Student Generator

- Input replay JSON: `artifacts/corl2027/nuplan_public_replay_study_interaction1000_expert/replay.json`
- Teacher targets: `artifacts/corl2027/nuplan_bootstrap_lambda80_teacher_targets_expert1000.jsonl`
- Split unit: `source_db_file`
- Train scenes: `700`
- Holdout scenes: `300`
- Training scene accuracy: `0.630`

## Train

| Selector | Replay-infeasible | Rate | Selection Gap | Teacher Agree | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|---:|---:|
| g0_proxy_top1 | 380 | 0.543 | 276 | 0.363 | 25.728 | 2.581 | 2.376 |
| g1_student | 184 | 0.263 | 80 | 0.630 | 18.144 | 4.899 | 2.686 |
| teacher_promoted_upper_bound | 108 | 0.154 | 4 | 1.000 | 18.003 | 4.994 | 2.786 |
| replay_oracle | 104 | 0.149 | 0 | 0.530 | 11.221 | 7.702 | 2.677 |

### Token Histograms

- `g0_proxy_top1`: `{"evasive_left": 92, "evasive_right": 45, "lane_recover": 72, "maintain": 299, "nudge_left": 17, "nudge_right": 73, "slow_yield": 102}`
- `g1_student`: `{"crawl": 86, "evasive_left": 219, "evasive_right": 22, "lane_recover": 14, "maintain": 71, "nudge_left": 66, "nudge_right": 19, "slow_yield": 164, "stop": 39}`
- `teacher_promoted_upper_bound`: `{"crawl": 107, "evasive_left": 124, "evasive_right": 31, "lane_recover": 67, "maintain": 95, "nudge_left": 10, "nudge_right": 31, "slow_yield": 201, "stop": 34}`
- `replay_oracle`: `{"crawl": 113, "evasive_left": 132, "evasive_right": 42, "lane_recover": 22, "maintain": 55, "nudge_left": 8, "nudge_right": 13, "slow_yield": 103, "stop": 212}`

## Holdout

| Selector | Replay-infeasible | Rate | Selection Gap | Teacher Agree | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|---:|---:|
| g0_proxy_top1 | 247 | 0.823 | 70 | 0.640 | 33.173 | 2.217 | 2.082 |
| g1_student | 237 | 0.790 | 60 | 0.593 | 27.324 | 4.276 | 2.815 |
| teacher_promoted_upper_bound | 186 | 0.620 | 9 | 1.000 | 27.475 | 4.151 | 2.367 |
| replay_oracle | 177 | 0.590 | 0 | 0.640 | 15.598 | 8.484 | 2.197 |

### Token Histograms

- `g0_proxy_top1`: `{"evasive_left": 7, "evasive_right": 67, "lane_recover": 57, "maintain": 136, "nudge_left": 6, "nudge_right": 7, "slow_yield": 20}`
- `g1_student`: `{"crawl": 13, "evasive_left": 35, "evasive_right": 61, "lane_recover": 37, "maintain": 84, "nudge_left": 14, "nudge_right": 5, "slow_yield": 20, "stop": 31}`
- `teacher_promoted_upper_bound`: `{"crawl": 8, "evasive_left": 7, "evasive_right": 119, "lane_recover": 19, "maintain": 85, "nudge_left": 2, "nudge_right": 9, "slow_yield": 27, "stop": 24}`
- `replay_oracle`: `{"crawl": 42, "evasive_left": 7, "evasive_right": 124, "lane_recover": 3, "maintain": 8, "nudge_left": 2, "nudge_right": 5, "slow_yield": 22, "stop": 87}`

