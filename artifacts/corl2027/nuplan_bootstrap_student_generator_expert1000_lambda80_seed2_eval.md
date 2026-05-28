# nuPlan Bootstrap Student Generator

- Input replay JSON: `artifacts/corl2027/nuplan_public_replay_study_interaction1000_expert/replay.json`
- Teacher targets: `artifacts/corl2027/nuplan_bootstrap_lambda80_teacher_targets_expert1000.jsonl`
- Split unit: `source_db_file`
- Train scenes: `700`
- Holdout scenes: `300`
- Training scene accuracy: `0.697`

## Train

| Selector | Replay-infeasible | Rate | Selection Gap | Teacher Agree | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|---:|---:|
| g0_proxy_top1 | 428 | 0.611 | 265 | 0.420 | 27.518 | 2.392 | 2.472 |
| g1_student | 246 | 0.351 | 83 | 0.697 | 20.366 | 4.583 | 2.875 |
| teacher_promoted_upper_bound | 167 | 0.239 | 4 | 1.000 | 19.994 | 4.888 | 2.694 |
| replay_oracle | 163 | 0.233 | 0 | 0.577 | 13.043 | 7.658 | 2.571 |

### Token Histograms

- `g0_proxy_top1`: `{"evasive_left": 92, "evasive_right": 106, "lane_recover": 93, "maintain": 258, "nudge_left": 11, "nudge_right": 64, "slow_yield": 76}`
- `g1_student`: `{"crawl": 59, "evasive_left": 133, "evasive_right": 128, "lane_recover": 44, "maintain": 58, "nudge_left": 24, "nudge_right": 46, "slow_yield": 178, "stop": 30}`
- `teacher_promoted_upper_bound`: `{"crawl": 87, "evasive_left": 110, "evasive_right": 147, "lane_recover": 47, "maintain": 93, "nudge_right": 12, "slow_yield": 178, "stop": 26}`
- `replay_oracle`: `{"crawl": 92, "evasive_left": 118, "evasive_right": 159, "lane_recover": 12, "maintain": 31, "nudge_left": 7, "nudge_right": 1, "slow_yield": 88, "stop": 192}`

## Holdout

| Selector | Replay-infeasible | Rate | Selection Gap | Teacher Agree | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|---:|---:|
| g0_proxy_top1 | 199 | 0.663 | 81 | 0.507 | 28.998 | 2.658 | 1.882 |
| g1_student | 163 | 0.543 | 45 | 0.643 | 23.878 | 4.392 | 2.568 |
| teacher_promoted_upper_bound | 127 | 0.423 | 9 | 1.000 | 22.830 | 4.400 | 2.835 |
| replay_oracle | 118 | 0.393 | 0 | 0.530 | 11.346 | 8.587 | 2.613 |

### Token Histograms

- `g0_proxy_top1`: `{"evasive_left": 7, "evasive_right": 6, "lane_recover": 36, "maintain": 177, "nudge_left": 12, "nudge_right": 16, "slow_yield": 46}`
- `g1_student`: `{"crawl": 13, "evasive_left": 22, "evasive_right": 6, "lane_recover": 10, "maintain": 126, "nudge_left": 21, "nudge_right": 23, "slow_yield": 55, "stop": 24}`
- `teacher_promoted_upper_bound`: `{"crawl": 28, "evasive_left": 21, "evasive_right": 3, "lane_recover": 39, "maintain": 87, "nudge_left": 12, "nudge_right": 28, "slow_yield": 50, "stop": 32}`
- `replay_oracle`: `{"crawl": 63, "evasive_left": 21, "evasive_right": 7, "lane_recover": 13, "maintain": 32, "nudge_left": 3, "nudge_right": 17, "slow_yield": 37, "stop": 107}`

