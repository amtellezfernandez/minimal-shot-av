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
| g0_proxy_top1 | 418 | 0.597 | 186 | 0.494 | 27.353 | 2.566 | 2.142 |
| g1_student | 283 | 0.404 | 51 | 0.697 | 20.572 | 4.715 | 2.793 |
| teacher_promoted_upper_bound | 245 | 0.350 | 13 | 1.000 | 21.432 | 4.231 | 2.906 |
| replay_oracle | 232 | 0.331 | 0 | 0.601 | 13.239 | 7.340 | 2.703 |

### Token Histograms

- `g0_proxy_top1`: `{"evasive_left": 16, "evasive_right": 112, "lane_recover": 109, "maintain": 337, "nudge_left": 17, "nudge_right": 25, "slow_yield": 84}`
- `g1_student`: `{"crawl": 28, "evasive_left": 83, "evasive_right": 177, "lane_recover": 52, "maintain": 150, "nudge_left": 13, "nudge_right": 16, "slow_yield": 110, "stop": 71}`
- `teacher_promoted_upper_bound`: `{"crawl": 29, "evasive_left": 75, "evasive_right": 137, "lane_recover": 83, "maintain": 132, "nudge_left": 12, "nudge_right": 40, "slow_yield": 134, "stop": 58}`
- `replay_oracle`: `{"crawl": 73, "evasive_left": 94, "evasive_right": 158, "lane_recover": 21, "maintain": 53, "nudge_left": 7, "nudge_right": 18, "slow_yield": 71, "stop": 205}`

## Holdout

| Selector | Replay-infeasible | Rate | Selection Gap | Teacher Agree | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|---:|---:|
| g0_proxy_top1 | 209 | 0.697 | 160 | 0.333 | 29.383 | 2.251 | 2.240 |
| g1_student | 107 | 0.357 | 58 | 0.587 | 20.769 | 5.156 | 2.481 |
| teacher_promoted_upper_bound | 49 | 0.163 | 0 | 1.000 | 19.474 | 5.933 | 2.179 |
| replay_oracle | 49 | 0.163 | 0 | 0.473 | 10.889 | 9.329 | 2.344 |

### Token Histograms

- `g0_proxy_top1`: `{"evasive_left": 83, "lane_recover": 20, "maintain": 98, "nudge_left": 6, "nudge_right": 55, "slow_yield": 38}`
- `g1_student`: `{"crawl": 60, "evasive_left": 15, "evasive_right": 8, "lane_recover": 17, "maintain": 37, "nudge_left": 6, "nudge_right": 45, "slow_yield": 112}`
- `teacher_promoted_upper_bound`: `{"crawl": 86, "evasive_left": 56, "evasive_right": 13, "lane_recover": 3, "maintain": 48, "slow_yield": 94}`
- `replay_oracle`: `{"crawl": 82, "evasive_left": 45, "evasive_right": 8, "lane_recover": 4, "maintain": 10, "nudge_left": 3, "slow_yield": 54, "stop": 94}`

