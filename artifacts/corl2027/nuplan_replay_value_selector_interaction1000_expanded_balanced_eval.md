# nuPlan Replay-Value Selector Evaluation

- Input replay JSON: `artifacts/corl2027/nuplan_public_replay_study_interaction1000_expanded/replay.json`
- Split unit: `source_db_file`
- Train scenes: `700`
- Holdout scenes: `300`
- Near-miss threshold: `1.00 m`
- Model kind: `mlp`
- Objective: `-250.0 * replay_failure + 2.0 * progress - 2.0 * ADE3`

## Train

| Selector | Replay-infeasible | Rate | Recovery | Progress | ADE3 | FDE3 | Entropy |
|---|---:|---:|---:|---:|---:|---:|---:|
| proxy_selector | 454 | 0.649 | 0/260 | 31.210 | 3.053 | 5.165 | 2.391 |
| replay_value | 209 | 0.299 | 245/260 | 22.561 | 4.685 | 7.657 | 3.807 |
| replay_oracle | 194 | 0.277 | 260/260 | 12.810 | 7.900 | 12.974 | 3.927 |

### Token Histograms

- `proxy_selector`: `{"evasive_left": 16, "evasive_right": 92, "fast_left": 201, "fast_right": 230, "lane_recover": 17, "maintain": 110, "nudge_left": 5, "nudge_right": 1, "slow_yield": 18, "wide_right": 10}`
- `replay_value`: `{"brake_left": 17, "brake_right": 16, "crawl": 5, "creep_right": 5, "evasive_left": 11, "evasive_right": 84, "fast_left": 63, "fast_right": 72, "lane_recover": 23, "maintain": 127, "micro_creep": 8, "nudge_left": 7, "nudge_right": 27, "reverse_creep": 28, "slow_yield": 49, "stop": 11, "wide_left": 55, "wide_left_crawl": 47, "wide_right": 32, "wide_right_crawl": 13}`
- `replay_oracle`: `{"brake_left": 22, "brake_right": 21, "crawl": 11, "creep_left": 23, "creep_right": 14, "evasive_left": 10, "evasive_right": 85, "fast_left": 10, "fast_right": 55, "lane_recover": 17, "maintain": 30, "micro_creep": 13, "nudge_left": 6, "nudge_right": 13, "reverse_creep": 108, "slow_yield": 17, "stop": 39, "wide_left": 64, "wide_left_crawl": 88, "wide_right": 40, "wide_right_crawl": 14}`

## Holdout

| Selector | Replay-infeasible | Rate | Recovery | Progress | ADE3 | FDE3 | Entropy |
|---|---:|---:|---:|---:|---:|---:|---:|
| proxy_selector | 222 | 0.740 | 0/193 | 31.181 | 2.714 | 4.136 | 2.773 |
| replay_value | 29 | 0.097 | 193/193 | 14.278 | 8.451 | 13.571 | 3.117 |
| replay_oracle | 29 | 0.097 | 193/193 | 11.258 | 9.601 | 15.488 | 3.053 |

### Token Histograms

- `proxy_selector`: `{"evasive_left": 78, "fast_left": 49, "fast_right": 54, "lane_recover": 6, "maintain": 39, "nudge_right": 41, "slow_yield": 11, "wide_left": 17, "wide_right": 5}`
- `replay_value`: `{"brake_left": 7, "crawl": 31, "creep_left": 13, "evasive_left": 19, "evasive_right": 8, "fast_left": 11, "maintain": 24, "slow_yield": 13, "stop": 5, "wide_left": 38, "wide_left_crawl": 101, "wide_right": 5, "wide_right_crawl": 25}`
- `replay_oracle`: `{"brake_left": 7, "brake_right": 2, "crawl": 13, "creep_left": 11, "evasive_left": 6, "evasive_right": 8, "fast_left": 11, "maintain": 4, "reverse_creep": 4, "slow_yield": 10, "stop": 28, "wide_left": 46, "wide_left_crawl": 105, "wide_right": 5, "wide_right_crawl": 40}`

