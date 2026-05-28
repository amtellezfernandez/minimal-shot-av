# nuPlan Regime-Gated Selector

- Input replay JSON: `artifacts/corl2027/nuplan_public_replay_study_interaction1000_expert/replay.json`
- Split unit: `source_db_file`
- Train scenes: `700`
- Holdout scenes: `300`
- Marginal retention floor: `0.500`
- Marginal fail penalty: `100.0`
- Intervention fail penalty: `250.0`
- Train regime accuracy: `0.870`

## Train

| Selector | Replay-infeasible | Rate | Recovery | Progress | ADE3 | FDE3 | Entropy |
|---|---:|---:|---:|---:|---:|---:|---:|
| proxy_selector | 380 | 0.543 | 0/276 | 25.728 | 2.581 | 4.539 | 2.376 |
| marginal_head | 224 | 0.320 | 156/276 | 22.925 | 3.234 | 5.374 | 2.455 |
| intervention_head | 104 | 0.149 | 276/276 | 13.993 | 6.496 | 10.622 | 2.768 |
| oracle_regime_gate | 104 | 0.149 | 276/276 | 19.968 | 4.495 | 7.559 | 2.642 |
| regime_gate | 105 | 0.150 | 275/276 | 17.377 | 5.393 | 9.007 | 2.686 |
| replay_oracle | 104 | 0.149 | 276/276 | 11.221 | 7.702 | 12.717 | 2.677 |

### Regime Diagnostics

- Oracle regimes: `{"intervention": 120, "keep_proxy": 424, "marginal_correction": 156}`
- Predicted regimes: `{"intervention": 190, "keep_proxy": 337, "marginal_correction": 173}`
- Introduced failures: `0`

### Token Histograms

- `proxy_selector`: `{"evasive_left": 92, "evasive_right": 45, "lane_recover": 72, "maintain": 299, "nudge_left": 17, "nudge_right": 73, "slow_yield": 102}`
- `marginal_head`: `{"evasive_left": 106, "evasive_right": 46, "lane_recover": 51, "maintain": 159, "nudge_left": 19, "nudge_right": 78, "slow_yield": 241}`
- `intervention_head`: `{"crawl": 142, "evasive_left": 132, "evasive_right": 36, "lane_recover": 10, "maintain": 60, "nudge_left": 40, "nudge_right": 11, "slow_yield": 159, "stop": 110}`
- `oracle_regime_gate`: `{"crawl": 84, "evasive_left": 45, "evasive_right": 44, "lane_recover": 57, "maintain": 216, "nudge_left": 2, "nudge_right": 34, "slow_yield": 185, "stop": 33}`
- `regime_gate`: `{"crawl": 112, "evasive_left": 38, "evasive_right": 33, "lane_recover": 54, "maintain": 181, "nudge_left": 2, "nudge_right": 28, "slow_yield": 187, "stop": 65}`
- `replay_oracle`: `{"crawl": 113, "evasive_left": 132, "evasive_right": 42, "lane_recover": 22, "maintain": 55, "nudge_left": 8, "nudge_right": 13, "slow_yield": 103, "stop": 212}`

## Holdout

| Selector | Replay-infeasible | Rate | Recovery | Progress | ADE3 | FDE3 | Entropy |
|---|---:|---:|---:|---:|---:|---:|---:|
| proxy_selector | 247 | 0.823 | 0/70 | 33.173 | 2.217 | 3.740 | 2.082 |
| marginal_head | 205 | 0.683 | 42/70 | 31.495 | 2.926 | 4.840 | 2.073 |
| intervention_head | 187 | 0.623 | 60/70 | 25.075 | 4.961 | 8.123 | 2.605 |
| oracle_regime_gate | 185 | 0.617 | 62/70 | 30.225 | 3.184 | 5.193 | 2.223 |
| regime_gate | 189 | 0.630 | 58/70 | 27.386 | 4.159 | 6.786 | 2.329 |
| replay_oracle | 177 | 0.590 | 70/70 | 15.598 | 8.484 | 14.080 | 2.197 |

### Regime Diagnostics

- Oracle regimes: `{"intervention": 20, "keep_proxy": 238, "marginal_correction": 42}`
- Predicted regimes: `{"intervention": 75, "keep_proxy": 171, "marginal_correction": 54}`
- Introduced failures: `0`

### Token Histograms

- `proxy_selector`: `{"evasive_left": 7, "evasive_right": 67, "lane_recover": 57, "maintain": 136, "nudge_left": 6, "nudge_right": 7, "slow_yield": 20}`
- `marginal_head`: `{"evasive_left": 7, "evasive_right": 110, "lane_recover": 27, "maintain": 113, "nudge_left": 4, "nudge_right": 10, "slow_yield": 29}`
- `intervention_head`: `{"crawl": 18, "evasive_left": 9, "evasive_right": 115, "lane_recover": 12, "maintain": 61, "nudge_left": 12, "nudge_right": 9, "slow_yield": 31, "stop": 33}`
- `oracle_regime_gate`: `{"evasive_left": 7, "evasive_right": 101, "lane_recover": 39, "maintain": 106, "nudge_left": 1, "nudge_right": 5, "slow_yield": 22, "stop": 19}`
- `regime_gate`: `{"crawl": 6, "evasive_left": 7, "evasive_right": 115, "lane_recover": 23, "maintain": 88, "nudge_right": 9, "slow_yield": 20, "stop": 32}`
- `replay_oracle`: `{"crawl": 42, "evasive_left": 7, "evasive_right": 124, "lane_recover": 3, "maintain": 8, "nudge_left": 2, "nudge_right": 5, "slow_yield": 22, "stop": 87}`

