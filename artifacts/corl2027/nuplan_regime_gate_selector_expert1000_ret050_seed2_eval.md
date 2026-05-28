# nuPlan Regime-Gated Selector

- Input replay JSON: `artifacts/corl2027/nuplan_public_replay_study_interaction1000_expert/replay.json`
- Split unit: `source_db_file`
- Train scenes: `700`
- Holdout scenes: `300`
- Marginal retention floor: `0.500`
- Marginal fail penalty: `100.0`
- Intervention fail penalty: `250.0`
- Train regime accuracy: `0.943`

## Train

| Selector | Replay-infeasible | Rate | Recovery | Progress | ADE3 | FDE3 | Entropy |
|---|---:|---:|---:|---:|---:|---:|---:|
| proxy_selector | 428 | 0.611 | 0/265 | 27.518 | 2.392 | 3.938 | 2.472 |
| marginal_head | 255 | 0.364 | 173/265 | 24.419 | 3.294 | 5.213 | 2.444 |
| intervention_head | 167 | 0.239 | 261/265 | 15.307 | 6.626 | 10.705 | 2.771 |
| oracle_regime_gate | 167 | 0.239 | 261/265 | 22.081 | 4.261 | 6.906 | 2.631 |
| regime_gate | 167 | 0.239 | 261/265 | 21.487 | 4.453 | 7.214 | 2.594 |
| replay_oracle | 163 | 0.233 | 265/265 | 13.043 | 7.658 | 12.482 | 2.571 |

### Regime Diagnostics

- Oracle regimes: `{"intervention": 88, "keep_proxy": 439, "marginal_correction": 173}`
- Predicted regimes: `{"intervention": 92, "keep_proxy": 399, "marginal_correction": 209}`
- Introduced failures: `0`

### Token Histograms

- `proxy_selector`: `{"evasive_left": 92, "evasive_right": 106, "lane_recover": 93, "maintain": 258, "nudge_left": 11, "nudge_right": 64, "slow_yield": 76}`
- `marginal_head`: `{"evasive_left": 118, "evasive_right": 150, "lane_recover": 28, "maintain": 146, "nudge_left": 9, "nudge_right": 53, "slow_yield": 196}`
- `intervention_head`: `{"crawl": 146, "evasive_left": 119, "evasive_right": 167, "lane_recover": 18, "maintain": 12, "nudge_left": 24, "nudge_right": 33, "slow_yield": 79, "stop": 102}`
- `oracle_regime_gate`: `{"crawl": 56, "evasive_left": 45, "evasive_right": 143, "lane_recover": 64, "maintain": 185, "nudge_right": 12, "slow_yield": 163, "stop": 32}`
- `regime_gate`: `{"crawl": 56, "evasive_left": 39, "evasive_right": 153, "lane_recover": 55, "maintain": 174, "nudge_right": 9, "slow_yield": 178, "stop": 36}`
- `replay_oracle`: `{"crawl": 92, "evasive_left": 118, "evasive_right": 159, "lane_recover": 12, "maintain": 31, "nudge_left": 7, "nudge_right": 1, "slow_yield": 88, "stop": 192}`

## Holdout

| Selector | Replay-infeasible | Rate | Recovery | Progress | ADE3 | FDE3 | Entropy |
|---|---:|---:|---:|---:|---:|---:|---:|
| proxy_selector | 199 | 0.663 | 0/81 | 28.998 | 2.658 | 5.142 | 1.882 |
| marginal_head | 167 | 0.557 | 32/81 | 27.659 | 2.966 | 5.533 | 2.374 |
| intervention_head | 125 | 0.417 | 78/81 | 16.822 | 6.136 | 10.324 | 2.722 |
| oracle_regime_gate | 119 | 0.397 | 80/81 | 24.966 | 3.844 | 6.882 | 2.438 |
| regime_gate | 157 | 0.523 | 42/81 | 25.627 | 3.829 | 6.973 | 2.172 |
| replay_oracle | 118 | 0.393 | 81/81 | 11.346 | 8.587 | 14.628 | 2.613 |

### Regime Diagnostics

- Oracle regimes: `{"intervention": 48, "keep_proxy": 220, "marginal_correction": 32}`
- Predicted regimes: `{"intervention": 28, "keep_proxy": 237, "marginal_correction": 35}`
- Introduced failures: `0`

### Token Histograms

- `proxy_selector`: `{"evasive_left": 7, "evasive_right": 6, "lane_recover": 36, "maintain": 177, "nudge_left": 12, "nudge_right": 16, "slow_yield": 46}`
- `marginal_head`: `{"evasive_left": 21, "evasive_right": 12, "lane_recover": 37, "maintain": 123, "nudge_left": 20, "nudge_right": 23, "slow_yield": 64}`
- `intervention_head`: `{"crawl": 43, "evasive_left": 21, "evasive_right": 6, "lane_recover": 8, "maintain": 30, "nudge_left": 24, "nudge_right": 13, "slow_yield": 99, "stop": 56}`
- `oracle_regime_gate`: `{"crawl": 15, "evasive_left": 7, "evasive_right": 5, "lane_recover": 32, "maintain": 134, "nudge_left": 3, "nudge_right": 26, "slow_yield": 45, "stop": 33}`
- `regime_gate`: `{"crawl": 12, "evasive_left": 7, "evasive_right": 12, "lane_recover": 26, "maintain": 158, "nudge_right": 14, "slow_yield": 55, "stop": 16}`
- `replay_oracle`: `{"crawl": 63, "evasive_left": 21, "evasive_right": 7, "lane_recover": 13, "maintain": 32, "nudge_left": 3, "nudge_right": 17, "slow_yield": 37, "stop": 107}`

