# nuPlan Regime-Gated Selector

- Input replay JSON: `artifacts/corl2027/nuplan_public_replay_study_interaction1000_expert/replay.json`
- Split unit: `source_db_file`
- Train scenes: `845`
- Holdout scenes: `155`
- Marginal retention floor: `0.500`
- Marginal fail penalty: `100.0`
- Intervention fail penalty: `250.0`
- Train regime accuracy: `0.863`

## Train

| Selector | Replay-infeasible | Rate | Recovery | Progress | ADE3 | FDE3 | Entropy |
|---|---:|---:|---:|---:|---:|---:|---:|
| proxy_selector | 547 | 0.647 | 0/284 | 28.472 | 2.388 | 4.001 | 2.411 |
| marginal_head | 373 | 0.441 | 174/284 | 25.791 | 3.164 | 5.104 | 2.520 |
| intervention_head | 272 | 0.322 | 277/284 | 17.293 | 6.098 | 9.898 | 2.948 |
| oracle_regime_gate | 268 | 0.317 | 279/284 | 23.543 | 4.104 | 6.738 | 2.630 |
| regime_gate | 274 | 0.324 | 274/284 | 20.901 | 5.067 | 8.341 | 2.704 |
| risk_regime_gate | 313 | 0.370 | 236/284 | 20.600 | 5.064 | 8.337 | 2.922 |
| replay_oracle | 263 | 0.311 | 284/284 | 12.179 | 8.327 | 13.705 | 2.591 |

### Regime Diagnostics

- Oracle regimes: `{"intervention": 105, "keep_proxy": 566, "marginal_correction": 174}`
- Predicted regimes: `{"intervention": 174, "keep_proxy": 458, "marginal_correction": 213}`
- Introduced failures: `1`

### Token Histograms

- `proxy_selector`: `{"evasive_left": 99, "evasive_right": 112, "lane_recover": 125, "maintain": 337, "nudge_left": 12, "nudge_right": 64, "slow_yield": 96}`
- `marginal_head`: `{"evasive_left": 126, "evasive_right": 162, "lane_recover": 72, "maintain": 203, "nudge_left": 17, "nudge_right": 50, "slow_yield": 215}`
- `intervention_head`: `{"crawl": 144, "evasive_left": 130, "evasive_right": 168, "lane_recover": 35, "maintain": 20, "nudge_left": 46, "nudge_right": 74, "slow_yield": 109, "stop": 119}`
- `oracle_regime_gate`: `{"crawl": 65, "evasive_left": 52, "evasive_right": 143, "lane_recover": 93, "maintain": 256, "nudge_right": 16, "slow_yield": 180, "stop": 40}`
- `regime_gate`: `{"crawl": 101, "evasive_left": 37, "evasive_right": 150, "lane_recover": 79, "maintain": 214, "nudge_right": 14, "slow_yield": 177, "stop": 73}`
- `risk_regime_gate`: `{"crawl": 85, "evasive_left": 80, "evasive_right": 166, "lane_recover": 80, "maintain": 146, "nudge_left": 3, "nudge_right": 59, "slow_yield": 155, "stop": 71}`
- `replay_oracle`: `{"crawl": 132, "evasive_left": 125, "evasive_right": 159, "lane_recover": 14, "maintain": 38, "nudge_left": 9, "nudge_right": 5, "slow_yield": 106, "stop": 257}`

## Holdout

| Selector | Replay-infeasible | Rate | Recovery | Progress | ADE3 | FDE3 | Entropy |
|---|---:|---:|---:|---:|---:|---:|---:|
| proxy_selector | 80 | 0.516 | 0/62 | 25.180 | 2.929 | 5.926 | 1.595 |
| marginal_head | 51 | 0.329 | 30/62 | 22.284 | 3.002 | 5.692 | 2.667 |
| intervention_head | 23 | 0.148 | 62/62 | 11.632 | 6.201 | 10.214 | 2.563 |
| oracle_regime_gate | 18 | 0.116 | 62/62 | 19.888 | 4.173 | 7.535 | 2.369 |
| regime_gate | 34 | 0.219 | 51/62 | 15.600 | 5.139 | 8.722 | 2.466 |
| risk_regime_gate | 31 | 0.200 | 51/62 | 18.051 | 4.351 | 7.682 | 2.780 |
| replay_oracle | 18 | 0.116 | 62/62 | 14.471 | 5.808 | 9.968 | 2.847 |

### Regime Diagnostics

- Oracle regimes: `{"intervention": 32, "keep_proxy": 93, "marginal_correction": 30}`
- Predicted regimes: `{"intervention": 62, "keep_proxy": 53, "marginal_correction": 40}`
- Introduced failures: `5`

### Token Histograms

- `proxy_selector`: `{"lane_recover": 4, "maintain": 98, "nudge_left": 11, "nudge_right": 16, "slow_yield": 26}`
- `marginal_head`: `{"evasive_left": 13, "evasive_right": 11, "lane_recover": 18, "maintain": 35, "nudge_left": 17, "nudge_right": 22, "slow_yield": 39}`
- `intervention_head`: `{"crawl": 48, "evasive_left": 12, "evasive_right": 6, "lane_recover": 3, "maintain": 3, "nudge_left": 3, "nudge_right": 12, "slow_yield": 38, "stop": 30}`
- `oracle_regime_gate`: `{"crawl": 15, "evasive_right": 3, "lane_recover": 3, "maintain": 63, "nudge_left": 2, "nudge_right": 25, "slow_yield": 27, "stop": 17}`
- `regime_gate`: `{"crawl": 28, "evasive_right": 8, "lane_recover": 3, "maintain": 35, "nudge_right": 13, "slow_yield": 48, "stop": 20}`
- `risk_regime_gate`: `{"crawl": 19, "evasive_left": 4, "evasive_right": 1, "lane_recover": 12, "maintain": 34, "nudge_left": 8, "nudge_right": 22, "slow_yield": 39, "stop": 16}`
- `replay_oracle`: `{"crawl": 23, "evasive_left": 14, "evasive_right": 7, "lane_recover": 11, "maintain": 25, "nudge_left": 1, "nudge_right": 13, "slow_yield": 19, "stop": 42}`

