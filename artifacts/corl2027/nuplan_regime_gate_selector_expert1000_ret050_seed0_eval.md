# nuPlan Regime-Gated Selector

- Input replay JSON: `artifacts/corl2027/nuplan_public_replay_study_interaction1000_expert/replay.json`
- Split unit: `source_db_file`
- Train scenes: `700`
- Holdout scenes: `300`
- Marginal retention floor: `0.500`
- Marginal fail penalty: `100.0`
- Intervention fail penalty: `250.0`
- Train regime accuracy: `0.930`

## Train

| Selector | Replay-infeasible | Rate | Recovery | Progress | ADE3 | FDE3 | Entropy |
|---|---:|---:|---:|---:|---:|---:|---:|
| proxy_selector | 418 | 0.597 | 0/186 | 27.353 | 2.566 | 4.651 | 2.142 |
| marginal_head | 297 | 0.424 | 121/186 | 25.189 | 3.008 | 5.055 | 2.531 |
| intervention_head | 237 | 0.339 | 181/186 | 19.974 | 4.679 | 7.683 | 2.922 |
| oracle_regime_gate | 237 | 0.339 | 181/186 | 23.820 | 3.592 | 6.172 | 2.505 |
| regime_gate | 238 | 0.340 | 180/186 | 22.878 | 3.897 | 6.642 | 2.541 |
| replay_oracle | 232 | 0.331 | 186/186 | 13.239 | 7.340 | 12.214 | 2.703 |

### Regime Diagnostics

- Oracle regimes: `{"intervention": 60, "keep_proxy": 519, "marginal_correction": 121}`
- Predicted regimes: `{"intervention": 73, "keep_proxy": 473, "marginal_correction": 154}`
- Introduced failures: `0`

### Token Histograms

- `proxy_selector`: `{"evasive_left": 16, "evasive_right": 112, "lane_recover": 109, "maintain": 337, "nudge_left": 17, "nudge_right": 25, "slow_yield": 84}`
- `marginal_head`: `{"evasive_left": 94, "evasive_right": 173, "lane_recover": 46, "maintain": 191, "nudge_left": 33, "nudge_right": 36, "slow_yield": 127}`
- `intervention_head`: `{"crawl": 31, "evasive_left": 93, "evasive_right": 152, "lane_recover": 61, "maintain": 105, "nudge_left": 11, "nudge_right": 38, "slow_yield": 112, "stop": 97}`
- `oracle_regime_gate`: `{"crawl": 15, "evasive_left": 9, "evasive_right": 135, "lane_recover": 90, "maintain": 251, "nudge_left": 3, "nudge_right": 36, "slow_yield": 117, "stop": 44}`
- `regime_gate`: `{"crawl": 18, "evasive_left": 7, "evasive_right": 149, "lane_recover": 80, "maintain": 230, "nudge_left": 4, "nudge_right": 32, "slow_yield": 125, "stop": 55}`
- `replay_oracle`: `{"crawl": 73, "evasive_left": 94, "evasive_right": 158, "lane_recover": 21, "maintain": 53, "nudge_left": 7, "nudge_right": 18, "slow_yield": 71, "stop": 205}`

## Holdout

| Selector | Replay-infeasible | Rate | Recovery | Progress | ADE3 | FDE3 | Entropy |
|---|---:|---:|---:|---:|---:|---:|---:|
| proxy_selector | 209 | 0.697 | 0/160 | 29.383 | 2.251 | 3.479 | 2.240 |
| marginal_head | 125 | 0.417 | 84/160 | 25.872 | 3.414 | 5.365 | 2.265 |
| intervention_head | 49 | 0.163 | 160/160 | 17.055 | 6.705 | 10.792 | 2.645 |
| oracle_regime_gate | 49 | 0.163 | 160/160 | 20.712 | 5.488 | 8.774 | 2.551 |
| regime_gate | 55 | 0.183 | 154/160 | 20.294 | 5.627 | 9.039 | 2.502 |
| replay_oracle | 49 | 0.163 | 160/160 | 10.889 | 9.329 | 15.255 | 2.344 |

### Regime Diagnostics

- Oracle regimes: `{"intervention": 76, "keep_proxy": 140, "marginal_correction": 84}`
- Predicted regimes: `{"intervention": 79, "keep_proxy": 111, "marginal_correction": 110}`
- Introduced failures: `0`

### Token Histograms

- `proxy_selector`: `{"evasive_left": 83, "lane_recover": 20, "maintain": 98, "nudge_left": 6, "nudge_right": 55, "slow_yield": 38}`
- `marginal_head`: `{"evasive_left": 51, "evasive_right": 13, "lane_recover": 7, "maintain": 64, "nudge_left": 3, "nudge_right": 51, "slow_yield": 111}`
- `intervention_head`: `{"crawl": 72, "evasive_left": 46, "evasive_right": 8, "lane_recover": 8, "maintain": 17, "nudge_left": 8, "nudge_right": 16, "slow_yield": 95, "stop": 30}`
- `oracle_regime_gate`: `{"crawl": 52, "evasive_left": 43, "evasive_right": 13, "lane_recover": 6, "maintain": 68, "nudge_right": 6, "slow_yield": 89, "stop": 23}`
- `regime_gate`: `{"crawl": 53, "evasive_left": 20, "evasive_right": 13, "lane_recover": 9, "maintain": 70, "nudge_right": 7, "slow_yield": 102, "stop": 26}`
- `replay_oracle`: `{"crawl": 82, "evasive_left": 45, "evasive_right": 8, "lane_recover": 4, "maintain": 10, "nudge_left": 3, "slow_yield": 54, "stop": 94}`

