# nuPlan Regime-Gated Selector

- Input replay JSON: `artifacts/corl2027/nuplan_public_replay_study_interaction1000_expert/replay.json`
- Split unit: `source_db_file`
- Train scenes: `845`
- Holdout scenes: `155`
- Marginal retention floor: `0.500`
- Marginal fail penalty: `100.0`
- Intervention fail penalty: `250.0`
- Train regime accuracy: `0.879`

## Train

| Selector | Replay-infeasible | Rate | Recovery | Progress | ADE3 | FDE3 | Entropy |
|---|---:|---:|---:|---:|---:|---:|---:|
| proxy_selector | 540 | 0.639 | 0/264 | 28.784 | 2.529 | 4.422 | 2.224 |
| marginal_head | 347 | 0.411 | 193/264 | 26.492 | 3.258 | 5.463 | 2.372 |
| intervention_head | 285 | 0.337 | 255/264 | 20.230 | 5.127 | 8.420 | 2.947 |
| oracle_regime_gate | 285 | 0.337 | 255/264 | 24.960 | 3.707 | 6.262 | 2.559 |
| regime_gate | 289 | 0.342 | 251/264 | 23.101 | 4.294 | 7.188 | 2.664 |
| risk_regime_gate | 321 | 0.380 | 219/264 | 23.067 | 4.338 | 7.339 | 2.818 |
| replay_oracle | 276 | 0.327 | 264/264 | 13.159 | 8.004 | 13.315 | 2.710 |

### Regime Diagnostics

- Oracle regimes: `{"intervention": 62, "keep_proxy": 590, "marginal_correction": 193}`
- Predicted regimes: `{"intervention": 116, "keep_proxy": 498, "marginal_correction": 231}`
- Introduced failures: `0`

### Token Histograms

- `proxy_selector`: `{"evasive_left": 93, "evasive_right": 112, "lane_recover": 117, "maintain": 399, "nudge_left": 12, "nudge_right": 28, "slow_yield": 84}`
- `marginal_head`: `{"evasive_left": 111, "evasive_right": 171, "lane_recover": 34, "maintain": 290, "nudge_left": 23, "nudge_right": 34, "slow_yield": 182}`
- `intervention_head`: `{"crawl": 51, "evasive_left": 111, "evasive_right": 173, "lane_recover": 47, "maintain": 98, "nudge_left": 15, "nudge_right": 79, "slow_yield": 167, "stop": 104}`
- `oracle_regime_gate`: `{"crawl": 19, "evasive_left": 44, "evasive_right": 148, "lane_recover": 91, "maintain": 298, "nudge_left": 3, "nudge_right": 36, "slow_yield": 164, "stop": 42}`
- `regime_gate`: `{"crawl": 33, "evasive_left": 38, "evasive_right": 161, "lane_recover": 82, "maintain": 242, "nudge_left": 4, "nudge_right": 34, "slow_yield": 190, "stop": 61}`
- `risk_regime_gate`: `{"crawl": 33, "evasive_left": 61, "evasive_right": 181, "lane_recover": 79, "maintain": 209, "nudge_left": 6, "nudge_right": 85, "slow_yield": 134, "stop": 57}`
- `replay_oracle`: `{"crawl": 129, "evasive_left": 110, "evasive_right": 166, "lane_recover": 24, "maintain": 57, "nudge_left": 7, "nudge_right": 18, "slow_yield": 94, "stop": 240}`

## Holdout

| Selector | Replay-infeasible | Rate | Recovery | Progress | ADE3 | FDE3 | Entropy |
|---|---:|---:|---:|---:|---:|---:|---:|
| proxy_selector | 87 | 0.561 | 0/82 | 23.482 | 2.159 | 3.629 | 2.253 |
| marginal_head | 76 | 0.490 | 11/82 | 23.086 | 2.684 | 4.457 | 2.127 |
| intervention_head | 5 | 0.032 | 82/82 | 11.901 | 6.377 | 10.111 | 1.901 |
| oracle_regime_gate | 5 | 0.032 | 82/82 | 13.009 | 6.055 | 9.719 | 2.007 |
| regime_gate | 9 | 0.058 | 78/82 | 13.031 | 6.024 | 9.694 | 1.998 |
| risk_regime_gate | 7 | 0.045 | 80/82 | 12.775 | 5.985 | 9.580 | 2.121 |
| replay_oracle | 5 | 0.032 | 82/82 | 9.125 | 7.570 | 12.098 | 2.218 |

### Regime Diagnostics

- Oracle regimes: `{"intervention": 71, "keep_proxy": 73, "marginal_correction": 11}`
- Predicted regimes: `{"intervention": 72, "keep_proxy": 76, "marginal_correction": 7}`
- Introduced failures: `0`

### Token Histograms

- `proxy_selector`: `{"evasive_left": 6, "lane_recover": 12, "maintain": 36, "nudge_left": 11, "nudge_right": 52, "slow_yield": 38}`
- `marginal_head`: `{"evasive_left": 29, "lane_recover": 1, "maintain": 38, "nudge_left": 3, "nudge_right": 41, "slow_yield": 43}`
- `intervention_head`: `{"crawl": 72, "evasive_left": 30, "lane_recover": 1, "maintain": 5, "nudge_left": 8, "slow_yield": 39}`
- `oracle_regime_gate`: `{"crawl": 71, "evasive_left": 8, "lane_recover": 5, "maintain": 25, "nudge_right": 6, "slow_yield": 40}`
- `regime_gate`: `{"crawl": 72, "evasive_left": 4, "lane_recover": 8, "maintain": 25, "nudge_right": 7, "slow_yield": 39}`
- `risk_regime_gate`: `{"crawl": 72, "evasive_left": 6, "lane_recover": 6, "maintain": 20, "nudge_left": 5, "nudge_right": 7, "slow_yield": 39}`
- `replay_oracle`: `{"crawl": 26, "evasive_left": 29, "lane_recover": 1, "maintain": 6, "nudge_left": 3, "slow_yield": 31, "stop": 59}`

