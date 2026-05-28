# nuPlan Border Residual Selector Evaluation

- Input replay JSON: `artifacts/corl2027/nuplan_public_replay_study_interaction1000_expert/replay.json`
- Baseline selector: `artifacts/corl2027/nuplan_replay_calibrated_selector_expert1000_lambda80.json`
- Split unit: `source_db_file`
- Train scenes: `700`
- Holdout scenes: `300`
- Near-miss threshold: `1.00 m`
- Objective: `-250.0 * replay_failure + 1.0 * progress - 2.0 * ADE3`
- Border residual: `recoverable_bonus=4.0, margin_temperature=0.50, residual_scale=0.50, margin_threshold=0.75`

## Train

| Selector | Replay-infeasible | Rate | Recovery | Progress | ADE3 | FDE3 | Entropy |
|---|---:|---:|---:|---:|---:|---:|---:|
| proxy_selector | 418 | 0.597 | 0/186 | 27.353 | 2.566 | 4.651 | 2.142 |
| baseline_selector | 245 | 0.350 | 173/186 | 21.432 | 4.231 | 7.051 | 2.906 |
| border_residual | 299 | 0.427 | 127/186 | 22.349 | 3.782 | 6.485 | 2.978 |
| replay_oracle | 232 | 0.331 | 186/186 | 13.239 | 7.340 | 12.214 | 2.703 |

## Holdout

| Selector | Replay-infeasible | Rate | Recovery | Progress | ADE3 | FDE3 | Entropy |
|---|---:|---:|---:|---:|---:|---:|---:|
| proxy_selector | 209 | 0.697 | 0/160 | 29.383 | 2.251 | 3.479 | 2.240 |
| baseline_selector | 49 | 0.163 | 160/160 | 19.474 | 5.933 | 9.457 | 2.179 |
| border_residual | 73 | 0.243 | 146/160 | 15.300 | 7.459 | 12.123 | 2.647 |
| replay_oracle | 49 | 0.163 | 160/160 | 10.889 | 9.329 | 15.255 | 2.344 |

