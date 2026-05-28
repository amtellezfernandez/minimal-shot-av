# nuPlan Selective Regret Selector Evaluation

- Input replay JSON: `artifacts/corl2027/nuplan_public_replay_study_interaction1000_expert/replay.json`
- Baseline selector: `artifacts/corl2027/nuplan_replay_calibrated_selector_expert1000_lambda80.json`
- Train scenes: `700`
- Holdout scenes: `300`
- Selective regret: `top_k=2, margin_threshold=0.25, intervention_threshold=5.00`

## Train

| Selector | Replay-infeasible | Rate | Recovery | Progress | ADE3 | FDE3 | Entropy |
|---|---:|---:|---:|---:|---:|---:|---:|
| proxy_selector | 418 | 0.597 | 0/186 | 27.353 | 2.566 | 4.651 | 2.142 |
| baseline_selector | 245 | 0.350 | 173/186 | 21.432 | 4.231 | 7.051 | 2.906 |
| selective_regret | 244 | 0.349 | 174/186 | 21.428 | 4.239 | 7.065 | 2.904 |
| replay_oracle | 232 | 0.331 | 186/186 | 13.239 | 7.340 | 12.214 | 2.703 |

## Holdout

| Selector | Replay-infeasible | Rate | Recovery | Progress | ADE3 | FDE3 | Entropy |
|---|---:|---:|---:|---:|---:|---:|---:|
| proxy_selector | 209 | 0.697 | 0/160 | 29.383 | 2.251 | 3.479 | 2.240 |
| baseline_selector | 49 | 0.163 | 160/160 | 19.474 | 5.933 | 9.457 | 2.179 |
| selective_regret | 49 | 0.163 | 160/160 | 19.539 | 5.936 | 9.479 | 2.180 |
| replay_oracle | 49 | 0.163 | 160/160 | 10.889 | 9.329 | 15.255 | 2.344 |

