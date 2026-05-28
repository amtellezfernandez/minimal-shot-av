# nuPlan Boundary Intervention Selector Evaluation

- Input replay JSON: `artifacts/corl2027/nuplan_public_replay_study_interaction1000_expert/replay.json`
- Baseline selector: `artifacts/corl2027/nuplan_replay_calibrated_selector_expert1000_lambda80.json`
- Train scenes: `700`
- Holdout scenes: `300`
- Boundary intervention: `top_k=2, margin_threshold=0.50, intervention_threshold=0.00`

## Train

| Selector | Replay-infeasible | Rate | Recovery | Progress | ADE3 | FDE3 | Entropy |
|---|---:|---:|---:|---:|---:|---:|---:|
| proxy_selector | 418 | 0.597 | 0/186 | 27.353 | 2.566 | 4.651 | 2.142 |
| baseline_selector | 245 | 0.350 | 173/186 | 21.432 | 4.231 | 7.051 | 2.906 |
| boundary_intervention | 244 | 0.349 | 174/186 | 21.421 | 4.240 | 7.066 | 2.907 |
| replay_oracle | 232 | 0.331 | 186/186 | 13.239 | 7.340 | 12.214 | 2.703 |

## Holdout

| Selector | Replay-infeasible | Rate | Recovery | Progress | ADE3 | FDE3 | Entropy |
|---|---:|---:|---:|---:|---:|---:|---:|
| proxy_selector | 209 | 0.697 | 0/160 | 29.383 | 2.251 | 3.479 | 2.240 |
| baseline_selector | 49 | 0.163 | 160/160 | 19.474 | 5.933 | 9.457 | 2.179 |
| boundary_intervention | 49 | 0.163 | 160/160 | 19.679 | 5.902 | 9.431 | 2.209 |
| replay_oracle | 49 | 0.163 | 160/160 | 10.889 | 9.329 | 15.255 | 2.344 |

