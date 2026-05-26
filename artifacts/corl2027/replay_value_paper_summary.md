# Expanded-Bank Replay-Value Summary

This summary consolidates the paper-facing holdout results on the 1,000-scene interaction-aware public nuPlan replay study.

## Generation Boundary

| Candidate bank | Oracle replay fail | Generation gap |
|---|---:|---:|
| 9-token ManeuverToken | 0.281 | 281 |
| 21-token expanded bank | 0.223 | 223 |

Interpretation: candidate expansion improves the oracle boundary directly.

## Holdout Comparison On Expanded 21-Token Bank

| Selector | Replay fail | Progress (m) | ADE3 (m) | FDE3 (m) | Entropy |
|---|---:|---:|---:|---:|---:|
| proxy top-1 | 0.740 | 31.181 | 2.714 | 4.136 | 2.773 |
| scene-token student | 0.460 | 23.062 | 4.737 | 7.555 | 2.471 |
| replay-value, safety-heavy | 0.097 | 12.276 | 9.190 | 14.804 | 3.089 |
| replay-value, balanced | 0.097 | 14.278 | 8.451 | 13.571 | 3.117 |
| replay-value oracle | 0.097 | 11.258 | 9.601 | 15.488 | 3.053 |

Interpretation:

- The scene-token student underuses the richer candidate bank.
- Candidate-level replay value reaches the oracle failure boundary.
- A balanced replay-value objective preserves more progress than the safety-heavy point while keeping the same failure rate.

## Source Artifacts

- `artifacts/corl2027/nuplan_recoverable_regret_bootstrap_interaction1000_expanded_iter3.json`
- `artifacts/corl2027/nuplan_recoverable_regret_bootstrap_interaction1000_expanded_replay_value.json`
- `artifacts/corl2027/nuplan_replay_value_selector_interaction1000_expanded_mlp_eval.json`
- `artifacts/corl2027/nuplan_replay_value_selector_interaction1000_expanded_balanced_eval.json`
