# Candidate Recoverable-Regret Audit

- Input: `artifacts/corl2027/nuplan_bootstrap_candidate_loop_expert1000_lambda80_g1_replay.json`
- Scenes: `1000`
- Valid scenes: `1000`
- Claim boundary: Model-agnostic recoverable-regret audit. It measures top-1, oracle@K, generation gap, recoverable safety regret, and recoverable utility regret for candidate generators; it does not assume ManeuverToken actions.

## Gap Decomposition

| Regret type | Count | Rate |
|---|---:|---:|
| generation gap | 281 | 0.281 |
| recoverable safety regret | 13 | 0.013 |
| recoverable utility regret | 228 | 0.228 |
| minor utility regret | 0 | n/a |
| no regret | 478 | n/a |

## Utility-Regret Sensitivity

| Utility threshold | Count | Rate |
|---:|---:|---:|
| 0.001 | 228 | 0.228 |
| 1 | 163 | 0.163 |
| 5 | 134 | 0.134 |
| 10 | 25 | 0.025 |

## Top-1 vs Oracle@K

| Metric | Value |
|---|---:|
| mean top-1 value | -62.138 |
| mean oracle@K value | -54.083 |
| mean recoverable regret | 8.056 |
| positive regret rate | 0.388 |
| top-1 replay fail rate | 0.294 |
| oracle@K replay fail rate | 0.281 |

## Candidate Surface

| Metric | Value |
|---|---:|
| mean candidates | 9.000 |
| top-1 entropy | 2.890 |
| candidate-id entropy | 3.170 |

## Bootstrap Targets

- Target count: `241`
- Target types: `{"recoverable_safety_regret": 13, "recoverable_utility_regret": 228}`

