# Candidate Recoverable-Regret Audit

- Input: `artifacts/corl2027/nuplan_public_replay_study_interaction1000_expanded/replay.json`
- Scenes: `1000`
- Valid scenes: `1000`
- Claim boundary: Model-agnostic recoverable-regret audit. It measures top-1, oracle@K, generation gap, recoverable safety regret, and recoverable utility regret for candidate generators; it does not assume ManeuverToken actions.

## Gap Decomposition

| Regret type | Count | Rate |
|---|---:|---:|
| generation gap | 223 | 0.223 |
| recoverable safety regret | 453 | 0.453 |
| recoverable utility regret | 22 | 0.022 |
| minor utility regret | 103 | n/a |
| no regret | 199 | n/a |

## Utility-Regret Sensitivity

| Utility threshold | Count | Rate |
|---:|---:|---:|
| 0.001 | 125 | 0.125 |
| 1 | 115 | 0.115 |
| 5 | 22 | 0.022 |
| 10 | 0 | 0.000 |

## Top-1 vs Oracle@K

| Metric | Value |
|---|---:|
| mean top-1 value | -143.702 |
| mean oracle@K value | -40.464 |
| mean recoverable regret | 103.238 |
| positive regret rate | 0.734 |
| top-1 replay fail rate | 0.676 |
| oracle@K replay fail rate | 0.223 |

## Candidate Surface

| Metric | Value |
|---|---:|
| mean candidates | 21.000 |
| top-1 entropy | 2.757 |
| candidate-id entropy | 4.392 |

## Bootstrap Targets

- Target count: `475`
- Target types: `{"recoverable_safety_regret": 453, "recoverable_utility_regret": 22}`

