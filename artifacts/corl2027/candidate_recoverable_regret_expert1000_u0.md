# Candidate Recoverable-Regret Audit

- Input: `artifacts/corl2027/nuplan_public_replay_study_interaction1000_expert/replay.json`
- Scenes: `1000`
- Valid scenes: `1000`
- Claim boundary: Model-agnostic recoverable-regret audit. It measures top-1, oracle@K, generation gap, recoverable safety regret, and recoverable utility regret for candidate generators; it does not assume ManeuverToken actions.

## Gap Decomposition

| Regret type | Count | Rate |
|---|---:|---:|
| generation gap | 281 | 0.281 |
| recoverable safety regret | 346 | 0.346 |
| recoverable utility regret | 44 | 0.044 |
| minor utility regret | 0 | n/a |
| no regret | 329 | n/a |

## Utility-Regret Sensitivity

| Utility threshold | Count | Rate |
|---:|---:|---:|
| 0.001 | 44 | 0.044 |
| 1 | 2 | 0.002 |
| 5 | 0 | 0.000 |
| 10 | 0 | 0.000 |

## Top-1 vs Oracle@K

| Metric | Value |
|---|---:|
| mean top-1 value | -133.732 |
| mean oracle@K value | -54.083 |
| mean recoverable regret | 79.649 |
| positive regret rate | 0.509 |
| top-1 replay fail rate | 0.627 |
| oracle@K replay fail rate | 0.281 |

## Candidate Surface

| Metric | Value |
|---|---:|
| mean candidates | 9.000 |
| top-1 entropy | 2.375 |
| candidate-id entropy | 3.170 |

## Bootstrap Targets

- Target count: `390`
- Target types: `{"recoverable_safety_regret": 346, "recoverable_utility_regret": 44}`

