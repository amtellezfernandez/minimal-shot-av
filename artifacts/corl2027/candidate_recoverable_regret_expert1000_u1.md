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
| recoverable utility regret | 2 | 0.002 |
| minor utility regret | 42 | n/a |
| no regret | 329 | n/a |

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

- Target count: `348`
- Target types: `{"recoverable_safety_regret": 346, "recoverable_utility_regret": 2}`

