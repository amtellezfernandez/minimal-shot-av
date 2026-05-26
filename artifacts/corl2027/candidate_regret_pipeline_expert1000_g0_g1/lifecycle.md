# Candidate Regret Lifecycle

- Stages: `2`
- Interpretation: bootstrap closes much of the recoverable safety regret, leaves generation gap unchanged, and exposes utility regret as the next optimization branch

## Stage Table

| Stage | Gen gap | Safety regret | Utility regret | Mean regret | Top-1 fail | Oracle fail |
|---|---:|---:|---:|---:|---:|---:|
| g0 | 281 | 346 | 44 | 79.649 | 0.627 | 0.281 |
| g1 | 281 | 13 | 228 | 8.056 | 0.294 | 0.281 |

## Deltas

| From | To | Gen gap Δ | Safety Δ | Utility Δ | Mean regret Δ |
|---|---|---:|---:|---:|---:|
| g0 | g1 | 0 | -333 | 184 | -71.594 |
