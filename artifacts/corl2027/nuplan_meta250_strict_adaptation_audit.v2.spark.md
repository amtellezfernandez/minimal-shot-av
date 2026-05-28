# Meta250 Strict Adaptation Audit

Outcome-defined held-out groups on the seed-0 DB split.

| Group | Scenes | Meta250 failures | Failure rate | Chosen lambda histogram |
|---|---:|---:|---:|---|
| both_lambda40_and_lambda80_safe | 206 | 2 | 0.010 | `{"0.0": 13, "20.0": 32, "40.0": 80, "80.0": 81}` |
| lambda40_fails_lambda80_safe | 45 | 1 | 0.022 | `{"40.0": 1, "80.0": 44}` |
| both_lambda40_and_lambda80_fail | 49 | 49 | 1.000 | `{"0.0": 13, "20.0": 19, "40.0": 15, "80.0": 2}` |
| lambda40_safe_lambda80_fail | 0 | 0 | n/a | `{}` |
