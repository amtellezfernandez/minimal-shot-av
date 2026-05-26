# Boundary-Consistency Audit

- Input JSON: `artifacts/corl2027/replay_value_repeated_splits_summary.json`
- Split count: `5`
- Oracle-gap tolerance: `0.050`

## Summary

- Scene-token improved over proxy in `3/5` splits.
- Safety-heavy replay value stayed within oracle tolerance in `5/5` splits and matched exactly in `3/5`.
- Balanced replay value stayed within oracle tolerance in `3/5` splits and matched exactly in `1/5`.
- Balanced replay value improved progress over the safety-heavy point in `5/5` splits.
- Mean oracle gap: safety-heavy `0.010 ± 0.013`, balanced `0.081 ± 0.091`.
- Mean balanced progress gain over safety-heavy: `3.655 ± 1.719 m`.

| Seed | Scene-token fail improvement | Safety-heavy oracle gap | Balanced oracle gap | Balanced progress gain vs safety-heavy (m) |
|---|---:|---:|---:|---:|
| 0 | 0.023 | 0.000 | 0.000 | 1.624 |
| 1 | -0.039 | 0.032 | 0.039 | 2.465 |
| 2 | -0.003 | 0.000 | 0.067 | 5.352 |
| 3 | 0.060 | 0.017 | 0.040 | 2.806 |
| 4 | 0.019 | 0.000 | 0.258 | 6.029 |
