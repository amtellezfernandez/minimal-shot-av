# Meta Fail-Penalty Sweep

This sweep diagnoses why the exploratory meta-value policy previously failed to beat fixed
`lambda40`: the meta objective used a failure penalty that was too small relative to
progress and ADE.

Meta objective:

```text
-fail_penalty * replay_failure + 1.0 * progress - 2.0 * ADE3
```

| fail penalty | meta replay rate | lambda40 replay rate | progress | ADE3 | recovery | train target implication |
|---:|---:|---:|---:|---:|---:|---|
| 30 | 0.435 | 0.404 | 24.433 | 3.334 | 0.552 | too utility-heavy; underperforms lambda40 on replay risk |
| 60 | 0.401 | 0.404 | 23.898 | 3.554 | 0.662 | approximately matches lambda40 |
| 100 | 0.398 | 0.404 | 23.024 | 3.883 | 0.667 | slightly better than lambda40 |
| 150 | 0.378 | 0.404 | 22.370 | 4.130 | 0.705 | safety-dominant enough to beat lambda40 |
| 250 | 0.368 | 0.404 | 22.057 | 4.255 | 0.730 | corrected exploratory setting |
| 500 | 0.367 | 0.404 | 21.865 | 4.330 | 0.735 | marginal replay gain over 250, more utility loss |

Conclusion: the previous non-overperformance was not evidence that meta-calibration is
intrinsically bad. It was an objective-scaling error. With the corrected safety-dominant
meta objective, `meta_lambda_value` beats fixed `lambda40` on replay-infeasible rate while
paying a progress/ADE cost.
