# Recoverable-Regret Bootstrap Summary

| Variant | Policy | Score | Regret | PosReg | Fail | Progress | ADE3 |
|---|---|---:|---:|---:|---:|---:|---:|
| uniform_student | g0_proxy_top1 | -139.880±23.406 | 89.217±25.822 | 0.509 | 0.652 | 28.043 | 2.443 |
| uniform_student | g1_student_top1 | -114.316±40.172 | 63.653±20.666 | 0.481 | 0.511 | 22.772 | 4.619 |
| uniform_student | teacher_promoted_upper_bound | -50.663±43.276 | 0.000±0.000 | 0.000 | 0.259 | 22.954 | 4.432 |
| regret_weighted_student | g0_proxy_top1 | -139.880±23.406 | 89.217±25.822 | 0.509 | 0.652 | 28.043 | 2.443 |
| regret_weighted_student | g1_student_top1 | -110.320±45.542 | 59.657±23.503 | 0.640 | 0.473 | 19.051 | 5.586 |
| regret_weighted_student | teacher_promoted_upper_bound | -50.663±43.276 | 0.000±0.000 | 0.000 | 0.259 | 22.954 | 4.432 |

Interpretation: regret-weighted G1 improves mean score/regret and replay failure over G0, but remains far from teacher-promoted oracle@K; this is a surrogate for candidate-generator bootstrapping, not proof of regenerated candidate improvement.
