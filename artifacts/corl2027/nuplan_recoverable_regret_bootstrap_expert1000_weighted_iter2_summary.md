# Recoverable-Regret Bootstrap Iteration-2 Summary

| Iter | Policy | Score | Regret | Fail | Progress | ADE3 |
|---:|---|---:|---:|---:|---:|---:|
| 1 | g0_proxy_top1 | -139.880±23.406 | 89.217±25.822 | 0.652 | 28.043 | 2.443 |
| 1 | g1_student_top1 | -110.320±45.542 | 59.657±23.503 | 0.473 | 19.051 | 5.586 |
| 1 | teacher_promoted_upper_bound | -50.663±43.276 | 0.000±0.000 | 0.259 | 22.954 | 4.432 |
| 2 | g0_proxy_top1 | -110.320±45.542 | 59.657±23.503 | 0.473 | 19.051 | 5.586 |
| 2 | g1_student_top1 | -117.243±36.281 | 66.580±23.856 | 0.531 | 23.851 | 4.230 |
| 2 | teacher_promoted_upper_bound | -50.663±43.276 | 0.000±0.000 | 0.259 | 22.954 | 4.432 |

Interpretation: iteration 1 closes part of the recoverable-regret gap; iteration 2 does not reliably improve mean score/regret with fixed ManeuverToken candidates, indicating that the next research step must regenerate the candidate distribution rather than repeatedly distill the same fixed candidate set.
