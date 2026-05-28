# nuPlan Bootstrap Candidate Loop

- Input replay JSON: `artifacts/corl2027/nuplan_public_replay_study_interaction1000_expert/replay.json`
- Teacher targets: `artifacts/corl2027/nuplan_bootstrap_lambda80_teacher_targets_expert1000.jsonl`
- Method: `teacher_promoted_g1_surrogate`
- Claim boundary: Promotes an existing teacher-selected candidate to G1 top-1. It measures selection-gap closure, not new-candidate generation.

## G1 Application

| Metric | Value |
|---|---:|
| scenes | 1000 |
| targets | 1000 |
| applied | 1000 |
| changed top-1 token | 554 |
| missing target | 0 |
| missing candidate | 0 |

## Gap Movement

| Gap | G0 | G1 | Delta |
|---|---:|---:|---:|
| oracle@K safe | 719 | 719 | +0 |
| generation gap | 281 | 281 | +0 |
| top-1 safe | 373 | 706 | +333 |
| selection gap | 346 | 13 | -333 |

## Selector Metrics

| Selector | Replay-infeasible | Rate | Recovery | Progress | ADE3 | Entropy |
|---|---:|---:|---:|---:|---:|---:|
| g0_proxy_top1 | 627 | 0.627 | 0/346 | 27.962 | 2.472 | 2.375 |
| g1_teacher_promoted_top1 | 294 | 0.294 | 333/346 | 20.844 | 4.741 | 2.890 |
| replay_oracle | 281 | 0.281 | 346/346 | 12.534 | 7.937 | 2.694 |

## Teacher Supervision Burden

| Metric | Value |
|---|---:|
| changed top-1 tokens | 554 |
| mean progress delta | -12.847 |
| mean ADE3 delta | 4.097 |
| mean final-pose shift | 13.084 |

