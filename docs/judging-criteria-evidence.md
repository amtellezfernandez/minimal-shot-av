# Judging Criteria Evidence

This is the direct evidence map for SoTA Commission I judging. It is generated
from artifacts that are bundled with the submission and audited by
`scripts/audit_sota_judging_criteria.py`.

## Technical Excellence

- WOD-E2E official validation-CV selected RFS improves from `7.022` constant
  velocity to `7.659`, a `+0.637` gain on 479 preference-labeled validation
  frames.
- The combined candidate oracle reaches `9.098` RFS, leaving `1.439` RFS of
  selector headroom. This makes the remaining bottleneck measurable rather than
  anecdotal.
- The simulator evidence contains 550 Spotlight Reflex closed-loop rollouts
  across 11 WOD-style long-tail clusters, with 50 seeds per cluster.
- The numeric WOD controller runtime report beats the 14 ms p95 latency budget
  with `1.402` ms p95 total latency.

## Novelty

- The architecture separates proposal diversity from trajectory selection:
  kinematic, learned residual, temporal, and scene candidates are evaluated
  independently from the selector.
- The submission includes a conservative improvement audit. A small scene-gate
  sweep improves mean RFS and normalized RFS while reducing worst-slice regret.
- The active runtime path is structured and non-text. It does not depend on
  prompt parsing or memorized AV demonstrations.

## Feasibility

- The simulation sweep reports 0 collisions and 0 near misses under the current
  benchmark diagnostic.
- Mean minimum clearance across the 11-cluster sweep is about `2.80` m, while
  mean intervention rate is about `0.094`.
- WOD hidden-test and production claims remain gated by explicit readiness
  audits. The current claim is a submission-candidate architecture, not a
  production AV stack.

## Adherence To The Brief

- The repo includes the requested analysis notebook:
  `notebooks/wod_e2e_analysis.ipynb`.
- The simulator track provides randomized scenario generation across the 11
  long-tail WOD-style clusters.
- The WOD-E2E path includes parser, scorer, candidate-generation, validation-CV
  analysis, packaging, and readiness audits.
- The submission is honest about its current boundary: validation-CV evidence is
  not presented as a hidden-test leaderboard result.
