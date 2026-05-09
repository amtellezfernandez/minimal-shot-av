# SoTA Grand Commission Slide Script

## Slide 1: Minimal-Shot Autonomy, Not Memorized Autonomy

**Visual:** `grand_spotlight_demo/latest_rollout.svg`

This project asks what a small autonomy stack can do when it is forced to reason
from structure instead of memorizing an AV dataset. The prototype centers on a
closed-loop Spotlight Reflex policy for rare, long-tail driving cases. A WOD-E2E
parsing and submission harness is included as benchmark infrastructure, not as
the zero-shot autonomy claim.

## Slide 2: Judging Scorecard

**Visual:** `docs/judging-criteria-evidence.md` or the criteria audit JSON.

The submission now has a direct evidence map for the four judging criteria:

- Technical excellence: the Spotlight Reflex demos are closed-loop,
  reproducible by seed, and backed by a 550-rollout randomized sweep; the
  numeric controller runtime is `1.402` ms p95 against a 14 ms budget.
- Novelty: the architecture separates candidate diversity from trajectory
  selection, and reports oracle regret rather than hiding selector failure.
- Feasibility: the randomized simulator sweep covers 550 rollouts across 11
  WOD-style long-tail clusters with 0 collisions and 0 near misses.
- Adherence: the bundle includes the analysis notebook, simulation evidence,
  WOD-E2E validation-CV reports, failure audit, and readiness gates.

Secondary WOD note: the benchmark harness includes a conservative `7.6594`
official validation-CV report and a `9.098` candidate oracle. That selector is
preference-calibrated on retained validation labels, so it is development
analysis rather than proof of strict zero-shot WOD-E2E generalisation.

Realtime note: the proposed fast/slow scheduler drops synchronous WOD runtime
p95 from `2.124 ms` to `0.625 ms` on the same 5,000-frame synthetic runtime
benchmark by moving rich candidate refresh off the actuation path.

## Slide 3: Why The Problem Matters

Most autonomy systems scale by collecting more data from places they already understand. That breaks down in rural roads, disaster zones, construction scenes, unusual agents, and rare hazards. Minimal-shot autonomy needs a world model, a candidate policy, and a safety selector that can act from sparse evidence.

## Slide 4: Architecture

**Visual:** show the architecture block from the README or explain over rollout SVG.

The stack is deliberately modular:

- procedural long-tail scene generator
- perception and world-state update
- maneuver library
- simulator-native trajectory selector
- uncertainty-aware safety filter
- WOD-E2E parser, validator, and official submission packager

The point is not a single monolithic imitation model. The point is a testable architecture where each failure can be attributed to perception, world state, candidate generation, selection, or safety.

## Slide 5: Spotlight Reflex Demo

**Visual:** `grand_spotlight_demo/latest_rollout.svg`

The Spotlight Reflex policy succeeds on a generated spotlight case with a low-visibility wrong-way/rare-hazard setup. The rollout is closed-loop: the policy replans at every step and reaches the goal without collision.

## Slide 6: Stress Case

**Visual:** `grand_intersection_stress_seed3/latest_rollout.svg`

The same policy handles an intersection stress case, showing that the behavior is not a one-off hand-authored scene. The scenario is generated from a cluster and seed, so it can be reproduced and varied.

## Slide 7: Understood Failure

**Visual:** `grand_baseline_spotlight_demo/latest_rollout.svg`

The baseline policy fails on the same Spotlight seed where Spotlight Reflex succeeds. This is included intentionally: the project is not claiming perfect autonomy. It shows that the architecture change matters and gives a concrete failure comparison.

## Slide 8: WOD-E2E Harness

The repository also includes the Waymo E2E path:

- official proto parsing
- validation preference-frame scoring
- official-format `E2EDChallengeSubmission` tarballs
- pre-registered blind submission matrix
- SHA-256 leaderboard result logging

The retained-validation selector uses rater preference labels under
segment-grouped CV, so it is not strict zero-shot. The test split is not present
in this workspace, so there is no hidden-test leaderboard claim yet. The code
prevents unconfirmed leaderboard claims from passing the readiness audit.

## Slide 9: What Prize Funding Enables

Funding would turn this from a prototype into a stronger minimal-shot autonomy loop:

- obtain and run blind WOD-E2E test submissions
- add real camera grounding
- replace the lightweight world model with persistent causal scene memory
- expand closed-loop tests in AlpaSim
- mine failure cases automatically and improve the policy under realistic latency constraints

## Closing

The submission is strongest as an architecture prototype: it is honest about limitations, produces reproducible closed-loop evidence, and already has the infrastructure to separate validation experiments from hidden-test truth.
