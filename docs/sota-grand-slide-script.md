# SoTA Grand Commission Slide Script

## Slide 1: Minimal-Shot Autonomy, Not Memorized Autonomy

**Visual:** `grand_spotlight_demo/latest_rollout.svg`

This project asks what a small autonomy stack can do when it is forced to reason from structure instead of memorizing an AV dataset. The prototype combines a WOD-E2E parsing and submission harness with a closed-loop Spotlight Reflex policy for rare, long-tail driving cases.

## Slide 2: Why The Problem Matters

Most autonomy systems scale by collecting more data from places they already understand. That breaks down in rural roads, disaster zones, construction scenes, unusual agents, and rare hazards. Minimal-shot autonomy needs a world model, a candidate policy, and a safety selector that can act from sparse evidence.

## Slide 3: Architecture

**Visual:** show the architecture block from the README or explain over rollout SVG.

The stack is deliberately modular:

- procedural long-tail scene generator
- perception and world-state update
- maneuver library
- simulator-native trajectory selector
- uncertainty-aware safety filter
- WOD-E2E parser, validator, and official submission packager

The point is not a single monolithic imitation model. The point is a testable architecture where each failure can be attributed to perception, world state, candidate generation, selection, or safety.

## Slide 4: Spotlight Reflex Demo

**Visual:** `grand_spotlight_demo/latest_rollout.svg`

The Spotlight Reflex policy succeeds on a generated spotlight case with a low-visibility wrong-way/rare-hazard setup. The rollout is closed-loop: the policy replans at every step and reaches the goal without collision.

## Slide 5: Stress Case

**Visual:** `grand_intersection_stress_seed3/latest_rollout.svg`

The same policy handles an intersection stress case, showing that the behavior is not a one-off hand-authored scene. The scenario is generated from a cluster and seed, so it can be reproduced and varied.

## Slide 6: Understood Failure

**Visual:** `grand_baseline_spotlight_demo/latest_rollout.svg`

The baseline policy fails on the same Spotlight seed where Spotlight Reflex succeeds. This is included intentionally: the project is not claiming perfect autonomy. It shows that the architecture change matters and gives a concrete failure comparison.

## Slide 7: WOD-E2E Harness

The repository also includes the Waymo E2E path:

- official proto parsing
- validation preference-frame scoring
- official-format `E2EDChallengeSubmission` tarballs
- pre-registered blind submission matrix
- SHA-256 leaderboard result logging

The test split is not present in this workspace, so there is no hidden-test leaderboard claim yet. The code prevents unconfirmed leaderboard claims from passing the readiness audit.

## Slide 8: What Prize Funding Enables

Funding would turn this from a prototype into a stronger minimal-shot autonomy loop:

- obtain and run blind WOD-E2E test submissions
- add real camera grounding
- replace the lightweight world model with persistent causal scene memory
- expand closed-loop tests in AlpaSim
- mine failure cases automatically and improve the policy under realistic latency constraints

## Closing

The submission is strongest as an architecture prototype: it is honest about limitations, produces reproducible closed-loop evidence, and already has the infrastructure to separate validation experiments from hidden-test truth.
