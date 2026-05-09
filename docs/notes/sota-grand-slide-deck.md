# Minimal-Shot Autonomy Architecture

Closed-loop long-tail reflexes, with a WOD-E2E harness as supporting benchmark infrastructure.

Repository: `https://github.com/amtellezfernandez/minimal-shot-av`

Tag: `sota-commission-2026-submission`

---

# 1. The Claim

This is a minimal-shot autonomy architecture prototype, not a memorized route follower or a hidden-test leaderboard claim.

The system combines:

- randomized long-tail simulation
- closed-loop Spotlight Reflex control
- WOD-E2E parsing, validation, and packaging as auxiliary infrastructure
- explicit readiness gates that prevent unconfirmed leaderboard claims

---

# 2. Judging Evidence

Technical excellence:

- Spotlight Reflex demos are closed-loop and reproducible by seed.
- The included randomized sweep covers 550 rollouts across 11 long-tail clusters.
- Numeric controller p95 runtime is `1.402 ms` against a 14 ms budget.
- WOD validation-CV evidence is included as audited development analysis, not as the zero-shot claim.

Novelty:

- Candidate generation is separated from trajectory selection.
- Oracle regret is reported instead of hidden.

Feasibility:

- 550 randomized closed-loop rollouts.
- 11 WOD-style long-tail clusters.
- 0 collisions and 0 near misses in the included sweep.

Adherence:

- Includes analysis notebook, randomized scenario generation, WOD-E2E validation-CV evidence, failure cases, and claim-boundary audits.

---

# 3. Why Minimal-Shot

Autonomy that depends on collecting every edge case will always lag reality.

The target is a system that can enter unfamiliar scenes and act from sparse structure:

- unusual roads
- construction and debris
- rare agents
- visibility changes
- off-distribution maneuvers

---

# 4. Architecture

The stack is deliberately modular:

- procedural long-tail scene generator
- perception and world-state update
- maneuver library
- simulator-native trajectory selector
- uncertainty-aware safety filter
- WOD-E2E parser, validator, and official submission packager

Failures are attributable to perception, world state, candidate generation, candidate selection, or safety filtering.

---

# 5. Spotlight Reflex Demo

Visual: `grand_spotlight_demo/latest_rollout.svg`

The Spotlight Reflex policy succeeds on a low-visibility spotlight hazard setup.

The rollout is closed-loop: the policy replans at each step and reaches the goal without collision.

---

# 6. Stress Case

Visual: `grand_intersection_stress_seed3/latest_rollout.svg`

The same policy handles an intersection stress case generated from a cluster and seed.

The point is reproducibility and variation, not a single hand-authored scene.

---

# 7. Understood Failure

Visual: `grand_baseline_spotlight_demo/latest_rollout.svg`

The baseline policy fails on the same Spotlight seed where Spotlight Reflex succeeds.

This failure case is included intentionally because the submission is an architecture prototype, not a perfect autonomy claim.

---

# 8. WOD-E2E Harness

The repository includes:

- official proto parsing
- validation preference-frame scoring
- official-format `E2EDChallengeSubmission` packaging
- submission matrix generation
- SHA-256 leaderboard result logging

Current promoted validation-CV result:

- selected RFS: `7.65941846208851`
- normalized RFS: `7.987543409079993`
- candidate oracle: `9.098014272661512`
- selector regret: `1.4385958105730023`
- reproducibility: exact rerun restored from commit
  `a934d2740cdf831c5f7ee58f5ddae3c7de8692e8` with
  `artifacts/cosmos_predict25_wan21_tokenizer_val479.json`

This selector is preference-calibrated on retained validation labels under
segment-grouped CV. It is development evidence, not strict zero-shot WOD-E2E.

No hidden-test leaderboard score is claimed.

---

# 9. Fast/Slow Runtime

The proposed fast/slow scheduler moves rich candidate refresh off the synchronous actuation path.

On the same 5,000-frame synthetic WOD-like runtime benchmark:

- monolithic synchronous p95: `2.124 ms`
- fast/slow synchronous p95: `0.625 ms`
- slow refresh p95: `2.132 ms`

---

# 10. Prize Funding

Funding would support:

- blind WOD-E2E test submissions
- stronger camera grounding
- persistent causal scene memory
- richer AlpaSim evaluation
- automatic failure mining
- more realistic latency-constrained policy iteration

---

# 11. Submission Boundary

Submit as:

Minimal-shot autonomy architecture prototype with reproducible closed-loop evidence and a rigorous WOD-E2E benchmark harness.

Do not submit as:

- a road-ready deployed AV system
- a solved zero-data WOD-E2E result
- a completed hidden-test leaderboard entry
