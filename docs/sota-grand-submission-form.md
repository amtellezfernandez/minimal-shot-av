# SoTA Grand Commission Form Draft

## Title

Minimal-Shot Autonomy Architecture: Closed-Loop Long-Tail Reflexes

## Track

Grand Commission: overall best autonomy architecture.

## Short Description

This project explores a minimal-shot autonomy architecture that separates world
understanding, trajectory generation, and safety selection instead of relying on
memorized maps or validation tuning. The current prototype is a closed-loop
Spotlight Reflex policy tested on randomized long-tail driving scenarios. A
WOD-E2E parsing/submission harness is included, but it is benchmark
infrastructure and analysis support, not the core zero-shot claim. The strongest
evidence is closed-loop architecture behavior, failure analysis, and tooling;
the WOD-E2E leaderboard path still needs local test data and a confirmed
hidden-test result.

## Motivation

Autonomy that depends on collecting every edge case will always lag reality. The important capability is not memorizing a city or a dataset; it is entering unfamiliar scenes and making safe decisions from sparse evidence. This prototype targets that gap by emphasizing generalizable scene structure, deterministic fallback behavior, and explicit safety gates over opaque validation tuning.

## Technical Approach

The system has three layers:

1. A closed-loop simulator stack that generates WOD-style rare scenarios and
   runs a deterministic Spotlight Reflex policy through repeated perception,
   world-state update, maneuver generation, trajectory selection, and safety
   filtering.
2. A minimal-shot autonomy policy that uses procedural scene evidence,
   counterfactual maneuver candidates, and conservative safety selection without
   AV-dataset fine-tuning.
3. A WOD-E2E model-side harness that parses official Waymo E2E frames,
   evaluates candidate trajectories against rater-feedback references where
   available, packages official-format submission tarballs, and records
   leaderboard results by exact SHA-256.
4. A submission/readiness protocol that separates validation evidence from
   hidden-test truth and rejects unconfirmed leaderboard claims.

The policy does not claim to be a production AV stack. It is a minimal-shot
architecture prototype designed to show how a small system can reason through
rare hazards without AV-dataset fine-tuning.

## Evidence Included

- Spotlight Reflex success demo on a generated spotlight scenario.
- Intersection stress demo.
- Baseline Spotlight failure comparison.
- WOD-E2E official-format submission machinery, presented as infrastructure.
- Readiness audit for hidden-test leaderboard truth.
- Novel-object stress audit: `artifacts/novel_object_stress_audit.json`.
- Judging criteria audit: `artifacts/sota_judging_criteria_audit.json`.
- Full test suite output: `UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python scripts/run_tests.py`.

Judging criteria evidence:

- technical excellence: the closed-loop Spotlight Reflex demos are reproducible,
  the WOD validation-CV harness is audited, and numeric controller p95 latency
  is 1.402 ms against a 14 ms budget.
- novelty: candidate generation and candidate selection are separated, and the
  repo reports selector regret explicitly instead of only reporting the best
  oracle candidate.
- feasibility: randomized closed-loop simulation covers 550 rollouts across 11
  WOD-style long-tail clusters with 0 collisions and 0 near misses under the
  benchmark diagnostic.
- unknown-object evidence: an abstract novel-object stress audit covers 81
  compositional cases across `fallen_sign`, `horse_trailer`, `parade_float`,
  `piano`, and `portable_toilet` labels with 0 collisions, 0 near misses, and
  100% goal completion. The default policy scores against current obstacle
  geometry, disables hidden simulator actor-behavior forecasting, and the active
  policy audit rejects scenario manifest or cluster-name lookup. This is
  geometry-level obstacle handling in the 2D simulator, not camera-based
  recognition of physical unknown objects.
- adherence to the brief: the bundle includes the analysis notebook, randomized
  scenario generation, WOD-E2E validation-CV evidence, and a documented failure
  case.

Current WOD-E2E validation evidence, declared as development analysis:

- 479 validation frames with valid rater preference labels.
- constant-velocity official validation-CV RFS: 7.022.
- combined candidate ranker official validation-CV RFS: 7.6594.
- gain versus constant velocity: +0.637 RFS.
- combined candidate oracle official validation-CV RFS: 9.098.
- selector regret to oracle: 1.439 RFS.
- conservative improvement audit status: passed versus the previous `7.659198`
  validation-CV report.
- minimal-shot caveat: this WOD selector is preference-calibrated on retained
  validation labels under segment-grouped CV. It is not strict zero-shot and is
  not the centerpiece of the Grand claim.
- no hidden-test leaderboard score is claimed.

## What Worked

The strongest result is the closed-loop architecture behavior: the policy uses
scenario structure and safety selection to navigate long-tail generated cases
that defeat a simpler baseline. The second strongest result is infrastructure:
official WOD-E2E parsing, scoring, packaging, and leaderboard-result logging are
now separated from validation-tuned claims.

The validation-CV candidate stack shows useful but limited benchmark signal. The
combined ranker improves over constant velocity, and the candidate oracle is
substantially stronger than the selected trajectory. This is evidence that the
harness can diagnose candidate routing bottlenecks, not evidence that the system
has solved zero-shot WOD-E2E.

For realtime behavior, I added a fast/slow controller proposal and benchmark.
The fast reflex loop keeps only compact local candidates plus one cached slow
plan on the synchronous path, while the slow loop refreshes richer candidates
asynchronously. On the same 5,000-frame synthetic WOD-like runtime benchmark,
the synchronous p95 latency drops from `2.124 ms` monolithic to `0.625 ms`
fast/slow; slow refresh p95 is reported separately at `2.132 ms`.

Preference-calibrated heldout experiments did reach the `7.8+` zone under a
local scorer: `7.838` selected local RFS on 159 heldout validation-preference
frames. The strongest official-scored neural heldout subset run reached
`7.737680847131364` mean RFS on 159 frames, and a separate heldout variant
reached `7.894` normalized RFS but lower mean RFS. The full retained-validation
official rerun scored `7.262`, so this is secondary evidence, not the promoted
WOD claim.

The criteria audit passes all four SoTA judging dimensions under the declared
submission-candidate boundary. That does not make the system a hidden-test
leaderboard winner; it means the repo now has concrete evidence for why the
architecture is technically serious, novel, feasible, and aligned with the
commission brief.

## What Did Not Work Yet

The lightweight WOD-E2E world-model experiment is not yet a solved autonomy model. It produced only a small official validation-CV selected RFS movement and should be treated as early evidence, not a final solution. Hidden-test leaderboard results are still missing because the local workspace does not contain the WOD-E2E test split.

The promoted WOD validation report has a small validation-CV margin and depends
on retained validation preference labels. It should be presented as transparent
development evidence rather than a solved model, strict minimal-shot result, or
hidden-test leaderboard result.

This should not be described as production autonomy, strict zero-shot WOD-E2E, or a completed leaderboard entry. The honest Grand claim is a runnable minimal-shot architecture prototype plus a rigorous submission harness.

## Next Use Of Prize Funding

Prize funding would go toward turning the prototype into a real minimal-shot evaluation loop: acquiring compute and dataset access for blind WOD-E2E submissions, integrating stronger vision grounding, expanding the world model beyond trajectory summaries, and running richer closed-loop tests in AlpaSim or a comparable simulator.
