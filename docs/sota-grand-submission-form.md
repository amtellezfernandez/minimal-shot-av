# SoTA Grand Commission Form Draft

## Title

Minimal-Shot Autonomy Architecture: Closed-Loop Long-Tail Reflexes with WOD-E2E Submission Harness

## Track

Grand Commission: overall best autonomy architecture.

## Short Description

This project explores a minimal-shot autonomy architecture that separates world understanding, trajectory generation, and safety selection instead of relying on memorized maps or hidden validation tuning. The current prototype combines a WOD-E2E parsing/submission harness with a closed-loop Spotlight Reflex policy tested on randomized long-tail driving scenarios. It is deliberately honest about what is solved and what is not: the strongest evidence is closed-loop architecture behavior and tooling, while the WOD-E2E leaderboard path still needs local test data and a confirmed hidden-test result.

## Motivation

Autonomy that depends on collecting every edge case will always lag reality. The important capability is not memorizing a city or a dataset; it is entering unfamiliar scenes and making safe decisions from sparse evidence. This prototype targets that gap by emphasizing generalizable scene structure, deterministic fallback behavior, and explicit safety gates over opaque validation tuning.

## Technical Approach

The system has three layers:

1. A WOD-E2E model-side harness that parses official Waymo E2E frames, evaluates candidate trajectories against rater-feedback references where available, packages official-format submission tarballs, and records leaderboard results by exact SHA-256.
2. A closed-loop simulator stack that generates WOD-style rare scenarios and runs a deterministic Spotlight Reflex policy through repeated perception, world-state update, maneuver generation, trajectory selection, and safety filtering.
3. A submission/readiness protocol that separates validation evidence from hidden-test truth and rejects unconfirmed leaderboard claims.

The policy does not claim to be a production AV stack. It is a minimal-shot architecture prototype designed to show how a small system can reason through rare hazards without AV-dataset fine-tuning.

## Evidence Included

- Spotlight Reflex success demo on a generated spotlight scenario.
- Intersection stress demo.
- Baseline Spotlight failure comparison.
- WOD-E2E official-format submission machinery.
- Readiness audit for hidden-test leaderboard truth.
- Judging criteria audit: `artifacts/sota_judging_criteria_audit.json`.
- Full test suite output: `UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python scripts/run_tests.py`.

Judging criteria evidence:

- technical excellence: WOD validation-CV selected RFS improves by +0.637 over
  constant velocity, the candidate oracle reaches 9.098 RFS, and numeric
  controller p95 latency is 1.402 ms against a 14 ms budget.
- novelty: candidate generation and candidate selection are separated, and the
  repo reports selector regret explicitly instead of only reporting the best
  oracle candidate.
- feasibility: randomized closed-loop simulation covers 550 rollouts across 11
  WOD-style long-tail clusters with 0 collisions and 0 near misses under the
  benchmark diagnostic.
- adherence to the brief: the bundle includes the analysis notebook, randomized
  scenario generation, WOD-E2E validation-CV evidence, and a documented failure
  case.

Current WOD-E2E validation evidence:

- 479 validation frames with valid rater preference labels.
- constant-velocity official validation-CV RFS: 7.022.
- combined candidate ranker official validation-CV RFS: 7.6594.
- gain versus constant velocity: +0.637 RFS.
- combined candidate oracle official validation-CV RFS: 9.098.
- selector regret to oracle: 1.439 RFS.
- conservative improvement audit status: passed versus the previous `7.659198`
  validation-CV report.
- no hidden-test leaderboard score is claimed.

## What Worked

The strongest result is the closed-loop architecture behavior: the policy uses scenario structure and safety selection to navigate long-tail generated cases that defeat a simpler baseline. The second strongest result is infrastructure: official WOD-E2E parsing, scoring, packaging, and leaderboard-result logging are now separated from validation-tuned claims.

The validation-CV candidate stack also shows real but limited signal. The combined ranker improves over constant velocity, and the candidate oracle is substantially stronger than the selected trajectory. Conservative scene-gate sweeps improved the official validation-CV report from `7.657` to `7.6594` while reducing selector regret from `1.441` to `1.4386`; the remaining gap points to the next technical bottleneck: candidate routing, not just generating more trajectories.

For realtime behavior, I added a fast/slow controller proposal and benchmark.
The fast reflex loop keeps only compact local candidates plus one cached slow
plan on the synchronous path, while the slow loop refreshes richer candidates
asynchronously. On the same 5,000-frame synthetic WOD-like runtime benchmark,
the synchronous p95 latency drops from `2.124 ms` monolithic to `0.625 ms`
fast/slow; slow refresh p95 is reported separately at `2.132 ms`.

Preference-calibrated heldout experiments did reach the `7.8+` zone under a
local scorer: `7.838` selected local RFS on 159 heldout validation-preference
frames. An official-scored heldout variant reached `7.894` normalized RFS but
only `7.628` official mean RFS. The full retained-validation official rerun
scored `7.262`, so this is secondary evidence, not the promoted WOD claim.

The criteria audit passes all four SoTA judging dimensions under the declared
submission-candidate boundary. That does not make the system a hidden-test
leaderboard winner; it means the repo now has concrete evidence for why the
architecture is technically serious, novel, feasible, and aligned with the
commission brief.

## What Did Not Work Yet

The lightweight WOD-E2E world-model experiment is not yet a solved autonomy model. It produced only a small official validation-CV selected RFS movement and should be treated as early evidence, not a final solution. Hidden-test leaderboard results are still missing because the local workspace does not contain the WOD-E2E test split.

The promoted WOD validation report passes the repo's conservative improvement audit, but the margin is small and remains validation-CV only. This is why the WOD evidence should be presented as a transparent development result rather than a solved model or hidden-test leaderboard result.

This should not be described as production autonomy, strict zero-shot WOD-E2E, or a completed leaderboard entry. The honest Grand claim is a runnable minimal-shot architecture prototype plus a rigorous submission harness.

## Next Use Of Prize Funding

Prize funding would go toward turning the prototype into a real minimal-shot evaluation loop: acquiring compute and dataset access for blind WOD-E2E submissions, integrating stronger vision grounding, expanding the world model beyond trajectory summaries, and running richer closed-loop tests in AlpaSim or a comparable simulator.
