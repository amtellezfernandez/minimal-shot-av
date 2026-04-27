# SoTA Grand Commission Form Draft

## Title

Minimal-Shot Autonomy Architecture: Closed-Loop Long-Tail Reflexes with WOD-E2E Submission Harness

## Track

Grand Commission: overall best autonomy architecture.

## Short Description

This project explores a minimal-shot autonomy architecture that separates world understanding, trajectory generation, and safety selection instead of relying on memorized maps or AV-dataset fine-tuning. The current prototype combines a WOD-E2E parsing/submission harness with a closed-loop Spotlight Reflex policy tested on randomized long-tail driving scenarios. It is deliberately honest about what is solved and what is not: the WOD-E2E leaderboard path is ready for blind submission once test data is available, while the simulator demonstrates the architecture under reproducible closed-loop stress cases.

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
- Full test suite output: `UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python scripts/run_tests.py`.

## What Worked

The strongest result is the closed-loop architecture behavior: the policy uses scenario structure and safety selection to navigate long-tail generated cases that defeat a simpler baseline. The second strongest result is infrastructure: official WOD-E2E parsing, scoring, packaging, and leaderboard-result logging are now separated from validation-tuned claims.

## What Did Not Work Yet

The lightweight WOD-E2E world-model experiment is not yet a solved autonomy model. It produced only a small official validation-CV selected RFS movement and should be treated as early evidence, not a final solution. Hidden-test leaderboard results are still missing because the local workspace does not contain the WOD-E2E test split.

## Next Use Of Prize Funding

Prize funding would go toward turning the prototype into a real minimal-shot evaluation loop: acquiring compute and dataset access for blind WOD-E2E submissions, integrating stronger vision grounding, expanding the world model beyond trajectory summaries, and running richer closed-loop tests in AlpaSim or a comparable simulator.
