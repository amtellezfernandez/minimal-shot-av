# Submission Tracks

This project has three separate submission tracks. They should not be merged.

## 1. SoTA Grand Commission

Purpose: autonomy architecture.

Submit:

- `artifacts/sota_submission_bundles/grand_commission.tar.gz`
- GitHub repository URL
- Grand slide/video using `docs/sota-grand-slide-script.md`
- Grand form copy using `docs/sota-grand-submission-form.md`

Claim:

Closed-loop minimal-shot autonomy architecture prototype with WOD-E2E harness as supporting infrastructure.

Do not claim:

- WOD-E2E hidden-test leaderboard performance
- no production AV readiness claim
- solved world-model autonomy

## 2. SoTA Minor Commission

Purpose: simulation environment.

Submit:

- `artifacts/sota_submission_bundles/minor_commission.tar.gz`
- GitHub repository URL
- Minor slide/video using `docs/sota-minor-slide-script.md`
- Minor form copy using `docs/sota-minor-submission-form.md`

Claim:

Randomized long-tail simulator with WOD-E2E-style cluster coverage,
compositional OOD suites, AlpaSignal bridge evidence, and reproducible
closed-loop evaluation.

Do not claim:

- photorealistic sensor simulation
- full AlpaSim perception integration
- WOD-E2E leaderboard result

## 3. Waymo WOD-E2E Challenge

Purpose: benchmark / leaderboard submission.

Submit:

- official `E2EDChallengeSubmission` tarball generated from WOD-E2E test frames
- result log recorded with `scripts/record_wod_leaderboard_result.py`

Current blocker:

- local WOD-E2E `test` split is missing
- no confirmed hidden-test leaderboard result is recorded

Claim only after upload:

- hidden-test RFS, rank, and SHA-256 from `artifacts/wod_e2e_leaderboard_results.json`

## Rule

SoTA Grand and Minor can be submitted now as prototype/evidence submissions. Waymo is separate and remains blocked until the official test split and authenticated upload path exist.
