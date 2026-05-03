# Final Submission Checklist

## Submit For SoTA Grand Commission

- Upload `artifacts/sota_submission_bundles/grand_commission.tar.gz`.
- Verify SHA-256 against `artifacts/sota_submission_bundles/SUBMISSION_INDEX.md`.
- Use repository URL: `https://github.com/amtellezfernandez/minimal-shot-av`.
- Reference tag: `sota-commission-2026-submission`.
- Attach or link the slide deck generated from `docs/sota-grand-slide-deck.md`.
- Paste the form response from `docs/sota-grand-submission-form.md`.

Grand claim:

Minimal-shot autonomy architecture prototype with closed-loop long-tail demos, WOD-E2E validation-CV evidence, analysis notebook, runtime evidence, and readiness gates. No hidden-test WOD leaderboard score is claimed.

## Optional Minor Commission Submission

- Upload `artifacts/sota_submission_bundles/minor_commission.tar.gz`.
- Verify SHA-256 against `artifacts/sota_submission_bundles/SUBMISSION_INDEX.md`.
- Use the same repository URL and tag.
- Attach or link the slide deck generated from `docs/sota-minor-slide-script.md`.

Minor claim:

Randomized WOD-style long-tail simulation environment with 550 closed-loop rollouts across 11 scenario clusters.

## Do Not Claim

- Do not claim a completed Waymo WOD-E2E hidden-test leaderboard result.
- Do not claim a road-ready deployed AV system.
- Do not claim solved zero-data WOD-E2E performance.

Current blockers for the separate Waymo leaderboard track:

- Missing local WOD-E2E test split.
- Missing confirmed hidden-test result log.

## Final Local Verification

These checks passed for the final submission bundle:

- `UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python scripts/check_code_quality.py`
- `UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python scripts/audit_sota_judging_criteria.py`
- `UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python scripts/audit_sota_submission_bundles.py --bundle-root artifacts/sota_submission_bundles`
- `UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python -m unittest tests.test_audit_wod_validation_breakthrough tests.test_run_wod_breakthrough_experiments tests.test_audit_sota_judging_criteria tests.test_run_submission_demos tests.test_audit_sota_submission_bundles tests.test_submission_claim_language`
