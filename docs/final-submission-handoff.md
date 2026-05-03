# Final SoTA Submission Handoff

Use this as the source of truth when completing the SoTA Commission form.

## Repository

- GitHub URL: `https://github.com/amtellezfernandez/minimal-shot-av`
- Submitted commit: `e56dc3b623ecea7f138cbdb032de6be862e7914a`
- Branch: `main`

## Grand Commission

Track: Grand Commission, overall best autonomy architecture.

Title:

Minimal-Shot Autonomy Architecture: Closed-Loop Long-Tail Reflexes with WOD-E2E Submission Harness

Attach:

- `artifacts/sota_submission_bundles/grand_commission.tar.gz`
- SHA-256: `b0390a1449ac63d87836b3a3a54f078d063df13452ff1b9e63c9cf196d846b16`
- Video or slide deck based on `docs/sota-grand-slide-script.md`

Short claim:

This is a runnable minimal-shot autonomy architecture prototype with a WOD-E2E parsing/scoring/submission harness and closed-loop Spotlight Reflex demos on generated long-tail scenarios.

Evidence to mention:

- Spotlight Reflex success demo.
- Intersection stress demo.
- Baseline Spotlight failure comparison.
- WOD-E2E official-format parsing, scoring, packaging, and readiness audits.
- Official-RFS validation-CV candidate evidence is included in the bundle.

WOD validation numbers:

- 479 validation frames with valid rater preference labels.
- Constant velocity official validation-CV RFS: `7.022`.
- Combined candidate ranker official validation-CV RFS: `7.657`.
- Gain versus constant velocity: `+0.635`.
- Combined candidate oracle official validation-CV RFS: `9.098`.
- Selector regret to oracle: `1.441`.
- Breakthrough audit status: failed because worst-slice regret regressed.

Required caveat:

Do not claim production autonomy, strict zero-shot WOD-E2E, or a hidden-test leaderboard result. The local workspace does not contain the WOD-E2E test split or confirmed leaderboard result log.

## Minor Commission

Track: Minor Commission, overall best simulation environment.

Title:

Randomized Long-Tail Scenario Generator for Minimal-Shot Autonomy

Attach:

- `artifacts/sota_submission_bundles/minor_commission.tar.gz`
- SHA-256: `013e96fca7d0cc14524218e24721d1bc8c3b3ed6bf78eb0506c66748caa78f4d`
- Video or slide deck based on `docs/sota-minor-slide-script.md`

Short claim:

This is a seeded procedural simulation environment for WOD-E2E-style long-tail driving cases, with reproducible scenario generation and closed-loop evaluation artifacts.

Evidence to mention:

- Construction scenario demo.
- Foreign-object-debris scenario demo.
- Spotlight scenario demo.
- Multi-cluster evaluation sweep with JSON and CSV reports.
- SVG and JSON artifacts for scenario inspection.

Bundled sweep:

- 550 closed-loop rollouts.
- 11 WOD-E2E-style scenario clusters.
- 50 seeds per cluster.
- 550 / 550 successful rollouts.
- 550 / 550 benchmark passes.
- 0 collisions.
- 0 safe stalls.
- 0 near misses under the benchmark diagnostic.
- Mean minimum clearance: `2.80 m`.
- Mean 5th-percentile clearance: `3.28 m`.
- Mean intervention rate: `9.44%`.

Required caveat:

This is a lightweight 2D procedural simulator, not a photorealistic sensor simulator. AlpaSim integration exists at trajectory/plugin level; full sensor-realistic perception integration is future work.

## Submission Email

Subject:

SoTA Commission I Submission: Minimal-Shot AV Architecture + Long-Tail Scenario Generator

Body:

Dear SoTA Commission team,

Please find my submissions for SoTA Commission I:

1. Grand Commission: Minimal-Shot Autonomy Architecture: Closed-Loop Long-Tail Reflexes with WOD-E2E Submission Harness
2. Minor Commission: Randomized Long-Tail Scenario Generator for Minimal-Shot Autonomy

GitHub repository:

`https://github.com/amtellezfernandez/minimal-shot-av`

Submitted commit:

`e56dc3b623ecea7f138cbdb032de6be862e7914a`

Attached bundles:

- `grand_commission.tar.gz`
  SHA-256: `b0390a1449ac63d87836b3a3a54f078d063df13452ff1b9e63c9cf196d846b16`
- `minor_commission.tar.gz`
  SHA-256: `013e96fca7d0cc14524218e24721d1bc8c3b3ed6bf78eb0506c66748caa78f4d`

The Grand submission includes a WOD-E2E harness and validation-CV evidence, but does not claim a hidden-test leaderboard result. The Minor submission includes the randomized scenario generator and 550-run closed-loop evaluation sweep.

Best,

Alba Maria Tellez Fernandez
