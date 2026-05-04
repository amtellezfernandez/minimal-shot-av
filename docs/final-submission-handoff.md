# Final SoTA Submission Handoff

Use this as the source of truth when completing the SoTA Commission form.

## Repository

- GitHub URL: `https://github.com/amtellezfernandez/minimal-shot-av`
- Submitted revision: the commit pointed to by tag
  `sota-commission-2026-submission`
- Branch: `main`

## Grand Commission

Track: Grand Commission, overall best autonomy architecture.

Title:

Minimal-Shot Autonomy Architecture: Closed-Loop Long-Tail Reflexes

Attach:

- `artifacts/sota_submission_bundles/grand_commission.tar.gz`
- SHA-256: see `artifacts/sota_submission_bundles/submission_bundles_manifest.json`
- Video or slide deck based on `docs/sota-grand-slide-script.md`
- Analysis notebook: `notebooks/wod_e2e_analysis.ipynb`
- Judging evidence audit: `artifacts/sota_judging_criteria_audit.json`
- Final readiness audit: `artifacts/final_submission_readiness_audit.json`

Short claim:

This is a runnable minimal-shot autonomy architecture prototype centered on
closed-loop Spotlight Reflex demos in generated long-tail scenarios. The WOD-E2E
code is included as a parsing/scoring/submission harness and analysis record,
not as the primary zero-shot autonomy claim.

Evidence to mention:

- Spotlight Reflex success demo.
- Intersection stress demo.
- Baseline Spotlight failure comparison.
- WOD-E2E official-format parsing, scoring, packaging, and readiness audits,
  clearly labelled as benchmark infrastructure.
- Official-RFS validation-CV candidate evidence is included in the bundle as
  development analysis, not as proof of strict zero-shot generalisation.
- Novel-object stress audit: 81 abstract compositional unknown-object cases,
  100% goal completion, 0 collisions, 0 near misses, and all five synthetic
  labels covered. The default policy disables hidden simulator actor-behavior
  forecasting and the minimal-shot audit rejects scenario manifest or
  cluster-name lookup, so this is current-geometry simulator evidence rather
  than camera recognition or privileged simulator-oracle control.
- Judging criteria audit passes for technical excellence, novelty, feasibility,
  and adherence to the brief under the declared submission-candidate boundary.
- Final readiness audit verifies archive hashes, bundled evidence, minimal-shot
  integrity, judging evidence, and the production no-go boundary.

WOD validation numbers, if mentioned:

- 479 validation frames with valid rater preference labels.
- Constant velocity official validation-CV RFS: `7.022`.
- Combined candidate ranker official validation-CV RFS: `7.6594`.
- Reproducibility audit: `artifacts/wod_fastkin_gate_ridge175_scene020_repro_audit.json`
  passes against commit `a934d2740cdf831c5f7ee58f5ddae3c7de8692e8` using the
  Cosmos external embedding cache and `scene_gate_max_rate=0.20`.
- Gain versus constant velocity: `+0.635`.
- Combined candidate oracle official validation-CV RFS: `9.098`.
- Selector regret to oracle: `1.4386`.
- Breakthrough audit status: failed because worst-slice regret regressed.
- Minimal-shot caveat: this selector is preference-calibrated on retained
  validation labels under segment-grouped CV. It is not strict zero-shot
  autonomy and should not be presented as the core SoTA claim.

Required caveat:

Do not claim production autonomy, strict zero-shot WOD-E2E, or a hidden-test
leaderboard result. The local workspace does not contain the WOD-E2E test split
or confirmed leaderboard result log. The Grand submission should lead with the
minimal-shot closed-loop architecture and simulator evidence; WOD is auxiliary
benchmark infrastructure.

## Minor Commission

Track: Minor Commission, overall best simulation environment.

Title:

Randomized Long-Tail Scenario Generator for Minimal-Shot Autonomy

Attach:

- `artifacts/sota_submission_bundles/minor_commission.tar.gz`
- SHA-256: see `artifacts/sota_submission_bundles/submission_bundles_manifest.json`
- Video or slide deck based on `docs/sota-minor-slide-script.md`

Short claim:

This is a seeded procedural simulation environment for WOD-E2E-style long-tail
driving cases and compositional OOD stress cases, with reproducible scenario
generation and closed-loop evaluation artifacts.

Evidence to mention:

- Construction scenario demo.
- Foreign-object-debris scenario demo.
- Spotlight scenario demo.
- Multi-cluster evaluation sweep with JSON and CSV reports.
- Compositional OOD evaluation sweep with JSON and CSV reports.
- AlpaSignal bridge audit with JSON report.
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

Bundled OOD sweep:

- 350 additional closed-loop rollouts.
- 5 suites: WOD, compositional, adversarial, gauntlet, hidden.
- 350 / 350 successful rollouts.
- 326 / 350 benchmark passes.
- 0 collisions.
- 0 safe stalls.
- Gauntlet pass rate: 36 / 60 under stricter near-miss, intervention, and progress gates.

AlpaSignal bridge audit:

- 3 deterministic adapter cases.
- Static structured hazards become obstacles.
- Moving structured hazards become actors.
- Low-visibility and hard-braking signals create a caution zone.
- Each case emits finite 20-point trajectories and decision reasoning with the
  signal fields used by the adapter.

Required caveat:

This is a lightweight 2D procedural simulator, not a photorealistic sensor simulator. AlpaSim integration exists at trajectory/plugin level; full sensor-realistic perception integration is future work.

## Submission Email

Subject:

SoTA Commission I Submission: Minimal-Shot AV Architecture + Long-Tail Scenario Generator

Body:

Dear SoTA Commission team,

Please find my submissions for SoTA Commission I:

1. Grand Commission: Minimal-Shot Autonomy Architecture: Closed-Loop Long-Tail Reflexes
2. Minor Commission: Randomized Long-Tail Scenario Generator for Minimal-Shot Autonomy

GitHub repository:

`https://github.com/amtellezfernandez/minimal-shot-av`

Submitted commit:

Use the commit pointed to by tag `sota-commission-2026-submission`.

Attached bundles:

- `grand_commission.tar.gz`
  SHA-256: see `artifacts/sota_submission_bundles/submission_bundles_manifest.json`
- `minor_commission.tar.gz`
  SHA-256: see `artifacts/sota_submission_bundles/submission_bundles_manifest.json`

The Grand submission includes a WOD-E2E harness and validation-CV analysis, but
the minimal-shot claim is the closed-loop Spotlight Reflex architecture rather
than a validation-trained WOD selector. It does not claim a hidden-test
leaderboard result. The Minor submission includes the randomized scenario
generator and 550-run closed-loop evaluation sweep.

Best,

Alba Maria Tellez Fernandez
