# Submission Checklist

## Shared Required

- Deadline tracked as May 10, 2026.
- GitHub repo is public or ready to share privately.
- README states that the repo supports two separate submissions.
- `docs/grand-submission.md` is complete enough to submit the Grand Commission track.
- `docs/minor-simulation-submission.md` is complete enough to submit the Minor Commission track.
- Tests pass with `uv run --no-sync python -m unittest discover -s tests`.
- Demo artifacts are generated into separate `artifacts/grand_*` and `artifacts/minor_*` directories.

## Grand Commission: Autonomy Architecture

- README states WOD-E2E as the target benchmark and describes the `(20, 2)` waypoint task.
- README states the no-AV-finetuning constraint clearly.
- `docs/waymo-data-access.md` records the official access/download links and current local data status.
- `docs/wod-e2e-deliverable.md` documents the dataset contract, scenario clusters, and RFS implications.
- `docs/wod-e2e-schema.md` documents the proto fields, submission fields, and loader/evaluator sanity checks.
- `models/DECLARATION.md` fully lists base models, datasets, prompts/scaffolds, and external services used.
- `notebooks/` contains the WOD-E2E exploration and analysis notebook.
- `docs/two-page-writeup.md` is condensed into the final PDF or document.
- `docs/video-outline.md` has been turned into a 1-5 minute video or slide deck.
- At least one failure case is documented with component-level diagnosis.
- Motivation and use of prize money are explicitly stated.
- Spotlight Reflex demo command is included:
  `uv run --no-sync python scripts/run_demo.py --policy spotlight-reflex --scenario-cluster spotlight --seed 3 --artifacts-dir artifacts/grand_spotlight_demo`

## Grand Commission: WOD-E2E Specific

- Official Waymo terms and access requirements are respected.
- Dataset is downloaded through <https://waymo.com/open/download/> after Google sign-in and terms acceptance.
- Dataset split usage is declared: train, validation, test, or sample-only.
- Validation rater labels are not represented as test performance.
- Output trajectory format is correct: 20 future points, first point at 0.25 seconds.
- Vehicle-coordinate convention is handled consistently, including the public-page/proto origin wording mismatch.
- `frame.context.name` is used as the submission `frame_name`.
- `submission_type`, public model pretraining fields, model names, and parameter count are filled if a proto submission is generated.
- Challenge-provided JSON frame list is used to select required test frames.
- Submission file is packaged as serialized `E2EDChallengeSubmission` proto file(s) in `.tar.gz`.
- Test submission limit is respected: 6 submissions every 30 days, excluding errored submissions.
- RFS is computed only on frames with valid `preference_trajectories`.
- Scenario-cluster analysis is included where labels are available.
- Submission proto writer is tested if leaderboard submission is attempted.

## Minor Commission: Simulation Environment

- README and `docs/minor-simulation-submission.md` describe the procedural generator as the simulation-environment submission.
- The generator supports all 11 WOD-E2E clusters.
- Scenarios are reproducible by `(cluster, seed)`.
- JSON artifacts include `scenario.cluster` and `scenario.tags`.
- JSON artifacts include typed actors, map features, and environment metadata.
- Ambient scene texture is separated from blocking evaluation hazards.
- Compositional OOD scenarios are generated independently from WOD cluster labels.
- Scenario manifests include `primary_hazard_id`, `intended_decision`, `difficulty`, `ood_axes`, and `hazard_composition`.
- The adversarial suite includes composed hazards and at least one understood failure case.
- COMPASS report is generated with oracle solvability, reasoning, recovery, and generalisation-gap scores.
- SVG artifacts visually show randomized lane/obstacle layouts.
- Scenario evaluation writes `scenario_eval.json` and `scenario_eval.csv`.
- The video or slide deck shows at least three clusters and at least two different seeds.
- The write-up is honest that this is a lightweight 2D simulator, not AlpaSim or photorealistic sensor simulation.
- AlpaSim is described precisely: trajectory-level plugin available, full
  sensor/perception integration still future work.
- Minor demo commands are included, for example:
  `uv run --no-sync python scripts/run_demo.py --policy spotlight-reflex --scenario-cluster construction --seed 1 --artifacts-dir artifacts/minor_construction_seed1`

## Strongly Recommended For Either Track

- Include latency and hardware assumptions.
- Include ablations against constant-velocity and route-following baselines.
- Include invalid-trajectory and route-violation checks.
- Separate frozen base-model reasoning from learned or hand-coded trajectory logic.
- Pin package versions before submission.
- Add a small evaluation table across WOD-E2E validation clusters.
