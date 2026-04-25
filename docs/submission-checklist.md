# Submission Checklist

## Required

- GitHub repo is public or ready to share privately.
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

## WOD-E2E Specific

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

## Strongly Recommended

- Include latency and hardware assumptions.
- Include ablations against constant-velocity and route-following baselines.
- Include invalid-trajectory and route-violation checks.
- Separate frozen base-model reasoning from learned or hand-coded trajectory logic.
- Pin package versions before submission.
- Add a small evaluation table across WOD-E2E validation clusters.
