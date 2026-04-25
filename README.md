# Minimal-Shot AV

Submission scaffold for **SoTA Commission I: Minimal-Shot Autonomy**.

## Thesis

This project targets the Waymo Open Dataset for End-to-End Driving (WOD-E2E) with a deliberately strict claim:

> build an end-to-end driving policy for long-tail WOD-E2E scenes without fine-tuning any base model on AV-specific data.

The proposed architecture is a **Predictive Reflex**: a zero-shot vision-language/world-model critic interprets the scene, a compact temporal state-space pilot compresses the driving history, and a latent maneuver library decodes the result into 5-second ego waypoints.

The goal is not to claim a production AV stack. WOD-E2E is an open-loop trajectory benchmark. The goal is to demonstrate minimal-shot generalization on rare, safety-critical driving scenarios where memorized routes, nominal imitation, and ADE-only optimization are weakest.

## Target Benchmark

Primary target: **WOD-E2E / Vision-based End-to-End Driving**.

Relevant task shape:

- Input: 8-camera 360-degree context, ego status history, and high-level route command.
- History: 12 seconds of test context before the target frame.
- Output: 5 seconds of future ego trajectory in vehicle coordinates, sampled at 4 Hz, shaped `(20, 2)`.
- Evaluation: Rater Feedback Score (RFS), with ADE as a secondary metric.
- Stress distribution: construction, intersections, pedestrians, cyclists, cut-ins, debris, special vehicles, spotlight cases, and other long-tail events.

Dataset-critical details:

- The record proto is `E2EDFrame`.
- `frame.context.name` is the required submission key.
- `past_states` covers `(-4s, 0]` at 4 Hz.
- `future_states` covers `(0, 5s]` at 4 Hz where labels are available.
- `intent` is one of `UNKNOWN`, `GO_STRAIGHT`, `GO_LEFT`, or `GO_RIGHT`.
- `preference_trajectories` contains up to 3 rater-scored futures on selected validation frames.
- The official proto states trajectory origin is the middle of the ego rear axle; public prose may describe the vehicle center, so implementation must verify the exact downloaded version.

Official references:

- Waymo Open Dataset access: <https://waymo.com/open/>
- Download page: <https://waymo.com/open/download/>
- WOD-E2E dataset page: <https://waymo.com/open/data/e2e/>
- 2025 E2E challenge page: <https://waymo.com/open/challenges/2025/e2e-driving/>
- Official codebase: <https://github.com/waymo-research/waymo-open-dataset>
- WOD-E2E paper: <https://arxiv.org/abs/2510.26125>

Repo references:

- Data access and local data check: [`docs/waymo-data-access.md`](docs/waymo-data-access.md)
- Dataset dossier: [`docs/wod-e2e-deliverable.md`](docs/wod-e2e-deliverable.md)
- Schema and metric reference: [`docs/wod-e2e-schema.md`](docs/wod-e2e-schema.md)
- Leaderboard reference: [`docs/leaderboard.md`](docs/leaderboard.md)
- Winning method reports: [`docs/winning-methods.md`](docs/winning-methods.md)
- Competitive analysis: [`docs/competitive-analysis.md`](docs/competitive-analysis.md)
- Proposed zero-shot architecture: [`docs/spotlight-reflex.md`](docs/spotlight-reflex.md)

Current local data status:

- Official Waymo code/protos/tutorial are present in `waymo-open-dataset/`.
- Actual WOD-E2E train/validation/test TFRecords are not present yet.
- The download page is Google-sign-in gated, so dataset acquisition must be completed manually through Waymo access.

## Architecture Direction

The recommended submission architecture is:

1. **Scene Critic**
   - Uses a general-purpose vision-language or world model without AV fine-tuning.
   - Produces semantic scene tags, hazard hypotheses, route intent checks, and uncertainty notes.

2. **Predictive State-Space Pilot**
   - Maintains a compact temporal state over the 12-second visual and ego history.
   - Uses linear-time recurrent memory rather than transformer-style full-context replay at every step.

3. **Latent Maneuver Library**
   - Represents maneuvers such as yield, brake, nudge, lane-change, cut-in response, debris avoidance, and conservative fallback.
   - Retrieves maneuver candidates from latent scene signatures instead of memorizing WOD-E2E routes.

4. **Trajectory Decoder and Safety Projector**
   - Converts maneuver candidates into 20 future `(x, y)` waypoints.
   - Enforces simple kinematic plausibility and route-command consistency.

5. **Rater-Aware Selection**
   - Scores candidates against validation-time RFS behavior where labels are available.
   - Treats rater preference as the target, not just L2 imitation of the logged future.
   - Pays special attention to 3s and 5s trajectory positions because RFS trust regions are evaluated at those times.

## Current Baseline

The runnable code in this repo is currently a lightweight 2D navigation harness:

- procedurally generated lanes and obstacle fields
- no stored route library or map-specific tuning
- a reactive policy that follows corridor geometry and avoids hazards online
- artifact generation for demos and documentation

This baseline is not the final WOD-E2E policy. It is a fast harness for demonstrating the architecture pattern before connecting real WOD-E2E TFRecords and submission protos.

## Repo Structure

- `src/minimal_shot_av/`: simulation and baseline policy
- `scripts/run_demo.py`: generate a random scenario and rollout artifacts
- `docs/`: WOD-E2E submission plan, write-up, video, and checklist
- `models/`: base-model and architecture declaration
- `notebooks/`: analysis notebook plan
- `artifacts/`: generated outputs

## Quickstart

Use Python 3.10+.

```bash
python3 scripts/run_demo.py
```

This writes:

- `artifacts/latest_rollout.json`
- `artifacts/latest_rollout.svg`

## Submission Deliverables

The submission should include:

- GitHub repo with code, README, and full base-model declaration.
- Analysis notebook showing WOD-E2E data inspection, scenario focus, failed scaffolds, and final design rationale.
- 1-5 minute video or slide deck showing the policy on WOD-E2E scenes, including one understood failure.
- Short write-up of at most two pages covering motivation, architecture, results, failures, and next funding milestone.

## Recommended Next Steps

1. Use `docs/spotlight-reflex.md` as the target architecture: frozen scene critic, counterfactual hypotheses, maneuver library, and exact RFS trust-region selection.
2. Add a `wod_e2e/` package for `E2EDFrame` parsing, camera extraction, ego-history parsing, RFS evaluation, and submission proto writing.
3. Download WOD-E2E through <https://waymo.com/open/download/> and complete the checks in `docs/waymo-data-access.md`.
4. Build the notebook from `notebooks/README.md`: parse records, visualize 8-camera context, inspect rater trajectories, and compute official RFS.
5. Implement constant-stop, constant-velocity, curvature, and intent-template baselines before claiming architecture value.
6. Implement a zero-AV-finetune first pass using a frozen VLM scene critic plus hand-coded trajectory primitives.
7. Evaluate on validation RFS by cluster and document at least one Spotlight failure with a concrete component-level cause.
