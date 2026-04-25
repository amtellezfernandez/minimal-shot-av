# Minimal-Shot AV

Submission scaffold for **SoTA Commission I: Minimal-Shot Autonomy**.

This repo is intentionally split into two independent submission tracks:

- **Grand Commission submission:** Spotlight Reflex, a minimal-shot autonomy architecture and runnable zero-AV-finetune policy slice.
- **Minor Commission submission:** WOD-E2E procedural scenario generator, a randomized simulation environment for long-tail driving cases.

The tracks share code, but can be submitted separately. See:

- [`docs/grand-submission.md`](docs/grand-submission.md)
- [`docs/minor-simulation-submission.md`](docs/minor-simulation-submission.md)

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

## Current Runnable System

The runnable code in this repo is currently a lightweight 2D navigation harness plus two scenario generators:

- procedurally generated lanes and obstacle fields
- 11 WOD-E2E-inspired long-tail scenario clusters
- compositional OOD suites that independently sample topology, hazards, conditions, and novel objects
- adversarial and hidden holdout suites for harder frozen-policy evaluation
- a gauntlet suite with synchronized threats and quality-gated benchmark scoring
- a baseline reactive policy for comparison
- a Spotlight Reflex policy with deterministic maneuver candidates and exact RFS trust-region scoring
- artifact generation for demos and documentation

This is not a production AV stack and it is not a completed WOD-E2E leaderboard submission. It is a fast, reproducible prototype for the SoTA Commission brief: randomized scenario generation plus a minimal-shot autonomy policy demonstration.

## Repo Structure

- `src/minimal_shot_av/`: simulation, scenario generation, baseline policy, and Spotlight Reflex
- `scripts/run_demo.py`: generate baseline, WOD-style, or compositional rollout artifacts
- `docs/`: WOD-E2E submission plan, write-up, video, and checklist
- `models/`: base-model and architecture declaration
- `notebooks/`: analysis notebook plan
- `artifacts/`: generated outputs

## Quickstart

Use Python 3.10+ and `uv`.

```bash
uv run --no-sync python scripts/run_demo.py
```

This writes:

- `artifacts/latest_rollout.json`
- `artifacts/latest_rollout.svg`

Grand Commission architecture demo:

```bash
uv run --no-sync python scripts/run_demo.py \
  --policy spotlight-reflex \
  --scenario-cluster spotlight \
  --seed 3 \
  --artifacts-dir artifacts/grand_spotlight_demo
```

Minor Commission simulation demo:

```bash
uv run --no-sync python scripts/run_demo.py \
  --policy spotlight-reflex \
  --scenario-cluster construction \
  --seed 1 \
  --artifacts-dir artifacts/minor_construction_demo
```

Minor Commission scenario sweep:

```bash
uv run --no-sync python scripts/evaluate_scenarios.py \
  --policy both \
  --suite wod \
  --seed-start 1 \
  --seed-end 20 \
  --output-dir artifacts/eval_wod
```

Compositional OOD sweep:

```bash
uv run --no-sync python scripts/evaluate_scenarios.py \
  --policy both \
  --suite compositional \
  --seed-start 1 \
  --seed-end 20 \
  --output-dir artifacts/eval_compositional
```

Adversarial stress sweep:

```bash
uv run --no-sync python scripts/evaluate_scenarios.py \
  --policy both \
  --suite adversarial \
  --seed-start 1 \
  --seed-end 20 \
  --output-dir artifacts/eval_adversarial
```

Gauntlet benchmark sweep:

```bash
uv run --no-sync python scripts/evaluate_scenarios.py \
  --policy both \
  --suite gauntlet \
  --seed-start 1 \
  --seed-end 20 \
  --output-dir artifacts/eval_gauntlet
```

COMPASS composite benchmark:

```bash
PYTHONPATH=src uv run --no-sync python -m minimal_shot_av.compass ladder \
  --policy spotlight-reflex \
  --profile compass-v0 \
  --seed-start 1 \
  --seed-end 10 \
  --output artifacts/compass_ladder.json
```

Config-driven stress benchmark:

```bash
PYTHONPATH=src uv run --no-sync python -m minimal_shot_av.compass ladder \
  --policy spotlight-reflex \
  --profile-json configs/compass_stress.json \
  --seed-start 1 \
  --seed-end 20 \
  --output artifacts/compass_stress.json
```

SOTIF-aligned evidence package:

```bash
PYTHONPATH=src uv run --no-sync python -m minimal_shot_av.certification \
  --policy spotlight-reflex \
  --profile sotif-v0 \
  --seed-start 1 \
  --seed-end 10 \
  --output artifacts/compass_evidence_report.json
```

This is a structured simulation evidence report, not a legal certification or
public-road deployment approval. The default `1%` collision-confidence
threshold requires `381` zero-collision ranked runs per official level, so short
runs are expected to report evidence gaps. Use `--profile smoke` for plumbing
checks, or pass `--odd-spec odd.json --thresholds thresholds.json` for a custom
ODD/evidence standard. Benchmark assumptions are also configuration, not source
edits: use `--profile-json compass_profile.json` with COMPASS, or
`--compass-profile-json compass_profile.json` with the evidence CLI. The COMPASS
profile can override official level weights, score weights, trajectory scoring,
suite penalties, and compositional scenario-generation settings such as hazard
counts, suite pressure, ambient density, corridor clearance, and difficulty
scoring. For harder ODD portability, the same profile can override topology
geometry, hazard geometry, visibility/latency ranges, and oracle feasibility
parameters.

Available procedural clusters:

`construction`, `intersection`, `pedestrian`, `cyclist`, `multi-lane maneuver`, `single-lane maneuver`, `cut-in`, `foreign object debris`, `special vehicle`, `spotlight`, `others`.

Available OOD suites:

`compositional`, `adversarial`, `gauntlet`, `hidden`.

See [`docs/compositional-ood-eval.md`](docs/compositional-ood-eval.md) for the scenario manifest and evaluation metrics.
See [`docs/compass-benchmark.md`](docs/compass-benchmark.md) for the oracle, reasoning, recovery, and generalisation-gap benchmark layer.

## Split Submission Deliverables

The repo can support two separate submissions:

- **Grand Commission:** Spotlight Reflex architecture, model declaration, WOD-E2E analysis plan, demo artifacts, and architecture write-up.
- **Minor Commission:** randomized WOD-style scenario generator, cluster templates, reproducibility evidence, demo artifacts, and simulation-environment write-up.

Shared supporting materials:

- [`models/DECLARATION.md`](models/DECLARATION.md)
- [`docs/two-page-writeup.md`](docs/two-page-writeup.md)
- [`docs/video-outline.md`](docs/video-outline.md)
- [`docs/submission-checklist.md`](docs/submission-checklist.md)

## Recommended Next Steps

1. For the **Grand** submission, turn `docs/grand-submission.md` into the slide/video script and connect WOD-E2E TFRecords when access is available.
2. For the **Minor** submission, turn `docs/minor-simulation-submission.md` into the simulation-environment slide/video script and show seeded cluster variation.
3. Keep artifacts for each track in separate directories under `artifacts/`.
