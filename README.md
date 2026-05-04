# Minimal-Shot AV

Submission scaffold and benchmark harness for **SoTA Commission I:
Minimal-Shot Autonomy**.

This repo is intentionally split into two independent submission tracks:

- **Grand Commission submission candidate:** minimal-shot Spotlight Reflex
  architecture, with closed-loop long-tail demos and a WOD-E2E harness as
  auxiliary benchmark infrastructure.
- **Minor Commission submission:** COMPASS/AlpaSim simulation stack, including Spotlight Reflex and randomized long-tail scenarios.

The tracks must stay agnostic to avoid simulator/model bias. The simulator has
its own trajectory selector, while WOD/RFS scoring is model-side only. See:

- [`docs/architecture-boundaries.md`](docs/architecture-boundaries.md)
- [`docs/grand-submission.md`](docs/grand-submission.md)
- [`docs/minor-simulation-submission.md`](docs/minor-simulation-submission.md)

Commission constraints to keep visible:

- Deadline: **May 10, 2026**.
- Required package: GitHub repo, 1-5 minute video or slide deck, motivation,
  and a short write-up.
- Judging criteria: technical excellence, novelty, feasibility, and adherence
  to the minimal-shot brief.
- The commission is not only a leaderboard contest; honest execution,
  randomized simulation, latency realism, and failure analysis matter.

## Thesis

This project includes a Waymo Open Dataset for End-to-End Driving (WOD-E2E)
harness, but the Grand Commission claim is intentionally narrower:

> build a runnable minimal-shot autonomy prototype that reasons through
> unfamiliar long-tail scenarios with explicit world-state, maneuver generation,
> and safety selection; use WOD-E2E as an auxiliary benchmark harness rather
> than the proof of zero-shot autonomy.

The active model path is non-text: WOD-E2E ego history and route intent feed
structured trajectory generators, residual proposal models, a lightweight
world-model candidate source, and an RFS-calibrated numeric selector that
outputs 5-second ego waypoints.

The goal is not to claim a production AV stack, strict zero-shot WOD-E2E, or
frontier-level scene reasoning. WOD-E2E is an open-loop trajectory benchmark,
and the current selector experiments use retained validation preference labels
under segment-grouped CV. The minimal-shot evidence should be read from the
closed-loop Spotlight Reflex architecture and randomized scenario demos; the
WOD path is supporting infrastructure and analysis.
See [`docs/solution-reset.md`](docs/solution-reset.md) for the current
solution bar and why the present world-model result is not yet strong enough.

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
- WOD-E2E validation TFRecords are present locally under
  `waymo_open_dataset_end_to_end_camera_v_1_0_0/val` with 93 shards.
- The local parser finds 479 preference-labeled validation frames and 1,437
  valid human-rated reference trajectories.
- Train and test TFRecords are not present in this workspace.
- The download page is Google-sign-in gated, so any missing splits must be
  acquired manually through Waymo access.

## Architecture Direction

The active WOD-E2E model architecture is:

1. **WOD-E2E Adapter**
   - Parses official `E2EDFrame` TFRecords with Waymo protos.
   - Uses frame name, past ego trajectory, initial speed, route intent, and future labels where allowed.

2. **Structured Trajectory Generators**
   - Kinematic candidates provide constant-velocity, acceleration, heading-change, and hold-position baselines.
   - Ridge residual candidates provide the current best learned non-text proposal set.
   - Anchor-residual candidates are experimental and disabled by default until selected RFS improves, not only oracle RFS.
   - Neural anchor-residual proposal ensembles are available as a measured
     model-side contender. The current best official held-out subset run is
     `7.737680847131364` RFS on `159` frames with oracle `9.178972912996967`,
     using the three-model ensemble in `artifacts/wod_neural_holdout/`; this is
     stronger than the full-479-frame non-neural `7.65941846208851` champion on
     its own subset, but it is not yet a full-validation or leaderboard result.
     It remains below the `8.0461` leaderboard snapshot target and must be
     described as minimal-shot WOD development evidence, not strict zero-shot
     deployment.
   - Lightweight world-model candidates are experimental. Current official
     validation-CV evidence shows only a tiny selected-RFS gain, so they are not
     yet the central solution.

3. **RFS-Calibrated Selector**
   - Trains a structured numeric selector on validation preference labels under segment-grouped cross-validation.
   - Uses official RFS for confirmed benchmark reports.
   - Includes source diagnostics so experimental proposal families cannot silently degrade the default path.
   - Scores final candidate JSONL rows with `ranker_score` before submission packaging.

4. **Decision Explainability**
   - Closed-loop Spotlight Reflex rollouts emit per-step selector references,
     effective score terms, safety/progress penalties, and top candidate
     summaries in `latest_rollout.json`.
   - Each step also records label-free world geometry: obstacle pressure,
     route blockage, corridor blockage, side clearances, and the preferred
     escape side. These fields are computed from occupancy and route geometry,
     not object category names or scenario cluster labels.
   - The AlpaSim adapter exports the same selected-maneuver rationale in
     `reasoning_text`, so reviewers can inspect why a trajectory was chosen
     instead of only seeing the final path.

5. **Submission Writer**
   - Selects one `(20, 2)` trajectory per required frame by contextual ranker score.
   - Packages and validates `E2EDChallengeSubmission` `.tar.gz` artifacts.

## Current Runnable System

The runnable code in this repo is split into independent model and simulator tracks.

Model track:

- official WOD-E2E parser integration
- non-text trajectory candidate generators
- official/local RFS evaluation
- segment-grouped validation CV
- lightweight world-model candidate ablations
- submission packaging and validation

Simulator track:

- procedurally generated lanes and obstacle fields
- 11 WOD-E2E-inspired long-tail scenario clusters
- compositional OOD suites that independently sample topology, hazards, conditions, and novel objects
- adversarial and hidden holdout suites for harder frozen-policy evaluation
- a gauntlet suite with synchronized threats and quality-gated benchmark scoring
- a baseline reactive policy for comparison
- a Spotlight Reflex policy with deterministic maneuver candidates and simulator-native trajectory selector scoring
- artifact generation for demos and documentation

This is not a production AV stack and it is not yet a completed WOD-E2E leaderboard submission because train/test shards and the official frame list are still required locally.

## Repo Structure

- `src/minimal_shot_av/simulator/`: simulation, scenario generation, baseline policy, and Spotlight Reflex
- `src/minimal_shot_av/model/`: WOD-E2E loading, structured trajectory candidates, rankers, and model-side evaluation
- `src/minimal_shot_av/neutral/`: shared metrics and evidence parsing with no simulator/model imports
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

Fast laptop test run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python scripts/run_tests.py
```

The test runner uses standard-library multiprocessing and chooses the worker
count automatically, capped for laptop responsiveness. For the fastest edit
loop, skip benchmark/evidence sweeps:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python scripts/run_tests.py --quick
```

Use `--slow` to run only the benchmark/evidence modules, `--workers 1` for
serial debugging, `--workers max` to use one process per test module, or pass
specific modules/files, for example `scripts/run_tests.py tests/test_compass.py`.

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

Minor submission OOD sweep:

```bash
uv run --no-sync python scripts/evaluate_scenarios.py \
  --policy spotlight-reflex \
  --suite all \
  --seed-start 1 \
  --seed-end 10 \
  --output-dir artifacts/minor_ood_eval
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

AlpaSim evidence import:

```bash
uv run alpasim-evidence alpasim_spotlight_run \
  --output artifacts/alpasim_spotlight_evidence.json
```

AlpaSignal bridge audit:

```bash
uv run --no-sync python scripts/audit_alpasignal_bridge.py \
  --output artifacts/minor_alpasignal_bridge/alpasignal_bridge_audit.json
```

Minor closed-loop runtime audit:

```bash
uv run --no-sync python scripts/audit_minor_runtime_constraints.py \
  --seed-start 1 \
  --seed-end 3 \
  --target-step-ms 50 \
  --output artifacts/minor_runtime/minor_runtime_constraints.json
```

Minor visual gallery:

```bash
uv run --no-sync python scripts/build_minor_visual_gallery.py \
  --bundle-root artifacts/sota_submission_bundles \
  --output-dir artifacts/sota_submission_bundles/minor_visual_gallery
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
PYTHONPATH=src uv run --no-sync python -m minimal_shot_av.simulator.compass ladder \
  --policy spotlight-reflex \
  --profile compass-v0 \
  --seed-start 1 \
  --seed-end 10 \
  --output artifacts/compass_ladder.json
```

Config-driven stress benchmark:

```bash
PYTHONPATH=src uv run --no-sync python -m minimal_shot_av.simulator.compass ladder \
  --policy spotlight-reflex \
  --profile-json configs/compass_stress.json \
  --seed-start 1 \
  --seed-end 20 \
  --output artifacts/compass_stress.json
```

SOTIF-aligned evidence package:

```bash
PYTHONPATH=src uv run --no-sync python -m minimal_shot_av.simulator.certification \
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

Controlled hardware / vehicle validation preflight:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python scripts/run_vehicle_validation_shadow.py
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python scripts/audit_hardware_vehicle_validation.py --hil-check
```

This gate is narrower than production readiness. It can authorize starting
bench HIL and supervised closed-course validation only when strict simulator
evidence, simulator integration, safety-case sections, estop/takeover controls,
shadow command replay, logging, rollback, speed limits, and geofencing are
present. It still blocks public-road deployment and unsupervised vehicle
control. See [`docs/hardware-vehicle-validation.md`](docs/hardware-vehicle-validation.md).

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

1. For the **Grand** submission, turn `docs/grand-submission.md` into the slide/video script and use the downloaded WOD-E2E validation split for analysis/failure cases.
2. For the **Minor** submission, turn `docs/minor-simulation-submission.md` into the simulation-environment slide/video script and show seeded cluster variation.
3. Keep artifacts for each track in separate directories under `artifacts/`.
