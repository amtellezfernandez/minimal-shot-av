# AlpaSim Integration

This repository connects to AlpaSim through optional model plugins:

- `spotlight_reflex`: the zero-shot geometry/rule selector.
- `token_dagger_bc`: a learned Token-BC/Token-DAgger checkpoint loaded from
  `model.checkpoint_path`.

## What The Bridge Does

AlpaSim's driver service discovers trajectory models through Python entry points
under `alpasim.models`. The adapter in
`minimal_shot_av.simulator.alpasim_spotlight:SpotlightReflexAlpaSimModel` implements
AlpaSim's `BaseTrajectoryModel` interface and returns a rig-frame trajectory:

- input: camera frame cache, route command, speed, acceleration, ego pose history
- output: `ModelPrediction(trajectory_xy, headings, reasoning_text)`
- trajectory: 20-point, 5-second Spotlight Reflex maneuver resampled to the
  configured output frequency
- debug: RFS score, selected maneuver, selected pseudo-reference labels

This is the right first connection because our current policy already emits
candidate trajectories. It avoids modifying AlpaSim internals.

## Install Into An AlpaSim Environment

From this repo, install the local plugin package into the AlpaSim driver
environment:

```bash
./.venv/bin/python scripts/setup_alpasim_local_plugin.py
```

This installs `minimal-shot-av` into `alpasim/.venv` and checks the AlpaSim
entry-point registry. The required models are:

- `spotlight_reflex`
- `token_dagger_bc`

You can rerun the check without reinstalling:

```bash
./.venv/bin/python scripts/setup_alpasim_local_plugin.py --check-only
```

The plugin list should include:

```text
Models: ..., spotlight_reflex, token_dagger_bc
```

For repeatable external-driver launches, use the repo-local launcher instead of
hand-editing Hydra overrides:

```bash
./.venv/bin/python scripts/run_alpasim_local_external.py \
  --mode print \
  --model token_dagger_iter2 \
  --scene-preset fresh_3scene
```

That writes:

- `runs/alpasim_<model>_<preset>_<timestamp>/launch-metadata.json`
- `driver-command.sh`
- `wizard-command.sh`

and prints the exact matched commands for the run.

## Run As An External Driver

The preferred path is the repo-local launcher, which writes a concrete
`external-driver-config.yaml` under the run directory and avoids Hydra config-discovery
edge cases:

```bash
./.venv/bin/python scripts/run_alpasim_local_external.py \
  --mode print \
  --model token_dagger_iter2 \
  --scene-preset fresh_3scene
```

If you want a real run immediately:

```bash
./.venv/bin/python scripts/run_alpasim_local_external.py \
  --mode both \
  --model token_dagger_iter2 \
  --scene-preset fresh_3scene
```

The launcher materializes:

- `external-driver-config.yaml`
- `driver-config.yaml` generated later by the wizard for its internal bookkeeping
- `driver-command.sh`
- `wizard-command.sh`
- `launch-metadata.json`

inside the generated run directory, so manual reruns do not require reconstructing
Hydra overrides.

The learned presets intentionally force `model.device: cuda`, so a production
AlpaSim sweep cannot silently run on CPU.

## One-Command Local Launch

To validate the full path on a small matched scene set:

```bash
./.venv/bin/python scripts/run_alpasim_local_external.py \
  --mode both \
  --model token_dagger_iter2 \
  --scene-preset fresh_3scene
```

For a config-only dry run:

```bash
./.venv/bin/python scripts/run_alpasim_local_external.py \
  --mode wizard \
  --model token_dagger_iter2 \
  --scene-preset fresh_3scene \
  --wizard-dry-run
```

Supported model presets:

- `spotlight_reflex`
- `token_dagger_iter2`
- `token_dagger_srcdecay`
- `token_dagger_iter2_clamped`
- `token_dagger_srcdecay_clamped`

Supported scene presets:

- `fresh_3scene`
- `front_camera_10scene_smoke`
- `front_camera_30scene_merged`

Notes:

- The learned presets force `model.device=cuda` in the external driver command.
- The launcher defaults to `topology=1gpu`; override with `--topology` if you
  want a larger AlpaSim layout.
- `token_dagger_iter2` and `token_dagger_srcdecay` currently share the same
  wizard-side driver family; the differentiator is the external driver config
  and checkpoint path.
- The `*_clamped` presets preserve learned token selection but clamp lateral
  token geometry to `max_lateral_offset_m=2.0`. Use these to test whether the
  AlpaSim failure is caused by unsafe lateral projection rather than learned
  safety calibration.

## Import AlpaSim Metrics Into Evidence

After a run finishes, AlpaSim writes native closed-loop metrics under the run
directory, typically:

```text
<run-dir>/aggregate/metrics_results.txt
```

Import those metrics into a COMPASS-side evidence JSON:

```bash
uv run alpasim-evidence \
  alpasim_spotlight_run \
  --output artifacts/alpasim_spotlight_evidence.json
```

The importer is dependency-free and reads AlpaSim aggregate CSV, text, or JSON
metrics. Preferred files are discovered automatically in this order:

- `aggregate/metrics_results.csv`
- `aggregate/metrics_results.txt`
- `aggregate/metrics_results.json`

It preserves the distinction between evidence tiers:

- AlpaSim evidence is sensor-realistic, closed-loop validation.
- It is not an official COMPASS score.
- It should be reported alongside abstract COMPASS results, not merged into
  them as if they measured the same thing.

Imported gates currently cover AlpaSim's native metrics when present:

- `collision_at_fault`
- `offroad`
- `dist_to_gt_trajectory` or `plan_deviation`
- `safety_monitor_triggered`

## Per-Episode Outcome Audit

Aggregate AlpaSim metrics are post-processed. In particular, the native
aggregation can truncate a rollout after `offroad_or_collision` and after
`dist_to_gt_trajectory >= 4m`, so a lower aggregate collision rate can hide a
policy that first leaves the route and only later collides or goes offroad.

Use the per-episode audit before interpreting an external result:

```bash
./.venv/bin/python scripts/summarize_alpasim_episodes.py \
  runs/alpasim_spotlight_reflex_fresh_3scene_20260515_195145 \
  runs/alpasim_token_dagger_iter2_fresh_3scene_20260515_210322 \
  runs/alpasim_token_dagger_srcdecay_fresh_3scene_20260515_211218
```

The audit reads `aggregate/metrics_unprocessed.parquet` and controller CSVs,
then reports both:

- `first_score_cutoff`: first route-deviation/collision/offroad cutoff relevant
  to aggregate scoring.
- `first_raw_incident`: first raw collision/offroad event in the full rollout.

For the matched two-scene fresh subset, the learned DAgger policies do not
simply stop safely. They cross `dist_to_gt_trajectory >= 4m` around 2.9s in one
scene and collide around 2.5s in the other; source decay avoids one raw collision
but still leaves route and later goes offroad.

## Signal Usage

The bridge now consumes more than the route command. It uses:

- route command
- ego speed and acceleration
- camera brightness as a lightweight low-visibility signal
- optional structured hazard metadata if an upstream AlpaSim/AlpaSignal layer
  attaches fields such as `structured_hazards`, `hazards`, `traffic_hazards`,
  or `alpasignal`

If structured hazards are present, they are converted into Spotlight Reflex
obstacles before simulator-native selection. If no hazards are present but the
camera/dynamics signal indicates low visibility or hard braking, the adapter
adds a conservative caution zone. This keeps the bridge simulator-only and does
not import WOD/RFS/model code.

A stronger submission version should replace the lightweight brightness/dynamics
heuristic with a richer permissively licensed perception model or structured
hazard extractor:

```text
AlpaSim cameras + ego state
  -> structured hazard extraction
  -> Spotlight Reflex maneuver candidates
  -> simulator-native trajectory selector
  -> AlpaSim trajectory controller
```
