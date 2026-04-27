# AlpaSim Integration

This repository connects to AlpaSim as an optional model plugin named
`spotlight_reflex`.

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

From this repo:

```bash
uv pip install -e .
```

From the AlpaSim repo/environment, verify discovery:

```bash
uv run alpasim-info
```

The plugin list should include:

```text
Models: ..., spotlight_reflex
Configs: ..., spotlight_reflex
```

## Run As An External Driver

Start the Spotlight Reflex driver:

```bash
uv run --project alpasim/src/driver python -m alpasim_driver.main \
  --config-path=pkg://minimal_shot_av.simulator.alpasim_configs \
  --config-name=driver/spotlight_reflex
```

Then launch AlpaSim with the external-driver deployment:

```bash
uv run --project alpasim/src/wizard alpasim_wizard \
  deploy=local_external_driver \
  driver=spotlight_reflex \
  wizard.log_dir=$PWD/alpasim_spotlight_run \
  scenes.scene_ids='["<scene-id>"]'
```

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
