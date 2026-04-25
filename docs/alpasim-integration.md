# AlpaSim Integration

This repository connects to AlpaSim as an optional model plugin named
`spotlight_reflex`.

## What The Bridge Does

AlpaSim's driver service discovers trajectory models through Python entry points
under `alpasim.models`. The adapter in
`minimal_shot_av.alpasim_spotlight:SpotlightReflexAlpaSimModel` implements
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
  --config-path=pkg://minimal_shot_av.alpasim_configs \
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
PYTHONPATH=src uv run --no-sync python scripts/import_alpasim_metrics.py \
  alpasim_spotlight_run \
  --output artifacts/alpasim_spotlight_evidence.json
```

The importer is dependency-free and reads either AlpaSim's aggregate
`metrics_results.txt` or a scripted `metrics_results.json`. It preserves the
distinction between evidence tiers:

- AlpaSim evidence is sensor-realistic, closed-loop validation.
- It is not an official COMPASS score.
- It should be reported alongside abstract COMPASS results, not merged into
  them as if they measured the same thing.

Imported gates currently cover AlpaSim's native metrics when present:

- `collision_at_fault`
- `offroad`
- `dist_to_gt_trajectory` or `plan_deviation`
- `safety_monitor_triggered`

## Honest Limitation

This first bridge is trajectory-level. It consumes AlpaSim route command and
ego speed, but it does not yet parse camera images into hazards. The stronger
submission version should add a perception adapter that converts AlpaSim camera
or traffic outputs into Spotlight Reflex hazards before RFS selection.

That next layer is where a permissively licensed perception model or structured
hazard extractor should sit:

```text
AlpaSim cameras + ego state
  -> structured hazard extraction
  -> Spotlight Reflex maneuver candidates
  -> RFS trust-region selector
  -> AlpaSim trajectory controller
```
