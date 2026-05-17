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

Use the full bootstrap first. It clones the pinned upstream AlpaSim checkout if
missing, standardizes on `uv`, creates `./.venv`, installs the `alpasim`
extra, and, if `ALPASIM_ROOT` points at a real AlpaSim checkout, installs this
repo into the AlpaSim driver environment too:

```bash
export ALPASIM_ROOT=/abs/path/to/alpasim
./scripts/bootstrap_alpasim_runtime.sh
./scripts/check_alpasim_readiness.py --scene-preset fresh_3scene
./.venv/bin/python scripts/fetch_checkpoints.py
```

`ALPASIM_ROOT` must be a real nested AlpaSim git checkout. A copied directory
with `src/driver` and `src/wizard` but no `.git` marker is not sufficient: the
wizard will discover the wrong repo root and fail to find its Hydra configs.
`./scripts/bootstrap_alpasim_checkout.sh` now detects that invalid layout,
moves it aside, and reclones the pinned upstream checkout automatically.

Platform note: local AlpaSim external-driver rollouts currently require an
`x86_64` host. The upstream NVIDIA NRE `sensorsim` image is amd64-only; on ARM
machines such as DGX Spark we observed emulated `sensorsim` processes hang or
crash before opening the expected gRPC port. `./scripts/check_alpasim_readiness.py`
now fails fast on ARM unless `MSA_ALLOW_UNSUPPORTED_ALPASIM_ARM=1` is set to
override the guard intentionally.

If the repo env already exists and you only need the repo-side env without
preparing runtime Docker execution, use:

```bash
./scripts/bootstrap_alpasim_env.sh
```

For the supported one-command runtime bootstrap, use:

```bash
./scripts/bootstrap_alpasim_runtime.sh
```

That wrapper validates the checkout, bootstraps both Python environments, runs
readiness before any expensive Docker work, and only then builds the local
`alpasim-base:0.66.0` image.

If you only need the plugin registration step:

```bash
ALPASIM_ROOT=/abs/path/to/alpasim ./.venv/bin/python scripts/setup_alpasim_local_plugin.py
```

This bootstraps a minimal `$ALPASIM_ROOT/.venv` with `uv`, installs the
AlpaSim driver and wizard components needed for local external-driver runs,
installs `minimal-shot-av` into that env, and checks the AlpaSim entry-point
registry. The required models are:

- `spotlight_reflex`
- `token_dagger_bc`

The learned-model checkpoints are published separately in the public Hugging Face repo
`amtellezfernandez/minimal-shot-av-corl2027-checkpoints`. Use
`./.venv/bin/python scripts/fetch_checkpoints.py` to populate the expected local
`artifacts/...` paths from the tracked manifest.

## Upstream Delta

The AlpaSim checkout used on this laptop is an external repo at:

```text
/home/amdev/sota/minimal-shot-av/alpasim
```

It is based on upstream `NVlabs/alpasim` `v2026.4` (`049f70f`) and currently
needs two local override files that are now stored directly in this repo:

- `third_party/alpasim_overrides/src/wizard/alpasim_wizard/deployment/docker_compose.py`
- `third_party/alpasim_overrides/Dockerfile.amd64`
- `third_party/alpasim_overrides/src/driver/src/alpasim_driver/models/__init__.py`

`scripts/setup_alpasim_local_plugin.py` now copies these overrides into
`$ALPASIM_ROOT` automatically unless `--skip-overrides` is passed.

The important behavior change is in `docker_compose.py`: it prevents the runtime
service from requesting `count: all` GPUs when the selected topology does not
assign GPUs to simulation containers.

The pinned external source version is:

- upstream repo: `https://github.com/NVlabs/alpasim.git`
- upstream ref: `v2026.4`

You can rerun the check without reinstalling:

```bash
ALPASIM_ROOT=/abs/path/to/alpasim ./.venv/bin/python scripts/setup_alpasim_local_plugin.py --check-only
```

The plugin list should include:

```text
Models: ..., spotlight_reflex, token_dagger_bc
```

For repeatable external-driver launches, use the repo-local launcher instead of
hand-editing Hydra overrides:

```bash
ALPASIM_ROOT=/abs/path/to/alpasim ./.venv/bin/python scripts/run_alpasim_local_external.py \
  --mode print \
  --model token_dagger_iter2_hybrid_clamped \
  --scene-preset fresh_3scene
```

That writes:

- `runs/alpasim_<model>_<preset>_<timestamp>/launch-metadata.json`
- `driver-command.sh`
- `wizard-command.sh`

and prints the exact matched commands for the run.

The launcher now rejects the three setup failures that previously looked like
runtime bugs:

- invalid copied `ALPASIM_ROOT` checkout with no `.git`
- Docker daemon not reachable from the current user
- missing local base image `alpasim-base:0.66.0`

Build the local image explicitly with:

```bash
./scripts/build_alpasim_base_image.sh
```

Or verify everything at once before launching:

```bash
./scripts/check_alpasim_readiness.py --scene-preset front_camera_10scene_smoke
```

## Run As An External Driver

The preferred path is the repo-local launcher, which writes a concrete
`external-driver-config.yaml` under the run directory and avoids Hydra config-discovery
edge cases:

```bash
ALPASIM_ROOT=/abs/path/to/alpasim ./.venv/bin/python scripts/run_alpasim_local_external.py \
  --mode print \
  --model token_dagger_iter2_hybrid_clamped \
  --scene-preset fresh_3scene
```

If you want a real run immediately:

```bash
ALPASIM_ROOT=/abs/path/to/alpasim ./.venv/bin/python scripts/run_alpasim_local_external.py \
  --mode both \
  --model token_dagger_iter2_hybrid_clamped \
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

## Repo-Native Batch Path

For the easiest repo-only path on a fresh machine, keep a repo-local HF token in
`.env.alpasim_hf` and use:

```bash
./scripts/run_alpasim_transfer_matrix_repo.sh \
  --mode both \
  --models token_dagger_iter2,token_dagger_iter2_clamped,token_dagger_iter2_axis_constrained_clamped,token_dagger_iter2_hybrid_clamped,token_dagger_srcdecay \
  --scene-presets front_camera_10scene_smoke \
  --matrix-dir runs/alpasim_transfer_matrix_run_10scene \
  --allow-existing-matrix-dir \
  --continue-on-error
```

That wrapper does all repo-side setup in order:

1. validates or repairs the nested AlpaSim checkout
2. bootstraps the AlpaSim env and plugin registry
3. builds `alpasim-base:0.66.0` if it is missing
4. launches the requested matrix

## Oracle Actor-Proxy Ablation

The CoRL diagnostic localizes AlpaSim collision failures to an actor-incomplete
proxy state: the selector logs can show route geometry while seeing no structured
dynamic actors. The falsification gate is an oracle actor-proxy ablation: keep the
learned checkpoint, token library, clamped trajectories, and axis-constrained
selector fixed, but inject actor poses reconstructed from AlpaSim ASL logs.

Build the proxy from an existing matched scene batch:

```bash
PYTHONPATH=alpasim/src/grpc:alpasim/src/utils \
  alpasim/.venv/bin/python scripts/build_alpasim_oracle_actor_proxy.py \
  --run-dir runs/alpasim_axis_constrained_10scene_eval/token_dagger_iter2_axis_constrained_clamped__front_camera_10scene_smoke \
  --output artifacts/alpasim_oracle_actor_proxy_10scene.json \
  --max-actors-per-frame 16
```

The builder reads `rollout.asl` files, projects non-ego actor poses into the ego
frame, estimates relative velocity, and writes timestamp-keyed hazards. The adapter
matches those timestamps at inference time and logs:

- `oracle_actor_proxy_hit`
- `oracle_actor_proxy_count`
- `oracle_actor_proxy_delta_us`
- `oracle_actor_proxy_scene_id`

Run the oracle-gated learned policy with:

```bash
ALPASIM_ROOT=/abs/path/to/alpasim ./.venv/bin/python scripts/run_alpasim_local_external.py \
  --mode both \
  --model token_dagger_iter2_axis_constrained_oracle_actor_clamped \
  --scene-preset front_camera_10scene_smoke \
  --oracle-actor-proxy artifacts/alpasim_oracle_actor_proxy_10scene.json
```

For scene-level paired statistics, use the same model and proxy path through the
batch runner:

```bash
ALPASIM_ROOT=/abs/path/to/alpasim ./.venv/bin/python scripts/run_alpasim_scene_batch.py \
  --mode both \
  --model token_dagger_iter2_axis_constrained_oracle_actor_clamped \
  --scene-preset front_camera_10scene_smoke \
  --oracle-actor-proxy artifacts/alpasim_oracle_actor_proxy_10scene.json \
  --batch-dir runs/alpasim_oracle_actor_proxy_10scene_eval \
  --allow-existing-batch-dir \
  --continue-on-error
```

Interpretation rule: first verify proxy hit rate from `driver/selection-log.jsonl`.
If hit rate is low, the run is an instrumentation failure. If hit rate is high and
collision is unchanged, the bottleneck shifts from proxy reconstruction to the
candidate/controller interface. If collision falls while offroad and wrong-lane stay
controlled, the actor-incomplete proxy diagnosis passes and the next method target is
a learned actor-aware proxy.

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
- `token_dagger_iter2_axis_constrained_clamped`
- `token_dagger_iter2_hybrid_clamped`
- `token_dagger_srcdecay_hybrid`
- `token_dagger_srcdecay_hybrid_clamped`

Supported scene presets:

- `fresh_3scene`
- `front_camera_10scene_smoke`
- `front_camera_30scene_merged`

Notes:

- The learned presets force `model.device=cuda` in the external driver command.
- `ALPASIM_ROOT` is the canonical way to point the repo at an AlpaSim checkout.
- The launcher defaults to `topology=1gpu`; override with `--topology` if you
  want a larger AlpaSim layout.
- `token_dagger_iter2` and `token_dagger_srcdecay` currently share the same
  wizard-side driver family; the differentiator is the external driver config
  and checkpoint path.
- The `*_clamped` presets preserve learned token selection but clamp lateral
  token geometry to `max_lateral_offset_m=2.0`. Use these to test whether the
  AlpaSim failure is caused by unsafe lateral projection rather than learned
  safety calibration.
- `token_dagger_iter2_axis_constrained_clamped` is the positive-method probe. It
  preserves the DAgger top-k preference when independent safety/progress
  constraints pass, rather than collapsing to the scalar Spotlight rank.

## CoRL Evidence Gates

Before rewriting the paper around any method claim, run:

```bash
./.venv/bin/python scripts/audit_corl_evidence_strength.py
./.venv/bin/python scripts/analyze_transfer_predictors.py
```

The current diagnostic artifacts intentionally fail the Best-Paper method gate:
the 10-scene matrix has external tradeoffs but no uniformly dominant
intervention. The decisive next matrix is:

```bash
./.venv/bin/python scripts/run_alpasim_transfer_matrix.py \
  --mode both \
  --scene-presets front_camera_30scene_merged \
  --matrix-dir runs/alpasim_transfer_matrix_30scene_axis \
  --allow-existing-matrix-dir \
  --continue-on-error
```

If that run completes, analyze it with:

```bash
./.venv/bin/python scripts/analyze_alpasim_transfer_matrix.py \
  runs/alpasim_transfer_matrix_30scene_axis \
  --scene-preset front_camera_30scene_merged \
  --output-json artifacts/alpasim_matrix30_axis_analysis.json \
  --output-markdown artifacts/alpasim_matrix30_axis_analysis.md
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
