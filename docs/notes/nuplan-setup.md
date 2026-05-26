# nuPlan Setup

This repo now supports a direct nuPlan setup path on a fresh machine without
manually creating `.venv-nuplan` or downloading the full 8.6 GB public mini
archive.

## Environment

```bash
./scripts/bootstrap_nuplan_env.sh
```

That script:

- creates `./.venv` with `uv`
- installs `minimal-shot-av[nuplan]`
- verifies the nuPlan DB query imports used by this repo

## Public mini DB subset

```bash
./.venv/bin/python scripts/fetch_nuplan_public_mini.py --db-count 5
```

This extracts a small real public nuPlan mini subset into:

```text
workspace/nuplan/public_mini/
```

The extractor reads directly from Motional's public `nuplan-v1.1_mini.zip` via
ranged requests and writes a `manifest.json` alongside the extracted DB files.
It avoids downloading the full archive when you only need a smoke-test bundle.

## Public maps

```bash
./.venv/bin/python scripts/fetch_nuplan_public_maps.py
```

This extracts the public nuPlan maps bundle into:

```text
workspace/nuplan/maps/
```

The closed-loop bridge uses `workspace/nuplan/maps/maps` as the nuPlan map root, because the public archive
contains a top-level `maps/` directory.

## Rollout and selector smoke runs

```bash
PYTHONPATH=src ./.venv/bin/python scripts/run_nuplan_maneuvertoken_rollout.py \
  --nuplan-db-file workspace/nuplan/public_mini \
  --limit 50 \
  --output-json artifacts/corl2027/nuplan_mini_rollout50.json \
  --output-markdown artifacts/corl2027/nuplan_mini_rollout50.md

PYTHONPATH=src ./.venv/bin/python scripts/train_nuplan_maneuvertoken_selector.py \
  --nuplan-db-file workspace/nuplan/public_mini \
  --limit 200 \
  --epochs 50 \
  --hidden-dim 16 \
  --output artifacts/corl2027/nuplan_mini_selector200.json
```

## One-shot replay study

```bash
PYTHONPATH=src ./.venv/bin/python scripts/run_nuplan_public_replay_study.py \
  --db-count 20 \
  --scene-limit 250 \
  --output-dir artifacts/corl2027/nuplan_public_replay_study
```

This command:

- extracts a larger public DB bundle
- runs the ManeuverToken rollout
- runs the replay audit
- writes `bundle_manifest.json`, `rollout.json/md`, and `replay.json/md`

## Closed-loop bridge

```bash
PYTHONPATH=src ./.venv/bin/python scripts/audit_nuplan_closed_loop_bridge.py \
  --input-json artifacts/corl2027/nuplan_public_replay_study/replay.json \
  --maps-root workspace/nuplan/maps/maps \
  --output-json artifacts/corl2027/nuplan_public_replay_study/closed_loop_bridge.json \
  --output-markdown artifacts/corl2027/nuplan_public_replay_study/closed_loop_bridge.md
```

This runs nuPlan's simulation loop on the decisive subset:

- replay-infeasible proxy-safe scenes
- matched replay-safe proxy-safe scenes

## Full dataset

If you already have the full nuPlan DB tree on disk, skip the public-mini fetch
step and point the commands at the dataset root instead:

```bash
PYTHONPATH=src ./.venv/bin/python scripts/run_nuplan_maneuvertoken_rollout.py \
  --nuplan-data-root /path/to/nuplan/db/root \
  --limit 50
```
