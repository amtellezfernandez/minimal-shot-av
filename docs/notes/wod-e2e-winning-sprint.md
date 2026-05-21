# WOD-E2E Winning Sprint

This is the plan for turning the repo into a real WOD-E2E leaderboard
contender. The goal is not a polished SoTA Commission story. The goal is to
produce a valid Waymo E2E submission that can challenge the active leaderboard.

## Current Reality

- The local workspace has WOD-E2E validation only:
  `workspace/waymo_open_dataset_end_to_end_camera_v_1_0_0/val`.
- The train split, test split, and official test frame list are not present.
- The current best repo path is still mostly ego-history, route-intent, ridge
  residual candidates, and a structured selector.
- The current best local validation-CV artifact is
  `artifacts/cache/wod_neural_ensemble_su_r3_three_models_top1_familycal_m6_veto_r100_p25_temp062.json`
  at `7.8052` selected RFS / `8.1302` normalized RFS. This uses a diverse
  three-model neural proposal ensemble with only the top proposal from each
  model, plus a temperature-0.62 listwise softmax selector, speed/source/family
  calibration, learned-source veto, and ridge-3 safety-utility postprocessing.
  Route-target routing and larger per-model top-k did not improve this setup in
  the current 479-frame validation CV contract.
- This is stronger than the previous calibrated linear selector result
  (`7.7061` selected RFS), but it is still not a hidden-test leaderboard result
  and is not enough to claim leaderboard-winning scene understanding.
- The candidate pool has useful oracle headroom, but the selector does not
  reliably exploit it and worsens some worst-slice regret.

The benchmark-winning problem is therefore:

> learn scene-conditioned proposal generation and RFS-aligned proposal
> confidence, then submit the highest expected-RFS 5-second trajectory for each
> hidden test frame.

## Hard Gates

These are required before a real leaderboard push is possible.

1. Get access to WOD-E2E train shards through one of:
   external SSD, cloud VM, cloud bucket, or small rotating shard batches.
2. Get access to WOD-E2E test shards through the same streaming/batch workflow.
3. Download or reconstruct the official test frame list required for
   submission packaging.
4. Verify parser compatibility against all splits through the existing official
   proto path.
5. Run `scripts/audit_wod_e2e_readiness.py` and
   `scripts/audit_wod_reproducibility.py` before packaging.

Validation-only work can improve the system, but it cannot produce a real
Waymo leaderboard submission.

## Storage Strategy

Do not try to keep the full raw WOD-E2E dataset on this machine. The local
validation split alone is about 226 GB. The winning workflow must be
streaming-first:

```text
raw shard batch on external/cloud storage
    -> compact frame/feature cache
    -> train checkpoint or prediction shard
    -> raw shard batch can be removed
```

Keep locally:

- repo code;
- official protos/parser environment;
- shard manifests;
- compact ego/future caches;
- visual feature caches, not raw JPEG TFRecords;
- model checkpoints;
- validation reports;
- test prediction JSONL/proto outputs.

Do not keep locally:

- all train TFRecords;
- all test TFRecords;
- decoded camera image dumps;
- duplicated validation copies.

The immediate local exporter supports shard-window processing:

```bash
PYTHONPATH=.wod-protos .venv-wod/bin/python \
  scripts/export_wod_neural_training_frames.py \
  --train-dir /mnt/wod-e2e/train \
  --shard-start 0 \
  --max-shards 8 \
  --output artifacts/cache/wod_train_shards_0000_0007.jsonl
```

Then move the raw shard window off the laptop, process the next window, and
append/train from compact caches. For visual models, the equivalent cache should
store frozen camera embeddings or raster features, not JPEG payloads.

## Target Architecture

The current ridge/ranker path should become the fallback, not the main method.

```text
camera encoder + ego/status encoder
    -> temporal scene state
    -> K learned trajectory proposals
    -> RFS/confidence head
    -> optional recovery/counterfactual proposal refinement
    -> highest expected-RFS trajectory
```

The closest competitive target is a compact version of the top method pattern:

- UniPlan/RAP-style anchored proposal generation.
- DiffusionLTF-style data curriculum and proposal diversity.
- Swin-Trajectory-style fast visual waypoint decoder.
- Poutine-style preference alignment, but numeric rather than text-first.

## Fastest Competitive Path

### 1. Build The Train-Scale Proposal Model

Train on WOD-E2E train futures, not only validation preferences.

Inputs:

- front three cameras first; all eight cameras as a later ablation;
- past ego trajectory, velocity, acceleration, yaw/curvature proxy;
- route intent;
- camera intrinsics/extrinsics only if used by the visual encoder.

Outputs:

- K trajectory proposals shaped `(K, 20, 2)`;
- confidence or expected-RFS logit per proposal;
- optional uncertainty/regret score.

The first implementation should be small and fast:

- frozen or lightly tuned image encoder;
- ego/status MLP;
- waypoint-query transformer or GRU decoder;
- anchor-residual heads.

### 2. Add RFS-Aligned Confidence

Use validation preference labels only inside segment-grouped CV.

Train the confidence head to prefer candidates with higher official RFS. Report:

- selected RFS;
- oracle RFS;
- regret;
- top-1 oracle match;
- worst-slice regret;
- selected source mix.

Promotion gate:

- at least `+0.10` selected official-RFS gain over the current full-validation
  baseline on the same 479-frame segment-grouped CV contract;
- no worse worst-slice regret;
- no validation leakage across segment groups.

### 3. Create Counterfactual Recovery Data

The strongest leaderboard pattern is not just bigger models. It is better data.
Generate recovery/counterfactual training examples from logged WOD train
trajectories:

- lateral offsets that must recover to lane center;
- speed perturbations that must stop/yield smoothly;
- route-intent perturbations where the valid maneuver changes;
- obstacle/occlusion tags when visual evidence supports them.

These should train the proposal model to recover from rare states rather than
imitate only the logged future.

### 4. Visual Scene Critic

The current repo is mostly blind to the scene. That will not win WOD-E2E.

Add a visual critic that scores whether a proposal is scene-compatible:

- free-space / blocking-object likelihood in front cameras;
- red-light / construction / pedestrian / debris cues where available;
- route consistency;
- slow-speed and turning-case confidence.

This critic can be frozen-feature and lightweight, but it must use pixels.

### 5. Submission Matrix

For hidden test, submit a small pre-registered matrix:

- fallback ridge/ranker baseline;
- visual proposal model;
- visual proposal model plus RFS confidence;
- counterfactual-recovery model;
- ensemble of the best two or three seeds.

Every `.tar.gz` must be tied to:

- git SHA;
- model artifact hash;
- data splits used;
- validation report;
- exact command.

## Score Targets

Current public top range is about `8.04` RFS overall. A realistic sprint target:

- `7.7`: competitive but not winning;
- `7.9`: serious top-table result;
- `8.05+`: leaderboard-winning territory.

The likely path to `8.05+` is not selector tuning. It is train-scale visual
proposal generation plus preference-aligned confidence and recovery data.

## Immediate Commands After Data Is Present

```bash
PYTHONPATH=.wod-protos .venv-wod/bin/python scripts/check_wod_e2e_parser.py
PYTHONPATH=.wod-protos .venv-wod/bin/python scripts/audit_wod_e2e_readiness.py
PYTHONPATH=.wod-protos .venv-wod/bin/python scripts/export_wod_neural_training_frames.py
PYTHONPATH=.wod-protos .venv-wod/bin/python scripts/train_wod_neural_trajectory_model.py
PYTHONPATH=.wod-protos .venv-wod/bin/python scripts/run_wod_breakthrough_experiments.py
```

Do not upload hidden-test submissions until validation promotion gates pass.
