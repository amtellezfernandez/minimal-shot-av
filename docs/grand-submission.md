# Grand Commission Submission

## Claim

Spotlight Reflex is a closed-loop driving policy that navigates unfamiliar long-tail
scenarios by making world-state reasoning and maneuver enumeration explicit — without
fine-tuning on any AV dataset, without route memorisation, and without
leaderboard-specific supervised training.

The WOD-E2E benchmark harness is supporting infrastructure: a preference-calibrated
trajectory selector reaching **7.845 RFS** (5-fold CV, local backend · +0.714 over
local baseline 7.131 · GPU MLP + 64d Cosmos embeddings, precision 0.60) on 479
validation frames under segment-grouped cross-validation. RFF champion: 7.834.
Optuna 2-fold peak: 7.880 (not directly comparable to 5-fold). Declared as
development analysis, not strict zero-shot deployment.

## Submission Contents

- GitHub repo: this codebase, with `README.md` and `models/DECLARATION.md`
- Architecture write-up: `docs/architecture-deep-dive.md`
- Slide deck: `docs/presentation.md`
- WOD-E2E pipeline and failure analysis: `docs/wod-e2e-system-walkthrough.md`
- Analysis notebook: `notebooks/wod_e2e_analysis.ipynb`

## Demo Commands

```bash
# Spotlight scenario — wrong-way actor, low visibility
uv run --no-sync python scripts/run_demo.py \
  --policy spotlight-reflex \
  --scenario-cluster spotlight \
  --seed 3 \
  --artifacts-dir artifacts/grand_spotlight_demo

# Intersection stress case
uv run --no-sync python scripts/run_demo.py \
  --policy spotlight-reflex \
  --scenario-cluster intersection \
  --seed 3 \
  --artifacts-dir artifacts/grand_intersection_stress_seed3

# Baseline policy comparison
uv run --no-sync python scripts/run_demo.py \
  --policy baseline \
  --scenario-cluster spotlight \
  --seed 3 \
  --artifacts-dir artifacts/grand_baseline_spotlight_demo
```

## Key Evidence

| Claim | Value | Source |
|-------|-------|--------|
| OOD rollouts with 0 collisions | 350 | `benchmarks/current/spotlight_reflex_procedural_wod.json` |
| COMPASS pass rate | 326 / 350 (93.1%) | Same |
| Gauntlet pass rate | 36 / 60 (60%) | Same |
| WOD-E2E constant-velocity baseline | 7.022 RFS | 479-frame validation CV |
| WOD-E2E local baseline (const-vel) | 7.131 RFS | 479-frame val, local backend |
| WOD-E2E champion (GPU MLP + Cosmos 64d) | **7.845 RFS** | 5-fold CV, local backend — commit `6f1232b` |
| WOD-E2E RFF direct policy | 7.834 RFS | 5-fold CV, local backend |
| WOD-E2E Optuna peak (HGB, 2-fold) | 7.880 RFS | 2-fold only — not comparable to 5-fold |
| WOD-E2E oracle (best candidate per frame) | **9.264 RFS** | 5-fold, local backend · oracle gap 1.419 |
| AlpaSim collision at fault | 0.0 | `benchmarks/current/spotlight_reflex_alpasim_front_camera_30scene_merged.json` |

Packaged submission archives for validation-set candidates are in
`artifacts/wod_e2e_submission_matrix/` (includes `hgb_selector_v3.tar.gz`).

## Boundaries

- The WOD-E2E selector is calibrated on retained validation preference labels
  under segment-grouped CV. This is development analysis, not strict zero-shot
  generalisation.
- No camera perception in the active model path — the WOD-E2E result is ego-history
  only. Camera encoder is the identified next step.
- No production AV stack. The simulation environment is 2D and abstract;
  AlpaSim integration is at trajectory plugin level.
- Test TFRecords are not downloaded. The official test frame list IS present
  (`data/waymo/e2e/submission_frames/test_frames.json`, 1,505 frames). Generating
  test predictions requires only the test TFRecords.
