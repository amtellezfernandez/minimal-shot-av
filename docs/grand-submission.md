# Grand Commission Submission

## Claim

Spotlight Reflex is a closed-loop driving policy that navigates unfamiliar long-tail
scenarios by making world-state reasoning and maneuver enumeration explicit — without
fine-tuning on any AV dataset, without route memorisation, and without
leaderboard-specific supervised training.

The **WOD-E2E** (Waymo Open Dataset End-to-End Driving) benchmark harness is supporting
infrastructure. It measures trajectory selection quality using **RFS (Rater Feedback Score)**
— a score derived from pairwise human rater preferences between candidate trajectories on
479 real Waymo validation frames. Higher RFS means the selected trajectory is more often
preferred by raters.

Our selector reaches **7.845 RFS** — the champion result, using a **GPU MLP** (a 2-layer
neural network, hidden size 64) that takes **64-dimensional Cosmos embeddings** (visual
latents from NVIDIA's Cosmos world-model tokenizer) and fires as a direct override on 4.2%
of frames with **precision 0.60** (meaning 60% of its overrides improve the score vs not
overriding). All results use **5-fold segment-grouped cross-validation**: the 479 frames are
split into 5 groups such that frames from the same driving segment stay together, preventing
data leakage; results are averaged across 5 held-out test folds. The **local backend** means
we compute RFS ourselves (local CV baseline 7.131); the official Waymo backend gives 7.022 for
the same constant-velocity baseline.

The **RFF champion** (7.834) is the same architecture but with a **Random Fourier Features**
gate — a non-linear classifier that approximates a Gaussian kernel by projecting features into
a random 512-dimensional space. It fires on 3.5% of frames with precision 0.41. The GPU MLP
(7.845) is +0.011 above this baseline, within the ±0.17 statistical CI.

The **Optuna 2-fold peak** of 7.880 was found by automated hyperparameter search (Optuna is
a Bayesian optimisation library) but evaluated under 2-fold CV only — a noisier estimate that
uses half as many held-out folds and is not comparable to the 5-fold protocol.

Declared as development analysis, not strict zero-shot deployment.

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

RFS (Rater Feedback Score) is Waymo's trajectory quality metric derived from human rater
pairwise preferences. All WOD-E2E numbers use the local scoring backend evaluated on 479
preference-labeled validation frames under 5-fold segment-grouped cross-validation unless
noted. "Oracle" means: if we always pick the best available candidate per frame, what is the
RFS? COMPASS is our own composite benchmark score across simulation runs (passing threshold
7.0 out of 10).

| Claim | Value | Source |
|-------|-------|--------|
| OOD rollouts with 0 collisions | 350 | `benchmarks/current/spotlight_reflex_procedural_wod.json` |
| COMPASS composite score (simulation) | 326 / 350 (93.1%) pass, 9.137/10 score | Same |
| Gauntlet pass rate (4 simultaneous hazards) | 36 / 60 (60%) | Same |
| WOD-E2E constant-velocity baseline (official) | 7.022 RFS | Official Waymo backend |
| WOD-E2E constant-velocity baseline (local) | 7.131 RFS | Local backend, our CV reference |
| WOD-E2E champion: GPU MLP + Cosmos 64d | **7.845 RFS** | 5-fold CV, local backend — commit `6f1232b` |
| WOD-E2E RFF direct policy (Random Fourier Features gate) | 7.834 RFS | 5-fold CV, local backend |
| WOD-E2E Optuna peak (HGB, 2-fold only) | 7.880 RFS | 2-fold only — not comparable to 5-fold |
| WOD-E2E oracle (best candidate per frame) | **9.264 RFS** | 5-fold, local backend |
| Oracle gap (champion vs oracle) | 1.419 RFS | 9.264 − 7.845 |
| AlpaSim collision at fault (sensor-realistic) | 0.0 | `benchmarks/current/spotlight_reflex_alpasim_front_camera_30scene_merged.json` |

Packaged submission archives for validation-set candidates are in
`artifacts/wod_e2e_submission_matrix/` (includes `hgb_selector_v3.tar.gz`).

## Boundaries

- **Development analysis, not zero-shot.** The WOD-E2E selector is trained on the
  validation preference labels it is evaluated on — under cross-validation (frames from
  the same segment are held out together), but still the validation set. Moving training
  to the Waymo train split would make this a true test-set generalisation claim.
- **No camera perception.** The WOD-E2E model processes only ego velocity history, speed,
  acceleration, and route intent. It cannot see the camera images. This is the identified
  bottleneck — a preference-aligned visual encoder is the next step.
- **2D abstract simulation.** The simulation environment is built from scratch; it is 2D
  with structured obstacles, not a physics engine or sensor simulation. AlpaSim integration
  runs the same policy in a sensor-realistic environment but is at the trajectory-plugin level.
- **Test TFRecords not downloaded.** The official test frame list is present
  (`data/waymo/e2e/submission_frames/test_frames.json`, 1,505 frames). The submission
  packaging pipeline is complete and validated. Only the test TFRecords (Waymo Google Drive,
  sign-in gated) are missing to generate a test-set submission.
