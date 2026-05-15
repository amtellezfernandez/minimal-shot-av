# Spotlight Reflex: Minimal-Shot Autonomous Driving

A closed-loop driving policy that reasons through unfamiliar long-tail scenarios
without fine-tuning on any AV dataset.

---

![Spotlight Reflex navigating a wrong-way actor at night](docs/images/spotlight_success.gif)

*Spotlight Reflex — wrong-way actor, low visibility, night. Selects `evasive_right`,
clears the actor, recovers to lane centre. No training data for this scenario.*

![Baseline policy failing the same scenario](docs/images/baseline_spotlight.gif)

*Baseline policy — no world-state reasoning, drives directly into the actor.*

---

## What This Is

A minimal-shot autonomy prototype for Waymo's WOD-E2E long-tail benchmark, built
entirely from scratch. Two independent tracks:

**Spotlight Reflex (simulation track):** A closed-loop policy built on six geometric
world-state scalars — obstacle pressure, route blockage, lateral clearances — and nine
maneuver candidates scored against trust-region references. The decision is a function
of geometry only, not object identity. Runs in a custom 2D simulator and in Waymo's
sensor-realistic AlpaSim with the same code.

**WOD-E2E harness (benchmark track):** A trajectory selector trained on Waymo's 479
preference-labeled validation frames using segment-grouped 5-fold cross-validation.
Candidates come from kinematic physics, ridge regression, and temporal ego-history
models; the selector picks among them using ~137 features plus visual embeddings
(NVIDIA Cosmos, 64d) fed to a GPU MLP direct policy. Best result: **7.845 RFS**.

The two tracks are deliberately isolated — simulator metrics are never used for WOD
model selection.

---

## More Scenarios

![Construction zone](docs/images/construction_success.gif)

*Construction zone — narrow corridor (5.2 m), cone field, lane closure. `nudge_right`.*

![Intersection stress](docs/images/intersection_stress.gif)

*Intersection stress — two crossing actors, different timings, 207 steps.*

![AlpaSim sensor input and reasoning output](docs/images/alpasim_reasoning_panel.png)

*AlpaSim — real WOD-E2E front-camera frames alongside the adapter's reasoning output.*

---

## Results

**Simulation — 350 rollouts, 0 collisions:**

| Suite | Runs | Pass |
|-------|------|------|
| WOD-style (all 11 clusters) | 110 | 100% |
| Compositional OOD | 60 | 100% |
| Adversarial (2–3 hazards) | 60 | 100% |
| Hidden holdout | 60 | 100% |
| Gauntlet (4 hazards, 3.5 m corridor) | 60 | 60% |

COMPASS: **9.137 / 10** (700 ranked runs) · 95% CI collision rate [0.0, 0.0053]

**Gauntlet vs baseline (same 420 scenarios, matched seeds):**

| Policy | Pass rate | Collision rate |
|--------|-----------|---------------|
| Baseline (no world-state reasoning) | 2.1% | 20.5% |
| Spotlight Reflex | **57.6%** | **7.9%** |

**WOD-E2E — 5-fold segment-grouped CV on 479 validation frames:**

| Selector | RFS | Folds |
|----------|-----|-------|
| Waymo baseline (official) | 7.022 | — |
| Local baseline | 7.131 | — |
| Gate-only | 7.803 | 5 |
| RFF direct policy | 7.834 | 5 |
| **GPU MLP + Cosmos 64d (champion)** | **7.845** | **5** |
| HGB Optuna peak (2-fold only, not comparable) | 7.880 | 2 |
| Oracle (perfect selector) | 9.264 | 5 |

Oracle gap: **1.419 RFS** — the right candidate exists in the pool but the selector
cannot identify it without visual scene information.

**AlpaSim:** same policy, sensor-realistic, `collision_at_fault: 0.0`, `dist_to_gt: 0.42 m`

---

## Scope and Limitations

- **Development analysis, not zero-shot.** The WOD-E2E selector is trained on the same validation preference labels it is evaluated on — under segment-grouped cross-validation to prevent data leakage, but still the validation set. Moving training to the Waymo train split would make this a true test-set generalisation claim.
- **No camera perception.** The WOD-E2E model sees only ego velocity history, speed, acceleration, and route intent — not the camera images. This is the identified bottleneck; a preference-aligned visual encoder is the next step.
- **2D abstract simulation.** The simulator is 2D with structured obstacles, not a physics engine. AlpaSim integration runs the same policy in a sensor-realistic environment but at the trajectory-plugin level.
- **Test set not evaluated.** The submission packaging pipeline is complete. The test TFRecords (Waymo Google Drive, sign-in gated) are the only missing piece for a test-set submission.

---

## Documentation

Two primary references — each focused on one track with no repeated content:

- **[`docs/simulation.md`](docs/simulation.md)** — Policy architecture (6 scalars,
  9 candidates, trust-region scoring, reference rules), simulator we built (11 WOD
  clusters, compositional OOD generator, 8 actor models, COMPASS), AlpaSim integration
  (4-channel adapter), results, and failure modes.

- **[`docs/wod-e2e-system-walkthrough.md`](docs/wod-e2e-system-walkthrough.md)** —
  WOD-E2E data and frame structure, candidate generation pipeline, selector training,
  cross-validation protocol, performance results, and a full external-tools section
  (Cosmos architecture, InternVLA architecture, Optuna TPE algorithm, RFF derivation,
  GPU MLP architecture).

- **[`docs/presentation.tex`](docs/presentation.tex)** — LaTeX Beamer slide deck
  (`pdflatex docs/presentation.tex` or paste into Overleaf).

---

## Quickstart

```bash
# Known repo bootstrap: creates .venv with the AlpaSim extra via uv.
./scripts/bootstrap_alpasim_env.sh

# Single demo rollout — wrong-way actor scenario
uv run --no-sync python scripts/run_demo.py \
  --policy spotlight-reflex \
  --scenario-cluster spotlight \
  --seed 3 \
  --artifacts-dir artifacts/demo_spotlight

# Full 350-rollout evaluation
uv run --no-sync python scripts/evaluate_scenarios.py \
  --policy spotlight-reflex \
  --suite all \
  --seed-start 1 \
  --seed-end 10 \
  --output-dir artifacts/eval_all

# Geometry invariance regression test
uv run --no-sync python scripts/run_tests.py --quick
```

Available clusters: `construction` · `intersection` · `pedestrian` · `cyclist`
· `cut-in` · `foreign object debris` · `special vehicle` · `spotlight` · `others`

For AlpaSim specifically, keep the checkout path explicit:

```bash
export ALPASIM_ROOT=/abs/path/to/alpasim
./scripts/bootstrap_alpasim_env.sh
./.venv/bin/python scripts/run_alpasim_local_external.py \
  --mode print \
  --model token_dagger_iter2_hybrid_clamped \
  --scene-preset fresh_3scene
```
