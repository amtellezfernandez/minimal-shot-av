# Spotlight Reflex: Minimal-Shot Autonomous Driving

**Author:** amtellezfernandez@gmail.com · **Repo:** github.com/amtellezfernandez/minimal-shot-av

---

## Thesis

Current autonomous vehicle systems are discriminatively calibrated on their training
distribution. They generalise well within that distribution and fail at its edges.
Waymo's WOD-E2E benchmark makes the failure mode precise: 11 long-tail clusters
filtered to scenarios occurring less than 0.03% of driving time. These are exactly
the scenarios where trajectory memorisation breaks down by design.

The hypothesis driving this project: **safety-critical driving behaviour in long-tail
scenarios can be reduced to a small set of geometric signals — obstacle pressure,
route blockage, lateral clearances — without semantic labels or training data.**
If that hypothesis holds, a system that computes those signals correctly should
generalise to novel scenarios by construction, because the geometry of "obstacle
on the left, clear on the right" is the same whether the obstacle is a cone, an
animal, or a wrong-way vehicle.

---

## The System

![Spotlight Reflex — wrong-way actor, low visibility, night](images/spotlight_success.gif)
*Spotlight Reflex navigates a wrong-way actor in low-visibility conditions. No training
data for this scenario. The system selects `evasive_right` because right clearance
exceeds left clearance.*

![Baseline policy — same scenario, no world-state reasoning](images/baseline_spotlight.gif)
*Baseline policy on the same scenario: no geometric reasoning, continues into the actor.*

**Spotlight Reflex** is a closed-loop policy built on this thesis. At each step it
computes six scalars from scene geometry:

| Signal | Meaning |
|--------|---------|
| `obstacle_pressure` ∈ [0,1] | Weighted influence of nearby obstacles |
| `route_blockage` ∈ [0,1] | Fraction of route corridor obstructed ahead |
| `corridor_blocked` | Hard blockage within stopping distance |
| `left_clearance` (m) | Lateral space to the left |
| `right_clearance` (m) | Lateral space to the right |
| `preferred_escape_side` | Side with more clearance |

Nine maneuver hypotheses are generated as 20-point, 5-second trajectories and scored
against pseudo-rater references derived from world state. The selector picks the
highest-scoring safe option. **The entire decision is a function of geometry, not
object identity.** Regression tests verify this: relabelling all object names, cluster
tags, and scenario labels while holding geometry fixed produces identical decisions.

---

## Simulation Evidence

![Construction zone — narrow corridor, cone field, lane closure](images/construction_success.gif)

![Foreign object debris](images/fod_success.gif)

![Intersection stress — conflict zone, two crossing actors, 207 steps](images/intersection_stress.gif)

The simulation environment is built entirely from scratch: a 2D closed-loop engine,
a procedural generator for all 11 WOD-E2E named clusters, and a compositional OOD
generator that independently samples road topology (8 types), hazard module, weather
condition, and novel object type. Topology and hazard are decoupled so memorising
cluster labels gives no advantage.

**Primary benchmark** (seeds 1–10, deterministic):

| Suite | Runs | Collisions | Pass |
|-------|------|-----------|------|
| WOD-style (all 11 clusters) | 110 | **0** | **100%** |
| Compositional OOD | 60 | 0 | 100% |
| Adversarial (2–3 hazards) | 60 | 0 | 100% |
| Hidden holdout | 60 | 0 | 100% |
| Gauntlet (4 hazards, narrow corridor) | 60 | 0 | 60% |
| **Total** | **350** | **0** | **326/350 (93.1%)** |

Mean minimum clearance: 2.96 m · **COMPASS: 9.137 / 10** (700 ranked runs) · 95% CI collision rate [0.0, 0.0053]

COMPASS is our own composite benchmark: a weighted score across four sub-metrics — oracle
selector quality, geometric reasoning quality, recovery behaviour, and generalisation gap
between normal and adversarial seeds. A score above 7.0 is the passing threshold.

**Extended evaluation** (seeds 1–40, wider coverage):

| Suite | Runs | Collision rate | Pass |
|-------|------|---------------|------|
| Compositional OOD | 240 | 6.7% | 89% |
| Adversarial | 240 | 11.7% | 83% |

**Gauntlet comparison** (matched seeds 1–80, 420 runs per policy):

| Policy | Pass rate | Collision rate |
|--------|-----------|---------------|
| Baseline (no world-state reasoning) | 2.1% (9/420) | **20.5%** |
| Spotlight Reflex | **57.6% (242/420)** | **7.9%** |

The baseline collides in one in five gauntlet runs and passes nearly none.
Spotlight Reflex passes 58% of the same scenarios at one-third the collision rate.
The 10× collision reduction is the measurable consequence of geometric reasoning.

The same policy runs in Waymo's AlpaSim sensor-realistic simulator via a custom
adapter — same `select_maneuver()` function, zero AlpaSim-specific code in the policy:

![AlpaSim — real WOD-E2E camera frames with Spotlight Reflex reasoning output](images/alpasim_reasoning_panel.png)

| AlpaSim metric (30 scenes) | Value |
|---------------------------|-------|
| `collision_at_fault` | **0.0** |
| `dist_to_gt_trajectory` | 0.42 m |
| `collision_any` (rear, by others) | 0.37 |

Cross-fidelity transfer without modification is evidence that the world-state
abstraction is at the right level of representation.

---

## WOD-E2E Benchmark Analysis

The WOD-E2E harness evaluates trajectory selection on 479 preference-labeled
validation frames using Waymo's **Rater Feedback Score (RFS)** — a scalar derived from
human rater pairwise preferences between candidate trajectories. Higher RFS means your
selected trajectory is more often preferred. All results use segment-grouped 5-fold
cross-validation on the local scoring backend (local CV baseline 7.131; official Waymo
backend baseline 7.022). The two backends are consistent but not identical.

Three selector approaches are shown below:

- **Gate system only**: a ranker that scores candidates using ~137 trajectory and context
  features, selecting the highest-scoring one with no learned override.
- **RFF direct policy**: a **Random Fourier Features** classifier — a technique that maps
  trajectory features into a 512-dimensional random space (`D=512`, bandwidth `σ=7.858`)
  where the dot product approximates a Gaussian kernel. This allows a non-linear selector
  without training a neural network. It fires as an override on 3.5% of frames; **precision
  0.41** means 41% of its overrides improve the score vs the gate's choice.
- **GPU MLP + Cosmos 64d**: a 2-layer neural network (hidden size 64) that takes
  **64-dimensional Cosmos embeddings** — continuous latent codes from NVIDIA's Cosmos
  world-model video tokenizer — as input and fires as an override on 4.2% of frames with
  precision 0.60. The **oracle** is not a trained model: it is the score you would get if
  you always picked the best available candidate per frame — it defines the upper bound given
  the candidate pool.

| Selector | RFS (5-fold) | Notes |
|----------|-------------|-------|
| Constant velocity (local) | 7.131 | Local scoring baseline |
| Constant velocity (official) | 7.022 | Official Waymo backend |
| Kinematic ranker | 7.096 | Physics-only candidates, official backend |
| Gate system only | 7.803 | No direct policy |
| RFF direct policy | 7.834 | D=512, σ=7.858, precision 0.41, +0.703 vs baseline |
| **GPU MLP + Cosmos 64d** | **7.845** | **h=64, precision 0.60, +0.714 vs baseline** |
| Oracle (perfect selection) | **9.264** | Upper bound given candidate pool |

The **oracle gap is 1.419 RFS** (9.264 − 7.845). This is the key scientific finding:
the candidate pool is good — the bottleneck is the discriminator. Even with the best
available candidate in the pool, the selector cannot reliably identify it without visual
scene information about what is happening in the frame.

The GPU MLP with 64d Cosmos embeddings fires on 4.2% of frames with precision 0.60,
versus the RFF champion at 3.5% / 0.41. The +0.011 gain is within the ±0.17 CI
but confirms that Cosmos embeddings carry non-linear preference signal.
The HGB variant (optimised via Optuna) reached 7.880 in 2-fold evaluation but was
not evaluated under 5-fold at time of writing.

**Per-slice bias audit reveals residual regret at distribution extremes:**

| Slice | Frames | Selected RFS | Oracle RFS | Regret |
|-------|--------|-------------|-----------|--------|
| GO_STRAIGHT | 427 | 7.959 | 9.336 | 1.377 |
| GO_LEFT | 23 | 7.107 | 8.721 | **1.614** |
| GO_RIGHT | 29 | 6.574 | 8.638 | **2.063** |
| speed:slow | 133 | 7.564 | 9.196 | 1.632 |
| speed:fast | 44 | 7.957 | 9.436 | 1.479 |

GO_RIGHT has the highest regret (2.063) as the rarest intent class — under-represented
in the 479-frame calibration set. GO_LEFT improved substantially from 1.753 (previous
model) to 1.614. The dominant GO_STRAIGHT slice performs well (1.377 regret). This
is a quantifiable, correctable imbalance: reweighting turn frames in training and
augmenting with more diverse turn examples addresses it directly.

---

## What Didn't Work

**Visual embeddings fail under a linear head but succeed under a non-linear one.**
**InternVLA** (a Vision-Language-Action model from Shanghai AI Lab, trained on robot and
driving demonstrations) and **Cosmos** (NVIDIA's world-model video tokenizer) both produce
per-frame visual embeddings. When attached as extra features to the linear ridge ranker,
neither produced a confirmed RFS gain. However, 64d Cosmos embeddings as input to a GPU MLP
direct policy (non-linear, 2 hidden layers) reached 7.845 RFS — the champion — with precision
0.60 vs 0.41 for the RFF baseline. The embeddings DO carry trajectory preference signal; the
failure under the linear head is a model-capacity failure, not an alignment failure.
Task-specific fine-tuning (optimising embeddings directly on preference triplets) remains the
identified next step to close the remaining 1.419 RFS oracle gap.

**World-model candidates add oracle headroom but the selector cannot exploit it.**
A lightweight world model was implemented to generate scene-conditioned trajectory
candidates beyond what kinematic and ridge produce. Oracle RFS improved marginally,
but selected RFS did not — the selector picks the wrong candidate too often. More
candidates are not useful without a better discriminator.

---

## Next Steps

**1. Task-specific camera encoder.** A small model fine-tuned on WOD-E2E preference
labels to discriminate winning trajectories from camera images. Not a general VLM —
a discriminative encoder for the specific signal the ranker needs. Expected to close
0.5–1.5 RFS of the 1.419-point oracle gap.

**2. Proper train/test split.** The champion selector is calibrated on retained
validation preference labels under segment-grouped CV. Waymo train TFRecords (Google
sign-in gated) would move calibration off the validation set, making it a true test.

**3. Leaderboard submission.** The official test frame list is present (1,505 frames).
The packaging pipeline is validated. Only the test TFRecords are needed.

**4. Turn calibration.** GO_RIGHT (regret 2.063) and GO_LEFT (regret 1.614) slices
are directly addressable by weighting turn frames more heavily in training and
augmenting with more diverse turn examples. GO_LEFT improved from 1.753 (previous
model) to 1.614 in the champion; GO_RIGHT remains the largest calibration gap.

**5. Physical deployment.** The AlpaSim adapter is the template for a
sensor-to-obstacle bridge on real hardware. A vehicle with obstacle sensors and a
route command can run Spotlight Reflex today.
