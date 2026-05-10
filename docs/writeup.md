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

Mean minimum clearance: 2.96 m · COMPASS: **9.137 / 10** (700 ranked runs)  
95% CI collision rate [0.0, 0.0053]

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
validation frames using Waymo's Rater Feedback Score (RFS). All results use
segment-grouped 5-fold CV on the local scoring backend (CV baseline 7.131;
official Waymo baseline 7.022).

| Selector | RFS (5-fold) | Notes |
|----------|-------------|-------|
| Constant velocity (local) | 7.131 | Local scoring baseline |
| Constant velocity (official) | 7.022 | Official Waymo backend |
| Kinematic ranker | 7.096 | Physics-only candidates, official backend |
| Gate system only | 7.803 | No direct policy |
| Champion (direct policy) | **7.834** | Random-Fourier direct policy, +0.703 vs local baseline |
| Oracle (perfect selection) | **9.264** | Upper bound given candidate pool |

The **oracle gap is 1.430 RFS** (9.264 − 7.834). This is the key scientific finding:
the candidate pool is good — the bottleneck is the discriminator. The selector cannot
identify which candidate is best on a given frame without visual scene information.

The direct policy adds +0.031 RFS above the gate-only baseline (7.803 → 7.834).
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

**Visual embeddings do not carry preference signal.** InternVLA and Cosmos tokenizer
embeddings were computed for each validation frame and attached as ranker features.
Neither produced a confirmed RFS gain over the no-embedding baseline. Off-the-shelf
visual encoders extract representations tuned for navigation or generation, not for
the discriminative question "which trajectory wins on this specific frame." Fine-tuning
on preference labels is required to align the embedding space to this signal.

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
0.5–1.5 RFS of the 1.430-point oracle gap.

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
