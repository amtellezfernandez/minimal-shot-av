# Minimal-Shot Autonomy: Submission Write-Up

**Author:** amtellezfernandez@gmail.com  
**Repo:** github.com/amtellezfernandez/minimal-shot-av  
**Track:** Grand Commission (architecture) + Minor Commission (simulation environment)

---

## Motivation

Autonomous vehicles fail at the edges of their training distribution. Waymo's WOD-E2E
dataset makes this precise: its 11 long-tail clusters — construction zones, erratic
pedestrians, debris, animals, wrong-way actors — are filtered to scenarios that occur
less than 0.03% of driving time. These are exactly the situations where a system that
has memorised typical trajectories will confidently do the wrong thing.

The hypothesis driving this project: if you make world-state and maneuver reasoning
explicit, the system should generalise to novel scenes without having seen them. A
trajectory that avoids obstacles, respects corridor geometry, and maintains progress
is correct whether the obstacle is a cone, a fallen tree, or a sheep.

---

## What I Built

**Spotlight Reflex** is a closed-loop policy that operates by enumeration rather than
imitation. At each step it computes six scalar fields from the scene geometry:
obstacle pressure, route blockage, whether the corridor is hard-blocked, left and right
clearance, and a preferred escape side. It then generates nine named maneuver candidates
(stop, crawl, maintain, slow yield, nudge left/right, evasive left/right, lane recover)
as 20-point 5-second trajectories, scores each candidate against pseudo-rater references
at the 3-second and 5-second marks, and selects the highest-scoring safe option.

No AV-dataset fine-tuning. No route memorisation. Every decision is derived from
occupancy and route geometry, not object category labels or cluster tags — the
regression tests verify this explicitly by relabelling all object names while holding
geometry fixed.

The simulation environment (Minor Commission) was built entirely from scratch: a 2D
closed-loop simulator, a procedural generator for all 11 WOD-E2E cluster types, a
compositional OOD generator that independently samples topology, hazard, weather, and
novel object type, a COMPASS benchmark system, and a SOTIF-aligned evidence framework.
The same Spotlight Reflex policy was then deployed as an AlpaSim trajectory plugin via
a custom adapter that bridges route commands, camera brightness, and ego dynamics into
the simulator's world-state representation.

For the WOD-E2E model track, I built a candidate generation and selection pipeline:
kinematic candidates from ego history, ridge learned candidates from 31 trajectory
features under segment-grouped cross-validation, and a temporal ridge model using
ego-history trends. The final selector is a stability-selected HistGradientBoosting
classifier with ~137 features including speed bins, intent dummies, and source
interaction terms.

---

## What Worked

**Simulation evidence:** 350 OOD rollouts, zero collisions, 93.1% COMPASS benchmark
pass rate across all suites. The gauntlet suite (synchronised multi-hazard,
narrow corridor, strict progress gates) passes at 60% — deliberately not saturated,
because a system that always passes isn't being tested hard enough.

**WOD-E2E:** The HGB stability-selected ranker reaches 7.880 RFS on the 479-frame
validation preference contract, compared to 7.022 for constant velocity (+0.86 RFS).
The oracle across all candidates is 9.068, so the selection gap is 1.19 RFS — showing
the candidate pool is good but the discriminator is the bottleneck.

**AlpaSim deployment:** The adapter successfully plugs Spotlight Reflex into Waymo's
sensor-realistic simulator. Route command → sigmoid lane geometry, camera brightness →
visibility risk, ego dynamics → braking risk, and structured upstream hazards → obstacle
and actor objects. Every prediction includes a full reasoning JSON so reviewers can
inspect why a trajectory was chosen.

**Key finding:** Temporal ridge candidates (ego-history trends) win selection on ~60%
of WOD-E2E frames. Most highway frames are predictable from recent motion; the selector
correctly identifies this.

---

## What Didn't Work

**Visual blindness is the hard ceiling.** The current WOD-E2E path has no camera
perception — it processes only ego history, speed, and route intent. It cannot see
pedestrians crossing, debris in the road, or red lights. The system sometimes gets the
right answer by coincidence (if the ego was already braking before the frame); it fails
when recent history gives no signal. This is the honest reason the WOD result sits below
the 8.05 leaderboard top.

**Visual embeddings didn't transfer.** InternVLA and Cosmos tokenizer embeddings were
computed for the camera frames and attached as features to the ranker. Neither produced
a confirmed RFS gain. The embedding spaces are not aligned to the discriminative signal
the ranker needs: which candidate trajectory wins on this specific frame.

**Selector mis-calibration on turns.** The worst slice is GO_LEFT frames (23 frames):
6.782 RFS selected vs 8.535 oracle, 1.75 RFS regret. The selector over-relies on
kinematic candidates for left turns (87% selected vs 57% oracle). The training
distribution is dominated by straight-ahead frames, and that prior bleeds into turn
decisions.

**World-model candidates added oracle headroom (+0.05 RFS) but the selector couldn't
exploit it reliably** — it picked the wrong candidate too often. More candidates are not
useful without a better discriminator.

---

## Where the Prize Goes

The gap is clear: the bottleneck is scene understanding, not candidate quality.

With the prize, I would build a lightweight camera encoder trained directly on WOD-E2E
preference labels — not a general-purpose VLM, but a small model fine-tuned to answer
"which trajectory wins on this frame" from the camera images. Expected gain: +0.5–1.5
RFS based on the oracle gap in the visual-failure clusters. The rest would go to
accessing the Waymo train split (currently gated) to move the selector off validation
labels and onto a proper train/test split, and submitting to the live leaderboard.

The simulation environment is ready for hardware deployment. A robot or vehicle with
obstacle sensors and a route command can run Spotlight Reflex today. The physical
prize would fund sensor integration, a first closed-course run, and the latency
profiling needed to hit the 50 ms control budget on real hardware.
