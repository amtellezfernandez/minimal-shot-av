# Spotlight Reflex: Minimal-Shot Autonomous Driving

**Author:** amtellezfernandez@gmail.com  
**Repo:** github.com/amtellezfernandez/minimal-shot-av

A closed-loop driving policy that reasons through unfamiliar long-tail scenarios
without fine-tuning on any AV dataset. Built from scratch: custom 2D simulator,
procedural scenario generator for all 11 WOD-E2E clusters, AlpaSim trajectory
plugin, and a WOD-E2E preference-calibrated benchmark harness.

---

## The System Operating

**Spotlight scenario** — wrong-way actor, low visibility (0.69), night conditions.
Spotlight Reflex selects `evasive_right`, clears the actor, recovers to lane centre.

![Spotlight Reflex success](images/spotlight_success.gif)

**Baseline policy on the same scenario** — no world-state reasoning, continues
into the actor. This is what trajectory imitation without scene understanding does.

![Baseline failure](images/baseline_spotlight.gif)

**Construction zone** — cone field, narrow corridor (5.2 m half-width), lane closure
ahead. Policy selects `nudge_right`, maintains progress through the gap.

![Construction rollout](images/construction_success.gif)

**Foreign object debris** — static debris blocking primary lane. Policy detects
obstacle pressure rising, selects `evasive_left`, clears the debris cleanly.

![FOD rollout](images/fod_success.gif)

**Intersection stress** — conflict zone, two crossing actors at different timings.
207 steps. Policy alternates between `slow_yield` and `maintain` as each threat clears.

![Intersection stress](images/intersection_stress.gif)

---

## Motivation

Today's autonomous vehicles work where they've been trained. Waymo's WOD-E2E
dataset makes the failure mode concrete: 11 long-tail clusters filtered to scenarios
occurring less than 0.03% of driving time — construction zones, erratic pedestrians,
animals, debris, wrong-way actors. These are exactly the scenarios where trajectory
memorisation breaks down.

The hypothesis: if you make world-state and maneuver reasoning explicit, the system
should generalise to novel scenes without having seen them. A trajectory that avoids
obstacles, respects corridor geometry, and maintains progress is correct whether
the obstacle is a cone, a fallen tree, or a sheep.

---

## Architecture

### Spotlight Reflex (Closed-Loop Policy)

At each step, the policy derives six scalar fields from scene geometry:

| Field | Description |
|-------|-------------|
| `obstacle_pressure` | Weighted sum of nearby obstacle influence [0,1] |
| `route_blockage` | Fraction of route corridor blocked ahead [0,1] |
| `corridor_blocked` | Hard blockage within stopping distance |
| `left_clearance` | Lateral space to the left (metres) |
| `right_clearance` | Lateral space to the right (metres) |
| `preferred_escape_side` | Which side has more clearance |

Nine maneuver candidates are generated as 20-point 5-second trajectories:
`stop · crawl · maintain · slow_yield · nudge_left · nudge_right · evasive_left · evasive_right · lane_recover`

A reference rule engine maps world state to pseudo-rater references with target
scores (e.g. clear corridor → `maintain` at 92, obstacle present → `nudge` at 94).
The selector scores each candidate against references at 3 s and 5 s trust regions,
applies safety and progress penalties, and outputs the highest-scoring safe option.

No AV-dataset fine-tuning. No route memorisation. The regression tests verify that
all world-state signals are unchanged when object names, labels, and cluster tags are
relabelled while geometry is held fixed.

### Simulation Environment (Built from Scratch)

The entire simulator is original code. Key components:

- **`environment.py`** — 2D closed-loop engine with 8 actor behavior models
  (cut-in, swerve, darting, erratic pedestrian, sudden brake, hesitating, wrong-way)
- **`wod_scenarios.py`** — procedural generator for all 11 WOD-E2E cluster types,
  each with cluster-specific lane geometry, obstacle fields, and actor roles
- **`compositional_scenarios.py`** — OOD generator that independently samples
  topology (8 types), hazard module, weather, novel object type, and visibility fraction.
  Topology and hazard are decoupled: a construction hazard can appear in a roundabout
  or on an S-curve — memorising cluster labels gives no advantage
- **`compass.py`** — profile-driven composite benchmark (oracle, reasoning, recovery,
  generalisation gap scores)

### AlpaSim Adapter

The same Spotlight Reflex policy runs inside Waymo's sensor-realistic AlpaSim
simulator via a custom adapter. Below: real WOD-E2E front-camera frames (AlpaSim
sensor input) alongside the adapter's selected maneuver and world-state readout.

![AlpaSim reasoning panel](images/alpasim_reasoning_panel.png) The engineering challenge: AlpaSim provides cameras,
speed, and a route command; Spotlight Reflex needs an obstacle field, a lane spline,
and a world state. Four signal channels bridge the representations:

1. **Route command → lane geometry**: `LEFT/STRAIGHT/RIGHT` maps to a sigmoid-shaped
   5-point lane spline without touching AlpaSim's internal map
2. **Camera brightness → visibility risk**: mean pixel brightness below 35% triggers
   risk ∈ (0,1]; no vision model needed for low-light caution
3. **Ego dynamics → braking risk**: hard braking (≤ −4 m/s²) or near-stop speed
   both trigger conservative candidate selection
4. **AlpaSignal structured hazards → Obstacle/Actor objects**: static hazards become
   simulator obstacles; moving hazards (vx/vy ≠ 0) become actors with inferred
   behavior; if no hazards but risk > 0.5, a synthetic caution zone is placed 10–18 m ahead

The Spotlight Reflex policy has zero AlpaSim imports. The adapter uses optional imports
with stub fallbacks so the policy runs identically in both environments.

Every prediction includes full `reasoning_text` JSON: selected maneuver, world-state
fields, reference scores, candidate summaries — so reviewers can inspect exactly why
a trajectory was chosen.

### WOD-E2E Harness

Model-side pipeline for the WOD-E2E trajectory benchmark:
- Kinematic candidates (constant velocity, acceleration, heading change, stop)
- Ridge learned candidates (31 features, segment-grouped 5-fold CV)
- Temporal ridge model (ego-history trends)
- `WodPreferenceRanker`: HGB stability-selected classifier with ~137 features
  (trajectory stats + speed bins + intent dummies + source × context interactions)
- Submission packaging to `E2EDChallengeSubmission.tar.gz` format

---

## Results

| Metric | Value |
|--------|-------|
| OOD rollouts (abstract simulator) | 350 |
| Collisions | 0 |
| COMPASS benchmark pass rate | 326 / 350 (93.1%) |
| Gauntlet pass rate | 36 / 60 (60%) |
| WOD-E2E constant-velocity baseline | 7.022 RFS |
| WOD-E2E champion (HGB stability-selected) | **7.880 RFS** |
| WOD-E2E oracle (best candidate per frame) | 9.068 RFS |

The gauntlet (synchronised multi-hazard, narrow corridor, strict progress gates)
is deliberately not saturated. 60% is where the policy's failure boundary is, and
reporting it honestly is more useful than inflating it with easier scenarios.

---

## Failures

**Visual blindness is the hard ceiling on WOD-E2E.** The model path processes only
ego history, speed, and route intent — no camera input. It cannot see pedestrians
crossing, debris, or traffic signals. The 7.880 RFS result reflects this: the
constant-velocity oracle gap is 9.068 − 7.880 = 1.19 RFS, and the worst clusters
(GO_LEFT: 6.782 selected vs 8.535 oracle; GO_RIGHT: 7.191 vs 8.829) are exactly
the visual-decision scenarios.

**Visual embeddings didn't transfer.** InternVLA and Cosmos tokenizer embeddings
were computed for the camera frames and attached as ranker features. Neither produced
a confirmed RFS gain. These embedding spaces are not aligned to the discriminative
signal the ranker needs: which candidate trajectory wins on this specific frame.

**Selector mis-calibrated on turns.** The training distribution is dominated by
straight-ahead frames. For left-turn frames the selector over-relies on kinematic
candidates (87% selected vs 57% oracle rate). This is a known training-distribution
bias, not a structural failure of the approach.

**Gauntlet failure modes** (the 40% that fail): over-stopping under cumulative
obstacle pressure in narrow corridors; slow re-entry after evasive maneuver; conflicting
references when multiple simultaneous threats produce opposing evasion signals.

---

## Next Steps

The bottleneck is scene understanding, not candidate quality. The oracle proves good
candidates exist for 9.068 RFS — the discriminator cannot pick them without visual input.

**Camera encoder.** A lightweight model fine-tuned on WOD-E2E preference labels to
discriminate winning trajectories from camera images should close 0.5–1.5 RFS of the
oracle gap. This is not a general VLM — it's a small discriminative model trained on
the specific signal the ranker needs.

**Train-split access.** The selector is currently calibrated on the validation preference
labels (segment-grouped CV). Moving to a proper train/test split requires the Waymo
train TFRecords (Google sign-in gated). Once available, the selector calibration becomes
cleaner and the held-out validation frames become a true test set.

**Turn calibration.** The GO_LEFT and GO_RIGHT slices are the worst-performing (regret
1.75 and 1.64 RFS respectively). Augmenting with turn-specific training frames and
weighting them more heavily in the selector loss should close this gap.

**Leaderboard submission.** The submission packaging pipeline is complete and validated
against the official schema. It only needs the test TFRecords and the official frame
list JSON to produce test predictions.

**Physical deployment.** The abstract simulator and AlpaSim adapter demonstrate the
policy is deployable. The next step is hardware integration: a robot or vehicle with
obstacle sensors and a route command can run Spotlight Reflex with a sensor-to-obstacle
adapter analogous to the AlpaSim bridge.
