# WOD-E2E System Walkthrough: Operation and Failure Analysis

![Spotlight success](images/spotlight_success.gif)
*Spotlight Reflex — wrong-way actor, night, no training data. Selects `evasive_right`,
clears the actor, recovers to lane centre.*

![Baseline failure](images/baseline_spotlight.gif)
*Baseline policy on the same scenario — trajectory extrapolation without scene
understanding, continues into the actor.*

![Intersection stress](images/intersection_stress.gif)
*Intersection stress — conflict zone, two crossing actors, 207 steps. 0 collisions.*

---

## System Diagram

```mermaid
flowchart LR
    A[TFRecord Shards\n93 validation] --> B[E2EDFrame Parser\nwod_e2e.py]
    B --> C{Candidate\nGeneration}
    C --> D[Kinematic\nconst-vel · accel\nheading · stop]
    C --> E[Ridge Learned\n31 features · 5-fold CV]
    C --> F[Temporal Ridge\nego-history trends]
    D & E & F --> G[WodPreferenceRanker\nHGB · ~137 features\nstability-selected]
    G --> H[Selected Trajectory\n20 × 2 waypoints]
    H --> I[E2EDChallenge\nSubmission.tar.gz]
```

---

## Part 1: Data and Protocol

### 1.1 Dataset

The Waymo Open Dataset for End-to-End Driving (WOD-E2E) provides:
- TFRecord files containing `E2EDFrame` protocol buffer messages
- Each frame: 8 cameras (360° coverage), 12 seconds of ego history, route intent
- Output format: 20 waypoints at 4 Hz over 5 seconds, in vehicle coordinates
- Evaluation: Rater Feedback Score (RFS) — human raters score trajectory proposals

Local availability:
- Validation split: 93 shards, 479 preference-labeled frames with rater scores
- Train and test splits: not downloaded (require Waymo Google sign-in access)
- Test frame list present: `data/waymo/e2e/submission_frames/test_frames.json` (1,505 frames)

### 1.2 Frame Structure

Each `E2EDFrame` contains:
- `frame.context.name` — required submission key
- `past_states` — ego trajectory over (−4s, 0] at 4 Hz (16 points)
- `future_states` — ego ground truth over (0, 5s] at 4 Hz where available
- `intent` — UNKNOWN(0), GO_STRAIGHT(1), GO_LEFT(2), GO_RIGHT(3)
- `preference_trajectories` — up to 3 rater-scored futures (validation only)
- Initial speed, acceleration, timestamp

### 1.3 TFRecord Parsing

**File:** `src/minimal_shot_av/model/wod_e2e.py`

The parser reads Waymo's `E2EDFrame` proto using the official Waymo Open Dataset
Python library. It handles the proto schema exactly as documented in WOD-E2E:
- Trajectory origin is the middle of the ego rear axle
- `past_states` covers (−4s, 0], and the `(0, 0)` point is implicit (not in the proto)
- Preference labels are only available on certain validation frames

---

## Part 2: Candidate Generation Pipeline

### 2.1 Kinematic Candidates

**File:** `src/minimal_shot_av/model/kinematic_candidates.py`

Generated from ego history — no training data required.

Four base families:
1. **Constant velocity** — extrapolate last velocity vector for 20 steps
2. **Constant acceleration** — apply measured acceleration trend
3. **Heading change** — vary heading at multiple rates (left/right, slow/fast)
4. **Hold position** — zero displacement (stopping candidate)

### 2.2 Ridge Learned Candidates

**File:** `src/minimal_shot_av/model/learned_trajectory_model.py`

A `RidgeTrajectoryModel` trained under segment-grouped 5-fold CV on the 479
validation preference frames.

Features (31 numeric):
- `intent`, `init_speed_mps`
- Trajectory statistics: `endpoint_distance`, `total_distance`, `mean_step_distance`,
  `max_step_distance`, `min_step_distance`
- Lateral profile: `final_lateral_abs`, `max_lateral_abs`, `lateral_range`
- Progress: `forward_progress`, `mean_speed_mps`, `max_speed_mps`, `final_speed_mps`
- Dynamics: `mean_abs_accel_mps2`, `max_abs_accel_mps2`
- Lateral dynamics: `mean_abs_lateral_step`, `max_abs_lateral_step`
- Direction: `signed_lateral_5s`, `intent_turn_alignment`
- Heading: `mean_abs_heading_change`, `max_abs_heading_change`
- Waypoints: `x_1s`, `y_1s`, ..., `x_5s`, `y_5s`

Target: `rfs_score_vs_cv_baseline` — RFS of this candidate relative to the
constant-velocity candidate on the same frame.

### 2.3 Temporal Candidates

A second ridge model with temporal ego-history features — differences and
trends in past velocity, heading, and acceleration. The temporal model captures
the "follow the recent trajectory" signal.

Temporal candidates are selected on ~60% of frames in the champion run —
the highest of any source — because most WOD-E2E frames are unambiguous
forward-driving scenes where the recent ego trajectory is the best predictor.

### 2.4 Candidate Scoring and JSONL Format

Each candidate is serialized as a JSONL row with:
- `frame_name` — submission key
- `trajectory` — list of 20 (x, y) waypoints
- `source` — candidate family (kinematic, learned, temporal, etc.)
- `rfs_score` — measured on frames with preference labels
- `ranker_score` — contextual ranker score (used for final selection)

---

## Part 3: Selector Training

**File:** `src/minimal_shot_av/model/wod_ranker.py`

### 3.1 Feature Space

`WodPreferenceRanker` in `contextual` mode uses:
- 31 numeric trajectory features
- 7 context features: 4 speed bins (stopped/creep, slow, urban, fast) + 3 intent dummies
- 9 source one-hot features (kinematic, learned, temporal, etc.)
- 27 source × intent interaction features (9 sources × 3 intent dummies)
- 63 source × context interaction features (9 sources × 7 context features)

Total: ~137 features per candidate row.

### 3.2 Training Protocol

- Cross-validation: 5-fold, grouped by scenario segment (frames from the same
  segment never span a fold boundary)
- Target: `frame_delta` = gain vs constant-velocity on the same frame
- Model: `sklearn.ensemble.HistGradientBoosting` with stability selection

Stability selection: train the HGB multiple times on bootstrapped subsets and
select the feature + hyperparameter configuration that consistently produces
similar fold scores.

### 3.3 Fallback Router

At low speeds (< 1.5 m/s), the ranker switches to preferring kinematic
stop/crawl candidates.

---

## Part 4: Submission Pipeline

**Script:** `scripts/write_wod_e2e_submission.py`

1. Reads the official challenge frame list JSON
2. Reads the merged candidates JSONL (one row per candidate per frame)
3. For each frame, selects the candidate with the highest `ranker_score`
4. Packs the selected trajectory into `E2EDChallengeSubmission` proto format
5. Writes a `.tar.gz` archive matching the official submission format

Validation: `scripts/validate_wod_e2e_submission.py` verifies frame coverage,
trajectory shape (20, 2), and no NaN/Inf values.

Packaged archives are in `artifacts/wod_e2e_submission_matrix/` including
`hgb_selector_v3.tar.gz`.

---

## Part 5: Performance Results

All results are internal validation CV evidence on the 479-frame preference contract.

### 5.1 Model Timeline

All RFS results use local scoring backend (CV baseline 7.131; official Waymo baseline 7.022).

| Selector | RFS | Folds | Notes |
|----------|-----|-------|-------|
| Constant velocity (official) | 7.022 | — | Official Waymo baseline |
| Constant velocity (local) | 7.131 | — | Local scoring baseline |
| Kinematic ranker | 7.096 | — | Physics-only, official backend |
| Ridge r175 contextual router | 7.695 | 2 | Intermediate, local backend |
| HGB (Optuna 2-fold best) | 7.880 | 2 | Optuna-found peak — commit `69b4efb` |
| Gate only (5-fold) | 7.803 | 5 | No direct policy |
| **Champion direct policy (5-fold)** | **7.834** | **5** | **Random-Fourier direct policy** |
| Oracle (perfect selector) | **9.264** | — | Upper bound, local backend |

Champion gap: **1.430 RFS** (9.264 − 7.834). The candidate pool is good.
The discriminator is the bottleneck. HGB variant reached 7.880 under 2-fold evaluation
but is not confirmed at 5-fold.

### 5.2 Source Selection Rates (Champion — RFF direct policy)

Base gate selection (frames not overridden by direct policy, ~96.5%):
- Temporal candidates: ~49%
- Kinematic candidates: ~26%
- Learned candidates: ~12%
- Other: ~10%

Direct policy overrides: ~3.5% of frames (17/479), precision 0.41 (7 TP / 10 FP).
Net direct policy contribution: +0.031 RFS over gate-only baseline.

The temporal dominance shows the WOD-E2E validation set is mostly frames where
recent ego motion is predictive. Unlike the HGB 2-fold analysis (temporal ~60%),
the RFF champion has more balanced kinematic and learned selection, reflecting
different gate configuration and lower temporal over-reliance.

---

## Part 6: Failure Analysis

### 6.1 Structural Failure: Visual Blindness

The system has no camera perception. It processes only ego trajectory history,
speed, and route intent. The camera-based clusters require scene understanding
that numeric ego-history features cannot provide.

### 6.2 Per-Slice Bias Audit

Source: `benchmarks/current/wod_champion_v20_5fold_champion.json` (champion 5-fold)

The champion selector (RFF direct policy, 5-fold) shows residual regret at distribution
extremes, most pronounced for GO_RIGHT and low-speed turn frames:

| Slice | Frames | Selected RFS | Oracle RFS | Regret |
|-------|--------|-------------|-----------|--------|
| GO_STRAIGHT | 427 | 7.959 | 9.336 | 1.377 |
| GO_LEFT | 23 | 7.107 | 8.721 | **1.614** |
| GO_RIGHT | 29 | 6.574 | 8.638 | **2.063** |
| speed:fast | 44 | 7.957 | 9.436 | 1.479 |
| speed:slow | 133 | 7.564 | 9.196 | 1.632 |
| speed:urban | 136 | 8.132 | 9.416 | 1.285 |
| speed:stopped+creep | 166 | 7.775 | 9.148 | 1.373 |

The champion improves substantially over earlier models (GO_LEFT improved from 6.782 to
7.107; GO_RIGHT from 6.088 to 6.574), but GO_RIGHT remains the largest regret slice.
Root cause: GO_RIGHT frames in the validation set are rare (29 frames) and
under-represented in training. The direct policy selects temporal candidates
disproportionately, while oracle often prefers kinematic or learned.

*Historical comparison (r100 ridge model, for reference):*
GO_LEFT: selected 6.782, regret 1.753 · GO_RIGHT: selected 6.088, regret 1.586 · speed:slow: regret 1.638

### 6.3 Global Oracle Gap

The **1.430 RFS oracle gap** (9.264 − 7.834) is frames where the right candidate
exists in the pool but the selector doesn't pick it. Root cause: trajectory statistics
cannot discriminate "good for this specific scene" without visual scene features.

### 6.4 Known Failure Modes in Simulation (Gauntlet)

From the gauntlet suite (60 rollouts, 36/60 pass, 0 collisions):

**Over-stopping under cumulative pressure**: In narrow corridors with multiple
background obstacles, obstacle pressure accumulates and pushes the selector toward
`stop` or `crawl` even when the primary route is clear.

**Slow re-entry after evasive maneuver**: After a nudge, `lane_recover` adds 2–3
steps of latency. The gauntlet progress gate penalizes this.

**Synchronized hazard overload**: Multiple simultaneous threats produce conflicting
reference signals. The combined score averages to a mediocre result; the policy
defaults to `slow_yield`, avoids collision but also avoids the goal.

---

## Part 7: Gauntlet Comparison

Same 120 gauntlet scenarios (4 topologies × 30 seeds). Two policies.

| Policy | Pass rate | Collisions |
|--------|-----------|-----------|
| Baseline (no world-state reasoning) | **0 / 120 (0%)** | **55** |
| Spotlight Reflex | **72 / 120 (60%)** | **1** |

Source: `artifacts/score_baseline_gauntlet_20260425/scenario_eval.json`

The baseline collides in 45% of the hardest scenarios and passes zero.
The geometric reasoning approach maintains near-zero collisions under the same load.

---

## Part 8: Reproducing Results

### Full 350-run evaluation

```bash
uv run --no-sync python scripts/evaluate_scenarios.py \
  --policy spotlight-reflex \
  --suite all \
  --seed-start 1 \
  --seed-end 10 \
  --output-dir artifacts/eval_full_350
```

This reproduces: `artifacts/minor_ood_eval/scenario_eval.json`  
Expected: 326/350 (93.1%) pass, 0 collisions

### WOD cluster sweep (110 runs)

```bash
uv run --no-sync python scripts/evaluate_scenarios.py \
  --policy spotlight-reflex \
  --suite wod \
  --seed-start 1 \
  --seed-end 10 \
  --output-dir artifacts/eval_wod_110
```

This reproduces: `artifacts/agnostic_sim_eval_wod_1_10/scenario_eval.json`  
Expected: 110/110 (100%) pass, 0 collisions, mean min clearance 2.96 m

### Gauntlet comparison (baseline vs Spotlight)

```bash
# Baseline
uv run --no-sync python scripts/evaluate_scenarios.py \
  --policy baseline \
  --suite gauntlet \
  --seed-start 1 \
  --seed-end 30 \
  --output-dir artifacts/eval_baseline_gauntlet

# Spotlight Reflex
uv run --no-sync python scripts/evaluate_scenarios.py \
  --policy spotlight-reflex \
  --suite gauntlet \
  --seed-start 1 \
  --seed-end 30 \
  --output-dir artifacts/eval_spotlight_gauntlet
```

This reproduces: `artifacts/score_baseline_gauntlet_20260425/scenario_eval.json`  
Expected: baseline 0/120 + 55 collisions, Spotlight 72/120 + 1 collision

### COMPASS evidence package

```bash
uv run --no-sync python scripts/run_compass_evidence.py \
  --policy spotlight-reflex \
  --output artifacts/compass_evidence_report_new.json
```

This reproduces: `artifacts/compass_evidence_report.json`  
Expected: COMPASS 9.137/10, 700 ranked runs, success CI [0.9945, 1.0]

### WOD-E2E champion (HGB selector)

```bash
uv run --no-sync python scripts/evaluate_wod_e2e.py \
  --selector hgb \
  --candidates artifacts/wod_e2e_submission_matrix/hgb_selector_v3.tar.gz \
  --output artifacts/eval_hgb_champion.json
```

Expected: 7.880 RFS on 479 validation frames

### Single demo rollout

```bash
uv run --no-sync python scripts/run_demo.py \
  --policy spotlight-reflex \
  --scenario-cluster spotlight \
  --seed 3 \
  --artifacts-dir artifacts/demo_spotlight
```

### Regression test (geometry invariance)

```bash
uv run --no-sync python scripts/run_tests.py --quick
```

Verifies that relabelling all object names, cluster tags, and scenario labels
while holding geometry fixed produces identical decisions.
