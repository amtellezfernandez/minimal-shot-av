# Grand Commission Plan

## Goal

Position the submission as a minimal-shot autonomy architecture for **WOD-E2E long-tail driving**, not as a simulator or route-following demo.

The strongest deliverable is an end-to-end policy that operates on WOD-E2E with a fully declared training and evaluation boundary. The judged contribution should be the architecture and analysis: how the system uses ego motion, route intent, candidate diversity, and rater-aligned selection to produce plausible 5-second ego trajectories in rare scenarios.

## Benchmark Grounding

WOD-E2E is the right target because it directly stresses the minimal-shot requirement:

- 4,021 curated long-tail driving segments.
- 20 seconds per segment, with 8-camera 360-degree imagery and ego status.
- Test-time input includes 12 seconds before the target moment.
- Test reserves the subsequent 8 seconds for evaluation.
- Required output is 5 seconds of ego waypoints at 4 Hz: `(20, 2)`.
- Validation includes rater feedback labels and scenario tags.
- Primary metric is Rater Feedback Score, so acceptable driving decisions matter more than raw imitation.
- The challenge page reports these mined events occur at less than `0.003%` frequency in daily driving.

Access and implementation must follow `docs/waymo-data-access.md`, the dataset dossier in `docs/wod-e2e-deliverable.md`, and the schema reference in `docs/wod-e2e-schema.md`. The important low-level facts are:

- parse `E2EDFrame` records, not generic Waymo perception frames
- use `frame.context.name` as the submission key
- treat `past_states` as `(-4s, 0]` at 4 Hz
- emit 20 future `(x, y)` points over `(0, 5s]`
- evaluate RFS at validation frames with valid rater preference trajectories
- package leaderboard submissions as serialized `E2EDChallengeSubmission` proto file(s) inside `.tar.gz`
- use the challenge-provided JSON to select required test frames
- verify the coordinate origin wording because the proto states rear axle while public prose may imply vehicle center

## Strongest Pitch

> We reject route memorization and slow per-frame deliberation. We propose a Predictive Reflex: non-text trajectory generators produce diverse futures from ego history and route intent, and a source-aware RFS selector chooses the rater-plausible 5-second waypoint path.

This is more credible than claiming a production driver. It targets the actual WOD-E2E question: can a compact, fast policy choose a human-preferred future path in long-tail scenes with minimal task-specific machinery?

## Recommended Stack

### 1. WOD-E2E Adapter

Parse the official records into the submission system:

- 8 synchronized camera views
- high-level command: left, straight, or right
- ego past trajectory, velocity, and acceleration
- scenario tags and rater feedback labels for validation
- output writer for `E2EDChallengeSubmission` protos

### 2. Non-Text Candidate Generators

Generate a diverse set of candidate trajectories from structured signals:

- kinematic extrapolations
- route-intent templates
- ridge residual proposals
- learned anchor/residual proposals as an experimental branch
- conservative fallbacks

All active candidates must produce numeric trajectories directly, without text prompts or language-mediated trajectory parsing.

### 3. Temporal Motion State

Maintain temporal state over the 12-second context:

- ego motion trend
- route intent
- speed and acceleration profile
- curvature trend
- stop/progress tendency
- candidate-source reliability

The latency claim should be bounded and measured: parse records, generate candidates, rank, and write the single `(20, 2)` output.

### 4. Latent Maneuver Library

Retrieve or score maneuver candidates:

- continue
- slow/yield
- stop
- nudge left/right
- lane change
- cut-in response
- debris or animal avoidance
- conservative fallback

The maneuver library is the bridge between semantic reasoning and continuous waypoint output.

### 5. Trajectory Decoder

Convert candidate maneuvers into the challenge output:

- 20 future points
- 0.25 seconds per step
- vehicle-frame `(x, y)` positions
- kinematic smoothing
- high-level route compliance

### 6. Rater-Aware Selector

Use validation labels to analyze, calibrate, and report behavior:

- RFS-style candidate selection
- ADE as a secondary diagnostic
- failure categories by scenario cluster
- confidence threshold for fallback behavior
- explicit 3s and 5s endpoint analysis because RFS trust regions are evaluated there

## Demo Strategy

For the video and write-up, show:

- one construction or debris case where the policy chooses a cautious trajectory
- one cut-in or pedestrian case where temporal memory matters
- one failure case where the system misreads the scene or overuses fallback

The failure case is not optional. It is evidence that the analysis was real.

## Evaluation Claims Worth Making

- validation RFS by scenario cluster
- ADE at 3 and 5 seconds as a secondary measure
- fallback rate
- invalid trajectory rate
- route-command violation rate
- runtime per target frame on stated hardware

## Fastest Path From Current Scaffold

1. Keep the current simulator as an architecture sketch and demo fallback.
2. Add a `wod_e2e/` data adapter and notebook-first workflow.
3. Build learned residual and anchor proposal baselines behind benchmark gates.
4. Implement learned candidate generators and keep each experiment benchmarked.
5. Compare against simple baselines: constant velocity, route-following, and logged-history extrapolation.
