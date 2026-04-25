# Minimal-Shot AV Write-Up Draft

## Motivation

Most end-to-end driving benchmarks reward performance on common driving distributions. WOD-E2E is different: it curates rare, long-tail events where generalization matters most, including construction zones, cut-ins, erratic pedestrians, debris, animals, and unusual maneuvers.

The submission claim is that a useful autonomy system should not require AV-specific fine-tuning for every new rare event category. It should combine general visual reasoning with a compact, reactive trajectory policy.

## Problem Setting

Input:

- 8-camera 360-degree WOD-E2E context
- ego pose/status history
- high-level route command
- 12 seconds of pre-decision context at test time

Output:

- 5-second ego trajectory
- 20 `(x, y)` waypoints in vehicle coordinates
- first waypoint at 0.25 seconds into the future

Primary target metric:

- Rater Feedback Score, because WOD-E2E evaluates whether the predicted trajectory matches human-rated acceptable driving decisions, not only the logged future.

Constraint:

- No base model is fine-tuned on AV-specific data for the submitted policy.

## Architecture

The proposed system is **Spotlight Reflex**:

- A frozen scene critic interprets the 8-camera context and identifies long-tail hazards.
- Counterfactual scene hypotheses cover ambiguity such as hidden pedestrians, debris, construction lane shifts, and cut-ins.
- A latent maneuver library retrieves candidate behaviors such as yield, brake, nudge, lane change, cut-in response, debris avoidance, and fallback.
- A trajectory decoder projects each maneuver into 20 future waypoints.
- An exact RFS trust-region selector ranks candidates at 3s and 5s using the known lateral/longitudinal thresholds and initial-speed scaling.
- An optional small RFS verifier can be trained on validation preference labels as a declared preference-calibrated variant; the strict zero-shot variant does not use this learned verifier.
- A safety projector applies simple kinematic smoothing, route-command checks, and invalid-trajectory rejection.

The key architectural bet is that minimal-shot driving should separate **semantic uncertainty** from **trajectory generation**. Frozen models identify rare hazards and plausible counterfactuals; the maneuver library generates candidates; exact RFS-style selection chooses the trajectory most likely to be rater-acceptable without AV fine-tuning.

## What Worked

To be filled after validation analysis:

- scenario clusters where frozen scene interpretation improved maneuver choice
- cases where temporal history changed the decision compared with a single-frame policy
- cases where maneuver candidates gave better RFS than constant-velocity or route-following baselines

## What Failed

At least one concrete failure must be documented:

- visual ambiguity, such as occluded pedestrian or distant debris
- route-command conflict, such as lane geometry suggesting one action while the command suggests another
- over-conservative fallback, such as stopping when a cautious nudge would score better
- trajectory projection failure, such as a physically smooth but rater-poor path

The failure diagnosis should name the component responsible: scene critic, temporal pilot, maneuver retrieval, trajectory decoder, or safety projector.

## Next Step With Prize Money

The next milestone is a WOD-E2E validation-quality prototype:

- submission writer and strict proto validator
- reproducible validation notebook
- RFS-based evaluation table by scenario cluster
- ablation against constant-velocity, route-following, and frozen-scene-only baselines
- small edge-deployable SSM pilot connected to the latent maneuver library
