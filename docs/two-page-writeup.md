# Minimal-Shot AV Write-Up Draft

## Motivation

Most end-to-end driving benchmarks reward performance on common driving distributions. WOD-E2E is different: it curates rare, long-tail events where generalization matters most, including construction zones, cut-ins, erratic pedestrians, debris, animals, and unusual maneuvers.

The submission claim is that a useful autonomy system should not require a bespoke rule for every new rare event category. It should combine compact motion priors, route intent, candidate diversity, and rater-aligned selection in a fast policy.

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

- All AV-specific training, validation use, and model components are declared explicitly.
- The active runtime is non-text and does not depend on prompt parsing.

## Architecture

The proposed system is **Spotlight Reflex**:

- Kinematic and learned residual generators propose multiple 20-waypoint futures from ego history and route intent.
- Anchor/residual models are evaluated as proposal generators, not enabled unless their validation score improves the selected RFS.
- A source-aware numeric selector ranks candidates at 3s and 5s with the official RFS trust-region utility.
- The simulator stack is kept separate from WOD-E2E scoring so simulator tuning cannot leak into model claims.
- A safety projector applies kinematic smoothing, route-command checks, and invalid-trajectory rejection.

The key architectural bet is that minimal-shot driving should separate **candidate diversity** from **trajectory selection**. Proposal models generate plausible futures; model-side RFS scoring evaluates WOD-E2E candidates without coupling that metric to the simulator.

## What Worked

To be filled after validation analysis:

- scenario clusters where learned residual candidates improved maneuver choice
- cases where temporal history changed the decision compared with a single-frame policy
- cases where maneuver candidates gave better RFS than constant-velocity or route-following baselines

## What Failed

At least one concrete failure must be documented:

- visual ambiguity, such as occluded pedestrian or distant debris
- route-command conflict, such as lane geometry suggesting one action while the command suggests another
- over-conservative fallback, such as stopping when a cautious nudge would score better
- trajectory projection failure, such as a physically smooth but rater-poor path

The failure diagnosis should name the component responsible: candidate generator, temporal state, source ranker, trajectory projector, or safety projector.

## Next Step With Prize Money

The next milestone is a WOD-E2E validation-quality prototype:

- submission writer and strict proto validator
- reproducible validation notebook
- RFS-based evaluation table by scenario cluster
- ablation against constant-velocity, route-following, kinematic candidate, and residual-candidate baselines
- small edge-deployable trajectory decoder connected to the source-aware selector
