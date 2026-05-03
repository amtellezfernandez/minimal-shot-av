# Minimal-Shot AV Write-Up Draft

## Motivation

Most end-to-end driving benchmarks reward performance on common driving distributions. WOD-E2E is different: it curates rare, long-tail events where generalization matters most, including construction zones, cut-ins, erratic pedestrians, debris, animals, and unusual maneuvers.

The submission claim is that a useful autonomy system should not require a bespoke rule for every new rare event category. It should combine compact motion priors, route intent, candidate diversity, and rater-aligned selection in a fast policy.

The judging evidence is now audited explicitly. The Grand bundle contains
`docs/judging-criteria-evidence.md` and
`artifacts/sota_judging_criteria_audit.json`, which map the repo evidence to
technical excellence, novelty, feasibility, and adherence to the brief.

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

The strongest working result is the simulator and evaluation harness. The
Minor submission bundle includes 550 closed-loop Spotlight Reflex rollouts
across all 11 WOD-E2E-style long-tail clusters, using 50 seeds per cluster. In
that sweep the policy records 550 / 550 benchmark passes, 550 / 550 successful
rollouts, 0 collisions, 0 safe stalls, and 0 near misses under the benchmark
diagnostic. The mean minimum clearance is 2.80 m, mean 5th-percentile clearance
is 3.28 m, and mean intervention rate is 9.44%.

On the WOD-E2E validation side, the best confirmed official-RFS development
report is a transparent non-text baseline rather than a solved vision policy.
Constant velocity scores 7.022 RFS on the 479 preference-labeled validation
frames, while the current combined candidate ranker scores 7.6594 RFS under
segment-grouped cross-validation. The combined candidate oracle is much higher
at 9.098 RFS, showing useful proposal headroom but also exposing selector
regret. A conservative scene-gate sweep improved the prior 7.657 report while
later tuning raised the promoted validation-CV score to 7.6594 without
worsening worst-slice regret.

A preference-calibrated neural heldout branch reached `7.838` local RFS on 159
heldout validation-preference frames, and an official-scored heldout variant
reached `7.894` normalized RFS. The full retained-validation official rerun
scored only `7.262` mean RFS, so this is evidence of promising calibration
headroom rather than the promoted WOD result.

The most important engineering result is separation of concerns: simulator
metrics are not used for WOD model selection, WOD validation labels are not
presented as hidden-test results, and submission packaging is kept behind
explicit readiness checks.

The realtime path now has a concrete fast/slow proposal: a sub-2 ms fast reflex
loop consumes compact local candidates and one cached slow plan, while a slower
loop refreshes richer candidates asynchronously. On a same-settings 5,000-frame
runtime benchmark, synchronous p95 latency improves from `2.124 ms` to
`0.625 ms`; the slow refresh p95 is reported separately at `2.132 ms`.

The direct criteria audit passes all four judging dimensions under this honest
boundary: technical excellence from WOD validation-CV gain, oracle headroom,
runtime margin, and closed-loop breadth; novelty from explicit
candidate-selector separation and negative-result reporting; feasibility from
zero-collision randomized sweeps and latency evidence; and adherence from the
analysis notebook, randomized scenario generation, WOD-E2E evidence, and
failure analysis.

## What Failed

The model-side world-model experiment did not yet produce a meaningful
architecture win. The lightweight world-model candidate moved official
validation-CV selected RFS only from 7.6028 to 7.6062. The stronger confirmed
selector run reaches 7.6594, but that improvement comes from structured
fallback/gating over existing candidates rather than a solved scene-understanding
model.

The WOD-E2E leaderboard path is also incomplete in this workspace. Validation
TFRecords are present, but train/test TFRecords and the official required-frame
list are missing locally. There is therefore no hidden-test score and no claim
that this is a completed leaderboard submission.

The known failure mode is candidate selection. The confirmed official report has
a 1.4386 RFS regret gap between selected candidates and the combined candidate
oracle. Before
this can become a stronger autonomy claim, the selector must reduce worst-slice
regret and learn when to trust learned, temporal, memory, or world-model
proposal families.

## Next Step With Prize Money

The next milestone is to turn the prototype into a stronger minimal-shot
evaluation loop:

- complete blind WOD-E2E data access, frame-list handling, and leaderboard
  submission runs;
- build a reproducible analysis notebook with camera montages, trajectory plots,
  RFS diagnostics, and failure cases;
- replace the current selector with a stronger rater-aware router that closes
  the oracle gap;
- add real camera grounding or cached scene tokens so rare visual evidence can
  influence trajectory choice;
- expand the simulator toward richer actor behavior, camera-like rendering, and
  AlpaSim sensor integration.
