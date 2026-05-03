# Solution Reset

This repository should not be presented as a completed autonomy solution.

The current evidence supports a narrower claim: this is a WOD-E2E evaluation and
submission scaffold with a fast non-text trajectory baseline, simulator evidence
tools, and initial world-model experiments. It is useful infrastructure, but it
does not yet demonstrate scene understanding strong enough to impress a serious
AV research audience.

## Current Truth

- The active WOD-E2E model path is still primarily ego-history and route-intent
  driven.
- Structured candidates, residual ridge models, and a linear/ridge selector are
  doing most of the work.
- The strongest confirmed structured-selector run reaches
  `7.657089971818379` official validation-CV RFS on the 479-frame validation
  preference contract, versus `7.0223571581211495` for constant velocity and a
  `9.098014272661512` combined candidate oracle.
- The lightweight world-model candidate path produced only a tiny confirmed
  official-RFS gain: selected RFS moved from `7.602814916652251` to
  `7.606198495114426` on the 479-frame validation preference CV contract.
- World-model oracle headroom improved from `9.015857402777051` to
  `9.046125057713054`, so there is signal, but the selector cannot reliably
  exploit it.
- Scene-token mode exists as an interface, but uncached camera-token loading is
  too slow for practical sweeps and has not produced a confirmed full result.

## Correct Problem Statement

The project should target:

> experience-conditioned scene understanding for WOD-E2E trajectory prediction,
> measured by segment-grouped official-RFS validation CV and worst-slice regret.

That means the solution is not "a world model exists in code." The solution must
prove that scene/experience features improve human-preferred trajectories over
the existing ego-history baseline.

## Acceptance Bar

A future model-side result should not be called a meaningful solution unless it:

- beats the current official selected-RFS baseline by at least `+0.1` RFS on the
  same 479-frame segment-grouped validation CV contract;
- reduces worst-slice regret, especially intent and slow/turning slices;
- uses no simulator feedback for WOD model selection;
- declares all validation preference use and does not call the result strict
  zero-shot;
- keeps test/leaderboard claims separate from validation claims.

## Near-Term Technical Direction

1. Cache camera/scene tokens per WOD frame so visual/scene features can be swept
   without repeatedly decoding TFRecords and JPEGs.
2. Replace the current linear selector with a selector that can learn when to
   trust proposal families, memory neighbors, and scene-conditioned candidates.
3. Add explicit diagnostics for world-candidate wins and failures by slice.
4. Only after a validation-CV gain is proven, train the final model on the
   proper train split and package a test submission.

Until those steps succeed, the honest presentation is: transparent baseline and
measurement harness, not a finished autonomy architecture.
