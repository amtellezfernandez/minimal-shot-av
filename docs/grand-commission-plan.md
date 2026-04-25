# Grand Commission Plan

## Goal

Position the submission for **overall best autonomy architecture**, not for best simulator.

The simulator is there to demonstrate generalization under controlled variation. The judged contribution should be the policy stack.

## Strongest pitch

Build an architecture with these properties:

- no route memorization
- no map-specific heuristics
- explicit scene representation at runtime
- online planning with safety constraints
- graceful degradation when uncertain

## Recommended stack

### 1. Perception

Convert raw observations into a compact scene graph or occupancy-style representation:

- drivable space
- lane or corridor hypotheses
- dynamic obstacles
- uncertainty estimates

### 2. World model

Maintain a short-horizon latent state that predicts:

- free-space evolution
- agent motion
- collision risk
- goal progress

### 3. Planner

Use constrained planning over the world model:

- propose candidate trajectories
- reject unsafe actions
- prefer progress only when safety margin is acceptable

### 4. Fallback controller

When confidence drops:

- slow down
- stop
- choose conservative recovery actions

This matters for the brief because minimal-shot capability is only credible if the system stays sane outside its comfort zone.

## Demo strategy

For the video and write-up, show:

- one unseen environment that works
- one difficult scenario that partly works
- one failure case with a clear diagnosis

## Evaluation claims worth making

- success rate on unseen seeds
- collision rate
- minimum clearance
- average intervention count
- latency per decision step

## Fastest path from current scaffold

1. Keep the existing procedural simulator as the first evaluation harness.
2. Replace the hand-built reactive policy with a modular planner stack.
3. Add scenario families that stress occlusion, narrow passages, and late hazard appearance.
4. Only then decide whether to extend into WOD-E2E or stay simulator-first.

