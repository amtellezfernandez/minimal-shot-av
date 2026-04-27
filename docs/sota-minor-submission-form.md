# SoTA Minor Commission Form Draft

## Title

Randomized Long-Tail Scenario Generator for Minimal-Shot Autonomy

## Track

Minor Commission: overall best simulation environment.

## Short Description

This submission is a seeded procedural simulation environment for WOD-E2E-style long-tail driving cases. It generates randomized construction zones, intersections, pedestrian conflicts, cyclist cases, cut-ins, debris, special vehicles, spotlight hazards, and unusual maneuvers. Each scenario is reproducible by cluster and seed, includes typed actors and map features, and can be evaluated in closed loop with the provided policy harness.

## Motivation

Minimal-shot autonomy cannot be judged only on fixed maps or memorized examples. A useful simulator should continuously create unfamiliar but structured cases, making it hard for a policy to overfit. This generator focuses on rare scenario families where generalization matters: occlusion, debris, ambiguous right-of-way, unusual agents, and constrained corridors.

## Technical Approach

The simulator separates visual scene texture from evaluation hazards. Ambient vehicles and background objects add context without accidentally blocking the route, while deliberate hazards are typed and placed through cluster-specific logic. The evaluation harness records success, collision, goal reach, clearance, intervention rate, progress, speed, comfort cost, and benchmark pass/fail.

The environment is intentionally lightweight and deterministic. It is not photorealistic, but it gives a compact testbed for closed-loop decision making, randomized scenario generation, and failure-case analysis.

## Evidence Included

- Construction scenario demo.
- Foreign-object-debris scenario demo.
- Spotlight scenario demo.
- Multi-cluster evaluation sweep with `scenario_eval.csv` and `scenario_eval.json`.
- SVG and JSON artifacts for visual inspection and reproducibility.
- Full test suite output: `UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python scripts/run_tests.py`.

## What Worked

The generator creates repeatable but varied long-tail scenarios, and the evaluation sweep produces concrete closed-loop metrics across clusters and seeds. This directly addresses the commission's request for randomized scenario generation under realistic evaluation constraints.

## What Did Not Work Yet

The simulator is a lightweight 2D environment, not a photorealistic sensor simulator. AlpaSim integration exists at a trajectory/plugin level, but full sensor-realistic camera and perception integration remains future work.

## Next Use Of Prize Funding

Prize funding would go toward richer simulation fidelity: camera-like rendering, improved actor behaviors, larger randomized scenario libraries, AlpaSim integration, and automated failure-case mining to stress minimal-shot policies.
