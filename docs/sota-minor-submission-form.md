# SoTA Minor Commission Form Draft

## Title

Randomized Long-Tail Scenario Generator for Minimal-Shot Autonomy

## Track

Minor Commission: overall best simulation environment.

## Short Description

This submission is a seeded procedural simulation environment for long-tail
minimal-shot autonomy. It generates WOD-E2E-style named clusters and
compositional OOD cases where topology, hazards, object novelty, conditions,
surface friction, and latency budgets are sampled independently. Each scenario
is reproducible by suite/cluster and seed, includes typed actors and map
features, and can be evaluated in closed loop with the provided policy harness.

## Motivation

Minimal-shot autonomy cannot be judged only on fixed maps or memorized examples. A useful simulator should continuously create unfamiliar but structured cases, making it hard for a policy to overfit. This generator focuses on rare scenario families where generalization matters: occlusion, debris, ambiguous right-of-way, unusual agents, and constrained corridors.

## Technical Approach

The simulator separates visual scene texture from evaluation hazards. Ambient
vehicles and background objects add context without accidentally blocking the
route, while deliberate hazards are typed and placed through cluster-specific
or compositional logic. The evaluation harness records success, collision, goal
reach, clearance, intervention rate, progress, speed, comfort cost, decision
mode diversity, and benchmark pass/fail.

The environment is intentionally lightweight and deterministic. It is not photorealistic, but it gives a compact testbed for closed-loop decision making, randomized scenario generation, and failure-case analysis.

## Evidence Included

- Construction scenario demo.
- Foreign-object-debris scenario demo.
- Spotlight scenario demo.
- Multi-cluster evaluation sweep with `scenario_eval.csv` and `scenario_eval.json`.
- Compositional OOD evaluation sweep with `minor_ood_eval/scenario_eval.csv`
  and `minor_ood_eval/scenario_eval.json`.
- SVG and JSON artifacts for visual inspection and reproducibility.
- Full test suite output: `UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python scripts/run_tests.py`.

Current bundled sweep:

- 550 closed-loop rollouts.
- 11 WOD-E2E-style scenario clusters.
- 50 seeds per cluster.
- 550 / 550 successful rollouts.
- 550 / 550 benchmark passes.
- 0 collisions.
- 0 safe stalls.
- 0 near misses under the benchmark diagnostic.
- mean minimum clearance: 2.80 m.
- mean 5th-percentile clearance: 3.28 m.
- mean intervention rate: 9.44%.

Additional bundled OOD sweep:

- 350 closed-loop rollouts.
- 5 suites: WOD, compositional, adversarial, gauntlet, hidden.
- 350 / 350 successful rollouts.
- 326 / 350 benchmark passes.
- 0 collisions.
- 0 safe stalls.
- 0 near misses under the benchmark diagnostic.
- Gauntlet pass rate: 36 / 60 under stricter near-miss, intervention, and
  progress gates.

## What Worked

The generator creates repeatable but varied long-tail scenarios, and the evaluation sweep produces concrete closed-loop metrics across clusters and seeds. This directly addresses the commission's request for randomized scenario generation under realistic evaluation constraints.

The strongest evidence is not a single cherry-picked scene. It is the same
generator and policy interface running construction, intersections, pedestrian
conflicts, cyclist cases, cut-ins, debris, special vehicles, spotlight hazards,
unusual maneuvers, and compositional OOD suites with deterministic seeds and
inspectable artifacts.

## What Did Not Work Yet

The simulator is a lightweight 2D environment, not a photorealistic sensor simulator. AlpaSim integration exists at a trajectory/plugin level, but full sensor-realistic camera and perception integration remains future work.

## Next Use Of Prize Funding

Prize funding would go toward richer simulation fidelity: camera-like rendering, improved actor behaviors, larger randomized scenario libraries, AlpaSim integration, and automated failure-case mining to stress minimal-shot policies.
