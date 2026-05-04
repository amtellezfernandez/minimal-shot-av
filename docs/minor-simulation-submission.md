# Minor Commission Submission: WOD-E2E Procedural Scenario Generator

## Submission Claim

The simulation-environment contribution is a seeded procedural generator for
minimal-shot long-tail autonomy. It includes WOD-E2E-style named clusters and a
second compositional OOD layer that independently samples route topology,
hazard modules, novel object types, weather, visibility, surface friction, and
latency budgets.

The generator separates **ambient scene texture** from **blocking evaluation hazards**. Ambient vehicles, sidewalk clutter, and background objects are rendered for visual context but do not accidentally block the primary route. Each scenario family places its deliberate hard decision point through typed actors or cluster-specific hazards with a minimum corridor-clearance contract.

This submission targets the **Minor Commission**: overall best simulation
environment.

## What To Submit

- GitHub repo: this codebase, especially `src/minimal_shot_av/simulator/wod_scenarios.py`.
- Video or slide deck: show multiple seeded scenarios from different clusters and the policy rolling through them.
- Short write-up: describe why randomized long-tail generation tests minimal-shot generalization better than a fixed memorized map.
- Demo artifacts: include SVG/JSON outputs from several clusters and seeds.

## Demo Commands

Generate a construction scenario:

```bash
uv run --no-sync python scripts/run_demo.py \
  --policy spotlight-reflex \
  --scenario-cluster construction \
  --seed 1 \
  --artifacts-dir artifacts/minor_construction_seed1
```

Generate a foreign-object-debris scenario:

```bash
uv run --no-sync python scripts/run_demo.py \
  --policy spotlight-reflex \
  --scenario-cluster "foreign object debris" \
  --seed 2 \
  --artifacts-dir artifacts/minor_fod_seed2
```

Generate a spotlight scenario:

```bash
uv run --no-sync python scripts/run_demo.py \
  --policy spotlight-reflex \
  --scenario-cluster spotlight \
  --seed 3 \
  --artifacts-dir artifacts/minor_spotlight_seed3
```

Run an all-cluster evaluation sweep:

```bash
uv run --no-sync python scripts/evaluate_scenarios.py \
  --policy spotlight-reflex \
  --seed-start 1 \
  --seed-end 20 \
  --output-dir artifacts/minor_eval
```

Each command writes:

- `latest_rollout.json`
- `latest_rollout.svg`

The evaluation command writes:

- `scenario_eval.json`
- `scenario_eval.csv`

Run the stronger compositional OOD sweep:

```bash
uv run --no-sync python scripts/evaluate_scenarios.py \
  --policy spotlight-reflex \
  --suite all \
  --seed-start 1 \
  --seed-end 10 \
  --output-dir artifacts/minor_ood_eval
```

This adds WOD-style, compositional, adversarial, gauntlet, and hidden-suite
rollouts to the Minor bundle.

## Supported Scenario Clusters

- construction
- intersection
- pedestrian
- cyclist
- multi-lane maneuver
- single-lane maneuver
- cut-in
- foreign object debris
- special vehicle
- spotlight
- others

Each generated scenario includes:

- deterministic seed
- cluster name
- generator tags
- weather, visibility, time of day, and road surface metadata
- typed dynamic actors with role, velocity, dimensions, and behavior profile
- map features such as crosswalks, lane closures, merge zones, and conflict zones
- visual-only ambient texture separated from blocking hazards
- lane geometry
- obstacle field
- start and goal
- rollout artifacts

## Compositional OOD Suites

The Minor bundle now includes `minor_ood_eval`, a second evaluation artifact
that exercises:

- `compositional`: independently sampled topology, hazards, conditions, and
  object types.
- `adversarial`: multiple hazards composed into the same route.
- `gauntlet`: narrow, low-visibility, synchronized hazards with stricter
  benchmark-pass gates.
- `hidden`: holdout seed offsets for frozen-policy evaluation.
- `wod`: the named WOD-style clusters retained for brief alignment.

Bundled OOD sweep:

- 350 additional closed-loop rollouts.
- 5 suites: WOD, compositional, adversarial, gauntlet, hidden.
- 0 collisions.
- 350 / 350 successful rollouts.
- 326 / 350 benchmark passes.
- Gauntlet pass rate: 36 / 60 under the stricter near-miss, intervention, and
  progress gates.

## Evidence To Highlight

- The generator is deterministic: same cluster and seed produce the same scenario.
- Different seeds produce different obstacle layouts and parameters.
- The simulation is closed-loop: the policy reacts step-by-step to the generated scene.
- It directly addresses the commission's "extra points" criterion for randomized scenario generation.
- The compositional OOD layer makes memorization harder by sampling topology,
  hazard type, conditions, and object novelty independently.
- The gauntlet suite is deliberately not saturated; it exposes a measurable
  failure boundary instead of only reporting easy perfect scores.
- It has an optional trajectory-level AlpaSim bridge; richer camera/perception
  integration remains future work.

## Boundaries

- This is a lightweight 2D simulator, not AlpaSim physics or photorealistic sensor simulation.
- The current generator produces abstract obstacles, not camera-realistic assets.
- The honest Minor Commission claim is randomized long-tail scenario design and reproducible closed-loop evaluation.
- AlpaSim integration should be described precisely: a trajectory-level plugin
  exists, but full sensor/perception integration is not implemented.

## Minor Submission Checklist

- [ ] Show at least three clusters and two seeds in the video/slide deck.
- [ ] Include generated SVGs as visual proof of scenario variation.
- [ ] Include JSON snippets showing `scenario.cluster` and `scenario.tags`.
- [ ] Report the deterministic test command: `uv run --no-sync python -m unittest discover -s tests`.
- [ ] Keep the simulation-environment claim separate from the Grand architecture claim.
