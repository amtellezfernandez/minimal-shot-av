# Compositional OOD Evaluation

The WOD-E2E generator is useful for brief adherence, but it is not enough to
demonstrate minimal-shot generalisation. It is still organized around the 11
named WOD clusters.

This repo therefore includes a second simulator layer:

```text
independent topology
  x independent hazard modules
  x independent conditions
  -> constrained scenario with one declared primary decision point
```

## Suites

- `wod`: WOD-E2E inspired procedural clusters for direct brief alignment.
- `compositional`: hazards, topology, weather, visibility, and object type are sampled independently.
- `adversarial`: two or three independently sampled hazards are composed in the same route.
- `gauntlet`: four synchronized threats in narrow, low-visibility routes; raw goal completion is not enough to pass.
- `hidden`: holdout seed offset for a frozen-policy evaluation pass.
- `all`: runs every suite.

## Scenario Manifest

Every compositional scenario emits interpretable manifest fields in
`scenario.tags`:

- `primary_hazard_id`
- `primary_hazard_type`
- `intended_decision`
- `allowed_maneuvers`
- `difficulty`
- `ood_axes`
- `ambient_objects`
- `blocking_hazards`
- `hazard_composition`

This separates the hard decision point from ambient scene texture. Ambient
objects enrich the scene but are not collision hazards; blocking hazards are
placed deliberately and reported in the manifest.

## Commands

RLVR curriculum manifest:

```bash
uv run --no-sync python scripts/build_rlvr_curriculum.py --seed-start 1 --seed-end 20 --output artifacts/rlvr_curriculum_v1.json
```

Brief-aligned WOD sweep:

```bash
uv run --no-sync python scripts/evaluate_scenarios.py --policy both --suite wod --seed-start 1 --seed-end 20 --output-dir artifacts/eval_wod
```

Generalisation sweep:

```bash
uv run --no-sync python scripts/evaluate_scenarios.py --policy both --suite compositional --seed-start 1 --seed-end 20 --output-dir artifacts/eval_compositional
```

Adversarial stress sweep:

```bash
uv run --no-sync python scripts/evaluate_scenarios.py --policy both --suite adversarial --seed-start 1 --seed-end 20 --output-dir artifacts/eval_adversarial
```

Gauntlet benchmark sweep:

```bash
uv run --no-sync python scripts/evaluate_scenarios.py --policy both --suite gauntlet --seed-start 1 --seed-end 20 --output-dir artifacts/eval_gauntlet
```

Novel-object stress audit:

```bash
uv run --no-sync python scripts/audit_novel_object_stress.py --seed-start 1 --seed-end 80 --min-runs 24 --output artifacts/novel_object_stress_audit.json
```

Topology susceptibility audit:

```bash
uv run --no-sync python scripts/audit_seizure_topology_susceptibility.py --seeds-per-topology 3 --suite hidden --output-dir artifacts/seizure_topology_susceptibility
```

Paper-grade topology susceptibility sweep:

```bash
uv run --no-sync python scripts/audit_seizure_topology_susceptibility.py --seeds-per-topology 20 --suite hidden --sweep-family all --output-dir artifacts/seizure_topology_susceptibility_paper
```

## Metrics And Evidence Discipline

`artifacts/rlvr_curriculum_v1.json` makes the gym explicit:

- observation modes: privileged debug geometry, noisy geometry, top-down raster
  spec, and camera/scene embedding mode.
- action space: bounded waypoint and velocity commands.
- task curriculum: WOD-style, compositional, adversarial, gauntlet, and hidden
  tasks with deterministic train/dev/blind/stress splits.
- grader library: reach goal, no collision, trajectory safety, progress floor,
  intervention budget, and benchmark pass.

The evaluator reports success, collision, reached-goal, safe-stall, final goal
distance, steps, min clearance, 5th-percentile clearance, average progress,
maximum collision risk, maximum lane error, comfort cost, intervention rate,
decision mode count, suite, topology, primary hazard, difficulty, and OOD axes.
Each row is one closed-loop rollout: the policy observes the active scenario,
chooses actions, the safety filter may intervene, actors move with time, and the
next state is generated from the executed action.

The evaluator also reports trajectory-level safety, not only final state:

- `trajectory_safety_pass`: the run reached the goal without any intermediate
  safety event.
- `trajectory_safety_event_count`: number of intermediate safety events.
- `trajectory_safety_events`: compact trace markers for low clearance, extreme
  collision-risk spikes with tight executed clearance, or terminal collisions.
  Planner context flags such as a blocked corridor are reported through normal
  rollout fields, but they are not counted as safety violations unless they
  produce a physical-risk event.

For the `gauntlet` suite the evaluator also treats route completion as
insufficient. A run must avoid collision, avoid near misses, keep intervention
rate low, and maintain progress. This produces `benchmark_pass`, aggregated as
`benchmark_pass_rate`.

`scenario_eval.json` includes:

- `runs`: per-rollout action-trace summaries.
- `summary`: per-policy, per-suite, per-cluster aggregates with Wilson 95%
  confidence intervals for success and benchmark pass rate.
- `curriculum`: suite, cluster, topology, hazard, and difficulty coverage.
- `statistics`: pass@1 and trajectory-safety pass rates by policy and suite.

The intended story is not a polished perfect score. WOD demonstrates baseline
competence; compositional OOD demonstrates generalisation; adversarial exposes
the current failure boundary; gauntlet is the deliberately hostile benchmark.

The novel-object audit isolates cases containing abstract unknown obstacle
labels such as `piano`, `parade_float`, and `portable_toilet`. Passing this
audit means the closed-loop policy handles unseen object categories through
current obstacle geometry and safety margins in the simulator. The primary
runtime disables hidden actor-behavior forecasts and the minimal-shot audit
rejects scenario manifest or cluster-name lookup in active policy sources, so
this audit is less privileged than a simulator-oracle rollout. It does not
establish camera-based recognition or semantic understanding of real physical
unknown objects.

The topology susceptibility audit asks a different question: which route
geometries are most fragile under synchronized guard perturbations? It injects
phase-locked phantom guard obstacles at curvature and corridor decision points,
sweeps perturbation budget, and reports the first budget where at least half of
clean-passing scenarios collapse. The default command separates `radius_only`,
`count_only`, `placement_only`, and `coupled_stealth` sweeps so the reported
budget is not a single hidden mix of obstacle size, count, and location. Phantom
obstacles are capped at 1.45m radius in the stealth-bounded regime, and collapse
reasons are decomposed into collision, near miss, trajectory safety event,
excessive intervention, slow crawl, goal failure, or progress collapse. The
artifact also emits a stratified topology/hazard/condition table and a confound
audit. If collapses concentrate in one hazard family, the result should be
reported as topology-by-scenario-family susceptibility rather than a pure
topology effect. The result is a controlled seizure-like proxy for false
transition guards, not a claim that the simulator contains real neurological or
physical seizure dynamics.
