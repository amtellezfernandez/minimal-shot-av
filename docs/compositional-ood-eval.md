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

## Metrics

The evaluator reports success, collision, reached-goal, safe-stall, final goal
distance, steps, min clearance, 5th-percentile clearance, average progress,
comfort cost, intervention rate, decision mode count, suite, topology, primary
hazard, difficulty, and OOD axes.

For the `gauntlet` suite the evaluator also treats route completion as
insufficient. A run must avoid collision, avoid near misses, keep intervention
rate low, and maintain progress. This produces `benchmark_pass`, aggregated as
`benchmark_pass_rate`.

The intended story is not a polished perfect score. WOD demonstrates baseline
competence; compositional OOD demonstrates generalisation; adversarial exposes
the current failure boundary; gauntlet is the deliberately hostile benchmark.

The novel-object audit isolates cases containing abstract unknown obstacle
labels such as `piano`, `parade_float`, and `portable_toilet`. Passing this
audit means the closed-loop policy handles unseen object categories through
geometry and safety margins in the simulator. It does not establish
camera-based recognition or semantic understanding of real physical unknown
objects.
