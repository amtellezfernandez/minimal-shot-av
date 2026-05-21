# Repo Audit Map

Use this file to route yourself quickly. Do not start by reading the whole repo.

## 1. Simulation Stack

Use this when the question is:

- how Spotlight Reflex works
- how long-tail scenarios are generated
- how simulator evaluation is measured
- how AlpaSim fits into the same stack

Start here:

- [`simulation.md`](simulation.md)
- [`../src/minimal_shot_av/simulator/README.md`](../src/minimal_shot_av/simulator/README.md)

Primary code roots:

- `src/minimal_shot_av/simulator/environment.py`
- `src/minimal_shot_av/simulator/spotlight_reflex.py`
- `src/minimal_shot_av/simulator/compositional_scenarios.py`
- `src/minimal_shot_av/simulator/compass.py`

### AlpaSim inside the simulation stack

Audit AlpaSim as a subsection of simulation, not as a separate top-level stack.

Start here:

- [`docs/corl2027/AUDIT.md`](corl2027/AUDIT.md)
- [`third_party/alpasim_overrides/README.md`](../third_party/alpasim_overrides/README.md)

Internal integration code:

- `src/minimal_shot_av/simulator/alpasim_signal.py`
- `src/minimal_shot_av/simulator/alpasim_spotlight.py`
- `src/minimal_shot_av/simulator/alpasim_token_bc.py`
- `src/minimal_shot_av/simulator/alpasim_direct_actor_planner.py`

Simulator-scoped third-party boundary:

- `third_party/alpasim_overrides/`

## 2. Waymo / WOD-E2E Stack

Use this when the question is:

- how Waymo frames are parsed
- how candidates are generated
- how the selector is trained
- how official submission archives are written

Start here:

- [`wod-e2e-system-walkthrough.md`](wod-e2e-system-walkthrough.md)
- [`../src/minimal_shot_av/model/README.md`](../src/minimal_shot_av/model/README.md)

Primary code roots:

- `src/minimal_shot_av/model/wod_e2e.py`
- `src/minimal_shot_av/model/kinematic_candidates.py`
- `src/minimal_shot_av/model/learned_trajectory_model.py`
- `src/minimal_shot_av/model/wod_ranker.py`
- `src/minimal_shot_av/model/wod_submission.py`

## 3. Neutral Reports / Cross-Cutting Analysis

Use this when the question is:

- how benchmark outputs are compared
- where analysis/report generation lives outside one subsystem

Start here:

- [`../src/minimal_shot_av/neutral/README.md`](../src/minimal_shot_av/neutral/README.md)

Primary code roots:

- `src/minimal_shot_av/neutral/benchmark_reports.py`
- `src/minimal_shot_av/neutral/benchmark_compare.py`
- `src/minimal_shot_av/neutral/alpasim_metrics.py`

## Fast Routing

| If you want... | Go here |
|---|---|
| simulator code | `src/minimal_shot_av/simulator/` |
| simulator doc | `docs/simulation.md` |
| AlpaSim reproduction | `docs/corl2027/AUDIT.md` |
| AlpaSim override boundary | `third_party/alpasim_overrides/` |
| Waymo parser / selector code | `src/minimal_shot_av/model/` |
| WOD walkthrough | `docs/wod-e2e-system-walkthrough.md` |
| reporting helpers | `src/minimal_shot_av/neutral/` |

## Boundary Rule

- `src/minimal_shot_av/**` is first-party project code.
- `third_party/alpasim_overrides/**` is simulator-scoped override material.
- `alpasim/**` is a separate nested checkout, not core repo source.
- `waymo-open-dataset/**` is a separate nested checkout, not core repo source.

## Recommended Audit Order

1. [`README.md`](../README.md)
2. this file
3. [`../src/minimal_shot_av/README.md`](../src/minimal_shot_av/README.md)
4. [`simulation.md`](simulation.md)
5. [`wod-e2e-system-walkthrough.md`](wod-e2e-system-walkthrough.md)
6. [`corl2027/AUDIT.md`](corl2027/AUDIT.md)

That order keeps the architecture, simulation stack, benchmark stack, and CoRL audit
surface separate without inventing extra top-level categories.
