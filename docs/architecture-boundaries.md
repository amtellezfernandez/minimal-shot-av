# Architecture Boundaries

This repository contains two independent SoTA Commission deliverables:

- `simulator`: the COMPASS/AlpaSim/Spotlight Reflex simulation and evidence stack.
- `zero-shot model`: the WOD-E2E/RFS non-text candidate and ranker stack.

They must stay agnostic to each other. Simulator work must not use model outputs,
training labels, or WOD preference-ranker logic. Zero-shot model
work must not import simulator policy, planner, environment, scenario generator,
or AlpaSim adapter logic.

There is no shared model/simulator scoring layer. WOD/RFS metric code belongs to
`minimal_shot_av.model.rfs_metric`; simulator trajectory selection belongs to
`minimal_shot_av.simulator.trajectory_selector`. Keeping these separate prevents
simulator work from being tuned against the zero-shot model objective.

Physical package ownership:

- `minimal_shot_av.simulator`: COMPASS, AlpaSim, scenario generation,
  policies, planner, environment, perception, safety, and renderer code.
- `minimal_shot_av.model`: WOD-E2E readers, non-text candidate generation,
  ranking, resampling, and model-side evaluation code.
- `minimal_shot_av.neutral`: evidence utilities that are allowed to be imported
  by both deliverables, but cannot import either deliverable.
- Judging and submission audits that read both model and simulator artifacts,
  such as `scripts/audit_sota_judging_criteria.py`, are neutral scripts. They
  may aggregate evidence, but must not import model or simulator implementation
  packages.

Module ownership:

- Simulator modules: `environment`, `perception`, `planner`, `policy`,
  `spotlight_reflex`, `trajectory_selector`, `compass`, `certification`, `alpasim_spotlight`,
  `compositional_scenarios`, `wod_scenarios`, `oracle`, `safety`,
  `world_model`, and `render`.
- Zero-shot model modules: `wod_e2e`, `wod_preference`, `wod_ranker`,
  `zero_shot_eval`, `rfs_metric`, `trajectory_io`, `trajectory_resampling`,
  `kinematic_candidates`, and `learned_trajectory_model`.
- Neutral modules: `alpasim_metrics`.

The boundary is enforced by:

```bash
python3 -m unittest tests.test_architecture_boundaries
```

The test covers both package modules and Python workflow scripts. Shell setup
scripts are treated as environment bootstrap and must not be used to route model
outputs into simulator evaluation.

Any score improvement must report which side it affects. Simulator experiments
can only change simulator-owned files and must be backed by simulator benchmark
artifacts. Model experiments can only change model-owned files and must be backed
by WOD-E2E/RFS artifacts.
