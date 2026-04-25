# Minimal-Shot AV

Submission scaffold for **SoTA Commission I: Minimal-Shot Autonomy**.

This repository is set up to support a focused prototype around a simple claim:

> an autonomous agent should generalize to unseen layouts by reasoning over scene structure at runtime, not by replaying memorized routes.

## Submission track

Primary target: **Grand Commission (£5,000) for overall best autonomy architecture**.

The simulator in this repo is support infrastructure. The actual submission angle is the policy architecture:

- minimal-shot scene understanding
- explicit world modeling instead of route memorization
- online decision-making in unseen environments
- realistic constraints on compute and latency

## Current architecture direction

The recommended direction for this repo is:

- visual or scene input into a compact world model
- object, free-space, and affordance extraction
- short-horizon planner with explicit safety checks
- fallback behavior under uncertainty
- evaluation on randomized unseen scenarios

This structure is stronger for the architecture prize than a pure simulator play because it makes a clear technical claim about generalization.

## Current baseline

The starter implementation is a lightweight 2D navigation benchmark:

- procedurally generated lanes and obstacle fields
- no stored route library or map-specific tuning
- a reactive policy that follows corridor geometry and avoids hazards online
- artifact generation for demos and documentation

It is not intended to be the final submission system. It is a clean baseline and submission skeleton you can extend into:

- a modular autonomy architecture
- a multimodal world model
- a planner with uncertainty-aware fallback
- a WOD-E2E evaluation track
- a robotics or off-road autonomy variant

## Repo structure

- `src/minimal_shot_av/`: simulation and baseline policy
- `scripts/run_demo.py`: generate a random scenario and rollout artifacts
- `docs/`: write-up, video, and submission planning
- `models/`: model declarations and architecture notes
- `notebooks/`: analysis workflow placeholder
- `artifacts/`: generated outputs

## Quickstart

Use Python 3.10+.

```bash
python3 scripts/run_demo.py
```

This writes:

- `artifacts/latest_rollout.json`
- `artifacts/latest_rollout.svg`

## Submission deliverables covered here

This scaffold prepares the repo for the commission requirements:

- GitHub repo structure and narrative
- architecture-first submission framing
- base-model and architecture declaration template
- analysis notebook plan
- short write-up template
- 1-5 minute video outline
- runnable baseline demo for early visuals

## Recommended next steps

1. Lock the target domain and architecture thesis in `docs/grand-commission-plan.md`.
2. Replace the reactive baseline with a modular perception-world-model-planner stack.
3. Add latency and hardware constraints to the evaluation loop.
4. Record both one success case and one understood failure case for the video.
5. If pursuing WOD-E2E, add a dedicated `wod_e2e/` package and keep the current simulator as a fast development harness.
