# Minimal-Shot AV

Submission for **SoTA Commission I: Minimal-Shot Autonomy** — two tracks evaluated
independently:

- **Grand Commission:** Spotlight Reflex closed-loop policy + WOD-E2E trajectory
  benchmark harness.
- **Minor Commission:** Custom simulation environment + AlpaSim trajectory plugin.

---

## Submission Documentation

- [`docs/writeup.md`](docs/writeup.md) — **Start here.** ≤2-page write-up:
  motivation, architecture, what worked, what didn't, where the prize goes.
- [`docs/architecture-deep-dive.md`](docs/architecture-deep-dive.md) — full
  architecture walkthrough: Spotlight Reflex design, WOD-E2E harness, what worked,
  what did not work, honest performance claims.
- [`docs/minor-commission.md`](docs/minor-commission.md) — simulation environment:
  abstract simulator and AlpaSim adapter (same policy at two fidelity levels).
- [`docs/wod-e2e-system-walkthrough.md`](docs/wod-e2e-system-walkthrough.md) —
  WOD-E2E pipeline, performance results, and per-cluster failure analysis.

---

## Thesis

A minimal-shot autonomy prototype for long-tail driving. The claim is narrow: a
closed-loop policy that uses explicit scene structure, maneuver hypothesis
enumeration, and safety-aware trajectory selection to navigate unfamiliar scenarios
without AV-dataset fine-tuning, route memorisation, or leaderboard-specific
supervised training.

Two assets:

- **Spotlight Reflex** — closed-loop policy backed by a 2D abstract simulator and
  deployed as an AlpaSim trajectory plugin. Primary evidence: 350 OOD rollouts,
  0 collisions, 93.1% COMPASS benchmark pass rate.
- **WOD-E2E harness** — model-side trajectory benchmark pipeline with preference
  labels. Champion: 7.880 RFS (vs 7.022 constant-velocity baseline). Auxiliary
  evidence; not a strict zero-shot deployment claim.

---

## Target Benchmark

WOD-E2E / Vision-based End-to-End Driving:

- Input: ego history (12 s), initial speed, route intent
- Output: 5 s future trajectory, 20 waypoints at 4 Hz, in vehicle coordinates
- Evaluation: Rater Feedback Score (RFS)
- Local data: 479 preference-labeled validation frames from 93 TFRecord shards

---

## Repo Structure

```
src/minimal_shot_av/simulator/   Simulation: environment, policies, scenario generators
src/minimal_shot_av/model/       WOD-E2E: parsing, candidates, rankers, evaluation
src/minimal_shot_av/neutral/     Shared metrics (no simulator/model imports)
scripts/                         Demo, evaluation, sweep, and audit scripts
docs/                            Submission documentation (judge-facing)
docs/notes/                      Development notes (internal)
artifacts/                       Generated evaluation outputs
benchmarks/                      Benchmark result JSON archives
models/                          Model declaration
```

---

## Quickstart

Requires Python 3.10+ and `uv`.

Run a single demo rollout:

```bash
uv run --no-sync python scripts/run_demo.py \
  --policy spotlight-reflex \
  --scenario-cluster spotlight \
  --seed 3 \
  --artifacts-dir artifacts/demo_spotlight
```

Full OOD evaluation sweep:

```bash
uv run --no-sync python scripts/evaluate_scenarios.py \
  --policy spotlight-reflex \
  --suite all \
  --seed-start 1 \
  --seed-end 10 \
  --output-dir artifacts/eval_all
```

AlpaSignal bridge audit:

```bash
uv run --no-sync python scripts/audit_alpasignal_bridge.py \
  --output artifacts/alpasignal_bridge_audit.json
```

COMPASS benchmark:

```bash
PYTHONPATH=src uv run --no-sync python -m minimal_shot_av.simulator.compass ladder \
  --policy spotlight-reflex \
  --profile compass-v0 \
  --seed-start 1 \
  --seed-end 10 \
  --output artifacts/compass_ladder.json
```

Fast test suite:

```bash
uv run --no-sync python scripts/run_tests.py --quick
```

Regenerate visual artifacts (GIFs):

```bash
uv run --no-sync python scripts/generate_submission_visuals.py
```

---

## Available Scenario Clusters

`construction` · `intersection` · `pedestrian` · `cyclist` · `multi-lane maneuver`
· `single-lane maneuver` · `cut-in` · `foreign object debris` · `special vehicle`
· `spotlight` · `others`

OOD suites: `compositional` · `adversarial` · `gauntlet` · `hidden`
