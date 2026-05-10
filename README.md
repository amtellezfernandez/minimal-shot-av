# Spotlight Reflex: Minimal-Shot Autonomous Driving

A closed-loop driving policy that reasons through unfamiliar long-tail scenarios
without fine-tuning on any AV dataset.

---

![Spotlight Reflex navigating a wrong-way actor at night](docs/images/spotlight_success.gif)

*Spotlight Reflex — wrong-way actor, low visibility, night conditions. Policy selects
`evasive_right`, clears the actor, recovers to lane centre. No training data used.*

![Baseline policy failing the same scenario](docs/images/baseline_spotlight.gif)

*Baseline policy on the same scenario — no world-state reasoning, continues into
the actor. This is what trajectory imitation without scene understanding produces.*

---

## What This Is

A minimal-shot autonomy prototype for Waymo's WOD-E2E long-tail benchmark.
Built entirely from scratch: a 2D closed-loop simulator with procedural scenario
generation for all 11 WOD-E2E cluster types, a compositional OOD generator, an
AlpaSim trajectory plugin, and a preference-calibrated WOD-E2E benchmark harness.

The core claim is narrow: if you make world-state reasoning and maneuver enumeration
explicit, the system generalises to novel scenarios without having seen them. A
trajectory that avoids obstacles, respects corridor geometry, and maintains progress
is correct whether the obstacle is a cone, a fallen tree, or a sheep.

---

## More Scenarios

![Construction zone — cone field, narrow corridor, lane closure](docs/images/construction_success.gif)

*Construction zone — narrow corridor (5.2 m), cone field, lane closure. Policy selects
`nudge_right` and maintains progress through the gap.*

![Intersection stress — conflict zone, two crossing actors](docs/images/intersection_stress.gif)

*Intersection stress — conflict zone, two crossing actors at different timings, 207 steps.*

## AlpaSim: Sensor-Realistic Validation

The same policy runs inside Waymo's AlpaSim simulator via a custom adapter. Real
WOD-E2E front-camera frames (sensor input) alongside the adapter's maneuver decision,
obstacle pressure, and route blockage readout.

![AlpaSim sensor input and reasoning output](docs/images/alpasim_reasoning_panel.png)

---

## Results

**350 rollouts across 5 suites — 0 collisions total:**

| Suite | Runs | Pass |
|-------|------|------|
| WOD-style (all 11 clusters) | 110 | 100% |
| Compositional OOD | 60 | 100% |
| Adversarial (2–3 hazards) | 60 | 100% |
| Hidden holdout | 60 | 100% |
| Gauntlet (4 hazards, narrow) | 60 | 60% |

COMPASS composite score: **9.137 / 10** across 700 ranked runs (threshold 7.0).  
95% CI collision rate: **[0.0, 0.0053]**.

**Gauntlet comparison — same 120 scenarios, two policies:**

| Policy | Pass | Collisions |
|--------|------|-----------|
| Baseline (no world-state reasoning) | 0 / 120 | **55** |
| Spotlight Reflex | 72 / 120 | **1** |

**WOD-E2E: 7.880 RFS** vs 7.022 constant-velocity baseline on the 479-frame
preference contract using a stability-selected HGB ranker. Oracle gap: 1.188 RFS
(the discriminator, not the candidate pool, is the bottleneck).

**AlpaSim**: same policy, sensor-realistic, `collision_at_fault: 0.0`, `dist_to_gt: 0.42 m`.

## What Didn't Work

- **No camera perception in the WOD-E2E path** — processes ego history, speed, and
  route intent only. Cannot see pedestrians, debris, or traffic signals. This is the
  honest reason the result sits below the 8.05 leaderboard top.
- **Visual embeddings didn't transfer** — InternVLA and Cosmos tokenizer embeddings
  attached as ranker features produced no confirmed RFS gain. Fine-tuning on
  preference labels is required to align the embedding space to this signal.
- **Turn calibration gap** — GO_LEFT: 6.782 RFS selected vs 8.535 oracle (regret 1.753).
  The training distribution is dominated by straight-ahead frames.

## Next Steps

The bottleneck is the discriminator, not the candidates. A lightweight camera encoder
fine-tuned on WOD-E2E preference labels should close 0.5–1.5 RFS of the 1.188-point
oracle gap. Waymo train-split access would move calibration off the validation set.
The test frame list (1,505 frames) and submission pipeline are both ready.

---

## Documentation

- [`docs/writeup.md`](docs/writeup.md) — 2-page submission write-up
- [`docs/presentation.md`](docs/presentation.md) — slide deck with full narrative arc
- [`docs/architecture-deep-dive.md`](docs/architecture-deep-dive.md) — complete
  architecture, what worked, what didn't, and how to reproduce every result
- [`docs/minor-commission.md`](docs/minor-commission.md) — simulation environment:
  abstract simulator and AlpaSim adapter
- [`docs/wod-e2e-system-walkthrough.md`](docs/wod-e2e-system-walkthrough.md) —
  WOD-E2E pipeline and per-cluster failure analysis

---

## Quickstart

```bash
# Single demo rollout
uv run --no-sync python scripts/run_demo.py \
  --policy spotlight-reflex \
  --scenario-cluster spotlight \
  --seed 3 \
  --artifacts-dir artifacts/demo_spotlight

# Full OOD evaluation sweep
uv run --no-sync python scripts/evaluate_scenarios.py \
  --policy spotlight-reflex \
  --suite all \
  --seed-start 1 \
  --seed-end 10 \
  --output-dir artifacts/eval_all

# Test suite
uv run --no-sync python scripts/run_tests.py --quick
```

Available clusters: `construction` · `intersection` · `pedestrian` · `cyclist`
· `cut-in` · `foreign object debris` · `special vehicle` · `spotlight` · `others`
