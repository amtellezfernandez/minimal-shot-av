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

---

## What Worked

- **350 OOD rollouts, 0 collisions, 93.1% COMPASS benchmark pass rate** across all
  scenario suites (compositional, adversarial, gauntlet, hidden holdout)
- **WOD-E2E: 7.880 RFS** vs 7.022 constant-velocity baseline (+0.86 RFS) on the
  479-frame preference contract using a stability-selected HGB ranker
- **AlpaSim integration** — same Spotlight Reflex policy runs inside Waymo's
  sensor-realistic simulator via a custom adapter. Route command, camera brightness,
  ego dynamics, and structured hazards all bridge to the policy's world-state
  representation without any AlpaSim-specific code in the policy itself

## What Didn't Work

- **No camera perception** — the WOD-E2E model path processes ego history, speed,
  and route intent only. It cannot see pedestrians, debris, or traffic signals.
  This is the honest reason the result sits below the 8.05 leaderboard top.
- **Visual embeddings didn't transfer** — InternVLA and Cosmos tokenizer embeddings
  attached as ranker features produced no confirmed RFS gain. Their embedding spaces
  are not aligned to the discriminative signal the ranker needs.
- **Turn calibration gap** — GO_LEFT frames are the worst slice: 6.782 RFS selected
  vs 8.535 oracle. The training distribution is dominated by straight-ahead frames.

## Where the Prize Goes

The bottleneck is the discriminator, not the candidates. A lightweight camera encoder
fine-tuned on WOD-E2E preference labels should close 0.5–1.5 RFS of the 1.19-point
oracle gap. The rest goes to Waymo train-split access, proper train/test split
calibration, and a live leaderboard submission.

---

## Documentation

- [`docs/writeup.md`](docs/writeup.md) — full submission write-up with all GIFs
- [`docs/architecture-deep-dive.md`](docs/architecture-deep-dive.md) — Spotlight
  Reflex architecture, WOD-E2E harness, and detailed what-worked / what-didn't
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
