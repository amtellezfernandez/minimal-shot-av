---
marp: true
theme: default
paginate: true
size: 16:9
title: Minimal-Shot Autonomous Driving
description: Minimal-shot AV architecture, transfer diagnostics, and WOD-E2E grounding
---

<!--
Minimal-shot AV presentation deck.
Render HTML:
  npx @marp-team/marp-cli presentation.md --html --allow-local-files -o presentation.html
Render PDF:
  npx @marp-team/marp-cli presentation.md --pdf --allow-local-files -o presentation-sota.pdf
PDF export requires Chrome, Chromium, Edge, or Firefox on the host.
-->

# Minimal-Shot Autonomous Driving

## Geometry-Grounded Runtime, Transfer Diagnostics, WOD-E2E Grounding

Alba Maria Tellez Fernandez<br>
GitHub branch: `CoRL-2027`

This stack is built to handle unfamiliar long-tail traffic structure through
explicit geometry, bounded actions, and auditable transfer tests.

---

# Motivation

Autonomy that depends on collecting every rare case will always lag reality.

Minimal-shot autonomy needs a different capability:

| Standard AV scaling | Minimal-shot target |
| --- | --- |
| train on many mapped cases | reason from first principles |
| optimize average-case driving | survive long-tail scenarios |
| hide failures inside aggregate metrics | expose failures by scenario and axis |
| expensive data dependence | reusable simulator and audit harness |

The repo connects runtime, simulator, transfer harness, and real-data grounding
in one reproducible pipeline.

---

# System Contributions

| Layer | What we built | Why it matters |
| --- | --- |
| runtime | Spotlight Reflex token policy | auditable geometry-first decision path |
| simulator | custom 2D long-tail environment | controlled causal stress testing |
| learning probes | BC, RNN, DAgger, scorer variants | isolates what imitation actually learns |
| transfer harness | AlpaSim proxy-state adapter | exposes external failure modes |
| grounding track | WOD-E2E selector harness | tests visual/world-model priors on real data |

The stack is useful because these layers connect into one evidence chain rather
than one isolated benchmark.

---

# Why Grounding, Not Language

![width:1000](docs/images/grounded_vla_comparison.svg)

Recent VLA work, including LaST-VLA, argues that driving decisions should be
grounded in continuous geometry and dynamics rather than long textual reasoning.

Our implementation is smaller and explicit: no full VLA training, but the same
principle appears as a bounded geometry-to-token control stack.

---

# Whole System Map

![width:1120](docs/images/system-map-sota-corl.svg)

Three connected layers:

1. custom simulator for controlled minimal-shot experiments
2. WOD-E2E selector harness for real-data grounding probes
3. AlpaSim transfer harness for sensor-realistic external validation

---

# Runtime Architecture

![width:1120](docs/images/spatiotemporal_geometry_dynamics_action.svg)

Spotlight Reflex is organized as a spatial-temporal compression path:

- scene signals are compressed into spatiotemporal pattern
- pattern is resolved into geometry and dynamic risk
- action stays bounded through the 9-token interface
- the last step is explicit control, not hidden language-like chaining

The runtime does not use scene IDs, maps of prior episodes, or dataset-specific
fine-tuning.

---

# Explicit Physical Bottleneck

![width:1000](docs/images/explicit_geometry_manifold.svg)

The physical bottleneck is visible: proxy geometry, clearance, dynamics, and the
9-token action set. This is why the same policy can be inspected in the custom
simulator, corrupted internally, and bridged into AlpaSim.

---

# Token Interface

The same action interface is used by the rule policy, learned policies, and
AlpaSim transfer experiments.

| Token family | Examples | Role |
| --- | --- | --- |
| longitudinal safety | stop, crawl, slow_yield | reduce collision pressure |
| nominal progress | maintain | continue along route |
| mild lateral evasion | nudge_left, nudge_right | create local clearance |
| emergency evasion | evasive_left, evasive_right | handle close hazards |
| recovery | lane_recover | return to lane center |

This shared token interface makes failures diagnosable: we can separate bad
candidate geometry from bad token selection.

---

# Simulator Environment

![width:1120](docs/images/simulator-evaluation-loop.svg)

The simulator is both the deployment sandbox and the controlled lab.

It supports WOD-style long-tail clusters, compositional OOD scenarios,
adversarial multi-hazard scenes, hidden holdout profiles, and Latin-hypercube
stress sampling.

---

# Demo: Minimal-Shot Success And Failure

| Baseline failure | Geometry-grounded success |
| --- | --- |
| ![width:430](docs/images/baseline_spotlight.gif) | ![width:430](docs/images/spotlight_success.gif) |

Same scenario family: wrong-way actor, low visibility, night.

The baseline extrapolates into the hazard. Spotlight selects an evasive token,
clears the actor, and returns to lane center.

---

# More Long-Tail Scenarios

| Construction zone | Foreign object debris |
| --- | --- |
| ![width:430](docs/images/construction_success.gif) | ![width:430](docs/images/fod_success.gif) |

The point is controlled variation of geometry, hazards, clearances, and
recovery choices.

---

# Wins Snapshot

| Win | Result | Why it matters |
| --- | --- | --- |
| closed-loop internal safety | `350` rollouts, `0` collisions, `93.1%` overall pass | runtime is not a toy one-off demo |
| baseline gap on hardest internal suite | `57.6%` vs `2.1%` pass, `7.9%` vs `20.5%` collision | geometry-grounded selection changes behavior materially |
| real-data grounding signal | `7.845` RFS on WOD-E2E validation-CV | frozen scene priors help when the selector head is right |

These are three different wins: internal closed-loop competence, strong baseline
separation, and non-trivial grounding signal on real WOD-E2E scenes.

---

# Primary Simulation Result

Closed-loop simulation evidence package:

| Suite | Runs | Collisions | Pass |
| --- | ---: | ---: | ---: |
| WOD-style, all 11 clusters | 110 | 0 | 100% |
| Compositional OOD | 60 | 0 | 100% |
| Adversarial | 60 | 0 | 100% |
| Hidden holdout | 60 | 0 | 100% |
| Gauntlet | 60 | 0 | 60% |
| Total | 350 | 0 | 93.1% |

COMPASS: **9.137 / 10** over 700 ranked runs.

Source: `README.md`, `artifacts/compass_evidence_report.json`.

---

# Baseline Comparison

Matched gauntlet comparison: same 420 scenarios, same seeds.

| Policy | Pass rate | Collision rate |
| --- | ---: | ---: |
| Baseline, no world-state reasoning | 2.1% | 20.5% |
| Spotlight Reflex | **57.6%** | **7.9%** |

Explicit geometric reasoning gives a large improvement over a
non-reasoning baseline in the randomized challenge environment.

---

# AlpaSim External Transfer

![bg right:45% width:92%](docs/images/alpasim_token_dagger_iter2_30scene.gif)

This is a WOD-E2E front-camera AlpaSim rollout with adapter map and metrics.

AlpaSim is not used to claim production safety. It is used to test whether the
token interface survives a different observation and execution stack.

---

# AlpaSim Adapter

![width:1120](docs/images/alpasim-transfer-stack.svg)

The policy is held fixed. What changes is the observation adapter and the
execution interface.

This makes the transfer failure interpretable instead of hand-wavy.

---

# Transfer Diagnostic Schema

![width:1120](docs/images/proxy_transfer_axes_static.svg)

The same learned token policies are tested
inside the simulator, under controlled proxy corruption, and in AlpaSim.

---

# What Failed Externally

10 matched WOD-E2E clips completed by all four learned variants:

| Model | Collision | Offroad | Wrong lane | Progress | Dist. m |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw DAgger iter2 | 0.600 | 0.900 | 0.700 | 0.034 | 61.98 |
| Clamped lateral | 0.700 | 0.500 | 0.200 | 0.384 | 59.85 |
| Hard-veto hybrid | 0.800 | 0.200 | 0.700 | 0.837 | 166.06 |
| Source decay | 0.600 | 0.900 | 0.400 | 0.175 | 62.92 |

No single learned intervention fixed all axes. This is the main understood
failure case: transfer breaks into collision, offroad, lane, and progress axes.

Source: `artifacts/alpasim_matrix10_analysis.md`.

---

# WOD-E2E Auxiliary Track

479 WOD-E2E validation frames, segment-grouped 5-fold CV.

| Selector | RFS | Folds | Role |
| --- | ---: | ---: | --- |
| Waymo baseline | 7.022 | - | reference |
| Local baseline | 7.131 | - | reference |
| Gate-only | 7.803 | 5 | selector baseline |
| RFF direct policy | 7.834 | 5 | learned baseline |
| GPU MLP + Cosmos 64d | **7.845** | 5 | best validation-CV |
| Oracle selector | 9.264 | 5 | upper bound |

This is auxiliary evidence: the right trajectory often exists, but selection
still needs stronger grounded scene understanding.

---

# Foundation-Prior Roadmap

![width:1000](docs/images/foundation_prior_roadmap.svg)

What we tested: frozen Cosmos and InternVLA priors as selector inputs.

What remains: preference-aligned geometry and dynamics adapters, connected to
the token selector and evaluated under external transfer axes.

---

# Reproducibility

Start here:

```text
README.md
presentation.md / presentation-sota.pdf
docs/corl2027/paper.pdf
docs/corl2027/AUDIT.md
./scripts/run_corl2027_audit.sh
```

Primary artifacts:

```text
artifacts/minimal_shot_claim_audit.json
artifacts/alpasim_matrix10_analysis.md
artifacts/compass_evidence_report.json
```

Branch:

```text
https://github.com/amtellezfernandez/minimal-shot-av/tree/CoRL-2027
```

---

# What This Unlocks

This is now a working testbed for grounded minimal-shot driving.

| Built asset | What it enables next |
| --- | --- |
| custom simulator with 11 long-tail clusters | controlled stress tests before external runs |
| explicit geometry-to-token runtime | camera-grounded hazard encoder |
| BC / DAgger / scorer / veto probes | axis-aware token selector |
| AlpaSim proxy-state bridge | 100+ paired external scenes |
| WOD-E2E Cosmos / InternVLA probes | preference-aligned visual adapters |

---

# Final Takeaway

The core contribution is a minimal-shot AV stack that makes the control path,
the stress environment, the transfer interface, and the failure modes all
explicit.

The strongest claim is concrete:

**the stack works internally, transfers unevenly across external axes, and
shows exactly where grounding and proxy-state mismatch start to matter.**

---
