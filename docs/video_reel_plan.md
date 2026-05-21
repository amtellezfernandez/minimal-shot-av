# README / GIF Reel Plan

The README media should use embedded GIFs and static diagnostic panels. The GIFs are
internal visual debugging artifacts; they are not CoRL evidence.

1. same hazard, better decision
2. breadth across long-tail scene families
3. pressure under multi-agent timing
4. honest external transfer diagnostics

## Best Clip Order

1. **Spotlight Reflex vs baseline**
   Assets:
   `docs/images/spotlight_vs_baseline.gif`
   Caption:
   Same seed, same wrong-way actor, different decision surface.
   Why it works:
   This is the strongest opening because it shows causal contrast, not just a success clip.

2. **Construction corridor**
   Asset:
   `docs/images/construction_success.gif`
   Caption:
   Narrow lane closure, explicit maneuver adaptation, closed-loop recovery.
   Why it works:
   It reads as practical driving rather than a toy obstacle dodge.

3. **Intersection stress**
   Asset:
   `docs/images/intersection_stress.gif`
   Caption:
   Multi-agent timing pressure instead of single-obstacle geometry.
   Why it works:
   It widens the claim from one hazard type to interaction reasoning.

4. **Foreign object debris**
   Asset:
   `docs/images/fod_success.gif`
   Caption:
   Long-tail obstacle avoidance without collapsing route progress.
   Why it works:
   It adds another recognizable AV case with a distinct failure mode.

5. **AlpaSim transfer clip**
   Asset:
   `docs/images/alpasim_transfer.gif`
   Caption:
   The same repo also runs a sensor-realistic transfer diagnostic stack.
   Why it works:
   It upgrades the project from simulator-only to transfer-aware.

6. **AlpaSim reasoning panel**
   Asset:
   `docs/images/alpasim_reasoning_panel.png`
   Caption:
   Sensor input is surfaced beside the proxy-state reasoning path.
   Why it works:
   It makes the project feel inspectable and technical rather than opaque.

## Best Titles

Use these titles if you upload clips externally:

- `Minimal-Shot AV: Wrong-Way Actor, Same Seed, Different Outcome`
- `Spotlight Reflex: Closed-Loop Construction Corridor`
- `Spotlight Reflex: Intersection Stress Test`
- `Spotlight Reflex: Foreign Object Debris Avoidance`
- `AlpaSim Transfer Diagnostic: Front-Camera Rollout`

## Embedded Local GIFs

- `docs/images/spotlight_vs_baseline.gif`
- `docs/images/spotlight_success.gif`
- `docs/images/baseline_spotlight.gif`
- `docs/images/construction_success.gif`
- `docs/images/intersection_stress.gif`
- `docs/images/fod_success.gif`
- `docs/images/alpasim_transfer.gif`

## Best One-Line Captions

- `Geometry-first action selection under rare-hazard pressure.`
- `Closed-loop rollouts, not static path overlays.`
- `Same candidate interface audited through AlpaSim transfer diagnostics.`
- `Failure is measured per axis, not hidden in a single scalar score.`

## If You Only Show Three Things

Show these:

1. Spotlight Reflex vs baseline
2. Intersection stress
3. AlpaSim transfer clip

That is the shortest sequence that still sells novelty, competence, and honesty.
