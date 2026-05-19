# AlpaSim partial bridge: pre-impact adapter residuals

This analysis uses only committed artifacts. It does not require rerunning AlpaSim.

## What is exact

- First-impact collision scenes: `18`; baseline adapter structured hazards are zero in `18/18`.
- At the same baseline impact frames, the world-frame oracle proxy has positive hazards in `18/18` scenes.
- Median oracle hazard count at impact: `22.500`; median closest clearance: `0.320 m`.

## Pre-impact window

- Collision actionable frames: `1027` over `18` scenes (lead >= `5` frames).
- Median oracle hazard count: `22.000`; median closest clearance: `5.588 m`; closest actor is rear-lane in `0.092` of frames.
- Direct-grid selected proxy-collision-free rate: `0.937`; low-margin (<0.55 m) selected rate: `0.104`.

## Terminal-matched non-collision control

- Control frames: `12324` from `12` non-collision scenes, matched by lead-to-end frame offsets.
- Median oracle hazard count: `4.000`; median closest clearance: `14.961 m`; closest actor is rear-lane in `0.002` of frames.

## Controlled-perturbation alignment

- Internal `latency_3` raw collision is `0.396`; hybrid collision is `0.146`.
- Supported bridge claim: The committed audits support an actor-visibility bridge in the collision-critical window: baseline adapter logs have zero structured hazards at first impact while the world-frame oracle proxy has positive actor hazards at the same impact frames.
- Not supported without rerun: The committed audits do not support full per-frame route, heading, lane-scale, or feature-noise residual distributions; those require regenerating raw adapter selection logs.

## Paper-safe interpretation

The committed audits support a partial bridge for actor-state corruption in the collision-critical window. They do not yet support a full residual-density bridge for route offset, heading bias, lane scale, or feature noise. A full regenerate should therefore be targeted at raw adapter logs for those axes, not at redesigning the method.
