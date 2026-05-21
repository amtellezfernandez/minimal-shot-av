# AlpaSim Override Boundary

This directory is the explicit override zone for the AlpaSim part of the simulation
stack.

Use it when the question is:

- what had to be patched outside the core repo code
- what is first-party adapter code vs modified AlpaSim-side material
- what belongs to the simulator audit surface but is not first-party source

## Contents

- `route_waypoints.patch` — tracked patch for route-waypoint bridge behavior
- `local_checkout.patch` — local checkout patch material
- `Dockerfile.amd64` — runtime image customization for supported hosts
- `src/wizard/**` — tracked wizard/deployment overrides
- `src/driver/**` — tracked external-driver override files

## Boundary Rule

These files are not the main simulator implementation and not the WOD model stack.
They still belong to the simulation audit surface because the AlpaSim reproduction path
depends on them.

The corresponding first-party integration code lives in:

- [`src/minimal_shot_av/simulator/README.md`](../../src/minimal_shot_av/simulator/README.md)

The corresponding audit / reproduction path lives in:

- [`docs/corl2027/AUDIT.md`](../../docs/corl2027/AUDIT.md)
