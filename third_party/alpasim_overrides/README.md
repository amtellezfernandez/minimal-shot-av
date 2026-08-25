# Patched Upstream AlpaSim Work

This directory is the explicit **patched upstream** zone for the AlpaSim part of the
simulation stack.

Use it when the question is:

- what had to be patched outside the core repo code
- what is first-party adapter code vs modified AlpaSim-side material
- what belongs to the simulator audit surface but is not first-party source

## What This Means

These files are not being presented as untouched third-party source.

They represent upstream AlpaSim surface area that required real project work:

- bug fixes
- bridge changes
- deployment/runtime adjustments
- integration-specific modifications needed to make the simulator transfer path work

So the correct label is:

- **patched upstream work**

not:

- "just external code"

## Contents

- `route_waypoints.patch` — tracked patch for route-waypoint bridge behavior
- `local_checkout.patch` — local checkout patch material
- `Dockerfile.amd64` — runtime image customization for supported hosts
- `src/wizard/**` — tracked wizard/deployment overrides
- `src/driver/**` — tracked external-driver override files

## Provenance & Licensing

- The patches and override files here are unified diffs and drop-in replacements
  against a **separate upstream AlpaSim checkout** that lives (gitignored) at
  `workspace/alpasim/` and is **not redistributed** with this repository.
- Because the upstream source is not shipped here, its license text is not
  reproduced here either. Anyone applying these patches must obtain AlpaSim
  from its owner and comply with the upstream license terms.
- Copyright in the unmodified upstream fragments quoted inside the diffs remains
  with the upstream AlpaSim authors; the project-authored modifications are
  covered by the repo root `LICENSE` (Apache-2.0).
- When regenerating these patches, record the upstream commit hash they were
  produced against in this README. (The historical patches predate this rule;
  their base commit was not recorded.)

## Boundary Rule

These files are not the main simulator implementation and not the WOD model stack.
They still belong to the simulation audit surface because the AlpaSim reproduction path
depends on them, and because project-authored modifications were made here.

The corresponding first-party integration code lives in:

- [`src/minimal_shot_av/simulator/README.md`](../../src/minimal_shot_av/simulator/README.md)

The corresponding audit / reproduction path lives in:

- [`docs/corl2027/AUDIT.md`](../../docs/corl2027/AUDIT.md)
