# Third-Party Boundary

Everything under `third_party/` is explicit non-core material: overrides, patches, or
adapter-facing external code boundaries.

This directory exists so an auditor can distinguish:

- first-party repo code in `src/minimal_shot_av/**`
- external override material in `third_party/**`
- separate nested upstream checkouts such as `alpasim/` and `waymo-open-dataset/`

## Current Contents

- [`alpasim_overrides/`](alpasim_overrides/) — tracked overrides and patches for the
  simulation stack's AlpaSim integration path

## Rule

If a reviewer asks "is this your code or upstream code?", the default answer is:

- `src/minimal_shot_av/**` -> ours
- `third_party/alpasim_overrides/**` -> simulation-scoped external override zone
- `alpasim/**` -> separate checkout
- `waymo-open-dataset/**` -> separate checkout
