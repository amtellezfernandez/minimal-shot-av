# External / Patched-Upstream Boundary

Everything under `third_party/` is external-origin material, but it is **not**
necessarily untouched upstream code.

This directory exists so an auditor can distinguish:

- first-party repo code in `src/minimal_shot_av/**`
- **patched upstream work** that was materially modified for this project
- separate nested upstream checkouts such as `alpasim/` and `waymo-open-dataset/`

## Current Contents

- [`alpasim_overrides/`](alpasim_overrides/) — tracked overrides and patches for the
  simulation stack's AlpaSim integration path

## Attribution Rule

If a reviewer asks "is this your code or upstream code?", the honest answer is:

- `src/minimal_shot_av/**` -> first-party repo code
- `third_party/alpasim_overrides/**` -> upstream-derived AlpaSim code or patches with
  project-authored modifications
- `alpasim/**` -> separate upstream checkout
- `waymo-open-dataset/**` -> separate upstream checkout

So `third_party/alpasim_overrides/**` should be read as **patched upstream work**,
not as "we did nothing here."
