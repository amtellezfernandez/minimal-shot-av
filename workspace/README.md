# Workspace Surface

This directory is the canonical home for large local-only checkouts and datasets that
support the repo but are not part of the CoRL audit surface.

Use it for:

- `alpasim/` — upstream AlpaSim checkout used by the simulator transfer path
- `waymo-open-dataset/` — upstream Waymo code checkout
- `waymo_open_dataset_end_to_end_camera_v_1_0_0/` — local WOD-E2E TFRecord tree
- other local external workspaces that should stay out of the tracked root

The tracked repo surface starts at the root README, `src/`, `docs/`, `tests/`,
`scripts/`, `artifacts/`, and `third_party/`.
