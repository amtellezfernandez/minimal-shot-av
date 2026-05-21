# CoRL 2027 Paper Artifacts

This directory contains generated evidence that supports the CoRL paper and its audit
surface.

Use it for:

- paper-facing audit outputs
- transfer analyses cited by the paper
- follow-up collision-surface and counterfactual analyses

## Boundary

- `docs/corl2027/` explains the paper and audit path
- `artifacts/corl2027/` stores generated proof artifacts for that paper surface
- `artifacts/hf_release/` stores Hugging Face release packaging metadata and should stay separate

## Core Audit Outputs

- `corl_evidence_strength_audit.md`
- `transfer_predictor_analysis.md`
- `alpasim_transfer_matrix_partial.md`
- `alpasim_matrix10_analysis.md`

## Fresh External Refresh

Run fresh AlpaSim evidence and rebuild the paper-facing audit surface with:

```bash
scripts/refresh_corl2027_external_evidence.sh \
  --matrix-dir runs/alpasim_transfer_matrix_refresh_20260522_smoke10 \
  --scene-preset front_camera_10scene_smoke \
  --allow-existing
```

This wrapper does four things in order:

- checks AlpaSim readiness, including NVIDIA container runtime
- reruns the selected transfer matrix
- rebuilds `alpasim_matrix10_analysis.*`
- rebuilds the CoRL audit outputs under `artifacts/corl2027/`

If you also have fresh published/front-camera comparison run dirs, add:

```bash
  --published-ours-run <run-or-metrics> \
  --front-ours-run <run-or-metrics> \
  --front-alpamayo-run <run-or-metrics>
```

## Deeper Follow-Ups

- `alpasim_collision_surface_30scene_audit.*`
- `alpasim_candidate_counterfactual_30scene.*`
- `alpasim_partial_bridge_preimpact.*`
- `alpasim_actor_blindness_30scene_raw_analysis.*`
- `alpasim_matrix10_analysis.*`
- `alpasim_matrix10_with_axis_analysis.*`
