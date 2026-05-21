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
- `release/hf/` stores Hugging Face release packaging metadata and should stay separate

## Core Audit Outputs

- `corl_evidence_strength_audit.md`
- `transfer_predictor_analysis.md`
- `alpasim_transfer_matrix_partial.md`

## Deeper Follow-Ups

- `alpasim_collision_surface_30scene_audit.*`
- `alpasim_candidate_counterfactual_30scene.*`
- `alpasim_partial_bridge_preimpact.*`
- `alpasim_actor_blindness_30scene_raw_analysis.*`
- `alpasim_matrix10_analysis.*`
- `alpasim_matrix10_with_axis_analysis.*`
