# SoTA Minor Commission Submission Bundle

Randomized long-tail simulation environment with WOD-style cluster coverage,
compositional OOD stress cases, and reproducible closed-loop evaluation.

## Demo Artifacts

- `minor_construction_seed1/`
- `minor_fod_seed2/`
- `minor_spotlight_seed3/`
- `minor_eval/`
- `minor_ood_eval/`
- `minor_alpasignal_bridge/`
- `minor_runtime/`
- `minor_visual_gallery/`

## Source Documents

- `README.md`
- `docs/minor-commission.md`
- `docs/architecture-deep-dive.md`

## Verification

`UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python scripts/run_tests.py`
