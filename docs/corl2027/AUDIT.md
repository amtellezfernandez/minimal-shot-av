# CoRL 2027 Audit Guide

This branch is intended to be auditable from the repository root without guessing
hidden setup steps.

## Scope

The branch contains three things that matter for audit:

1. the paper source: `docs/corl2027/paper.tex`
2. the reproducibility path for the AlpaSim transfer diagnostic
3. the published checkpoint manifest used to fetch learned-policy artifacts

Canonical branch for this audit surface:

```bash
git branch --show-current
```

Expected output:

```text
main
```

If you need subsystem entry points before reading the paper-specific audit, use:

```text
src/minimal_shot_av/simulator/README.md
src/minimal_shot_av/model/README.md
```

## Fast Audit

This path does not require launching AlpaSim:

```bash
./scripts/run_corl2027_audit.sh
```

This wrapper runs:

- `scripts/audit_corl_evidence_strength.py`
- `scripts/analyze_transfer_predictors.py`
- `scripts/analyze_alpasim_transfer_matrix.py` if a local transfer matrix already exists

Primary outputs:

- `artifacts/corl2027/corl_evidence_strength_audit.md`
- `artifacts/corl2027/transfer_predictor_analysis.md`
- `artifacts/corl2027/alpasim_transfer_matrix_partial.md` if matrix data is present

## Checkpoints

Published learned-model checkpoints live in the public Hugging Face repo:

```text
amtellezfernandez/minimal-shot-av-corl2027-checkpoints
```

The tracked manifest is:

```text
artifacts/models_manifest.json
```

Release packaging metadata is kept separate from the paper surface:

```text
artifacts/hf_release/
```

Fetch them with:

```bash
./.venv/bin/python scripts/fetch_checkpoints.py
```

## Full AlpaSim Reproduction

This requires an `x86_64` machine with Docker access and a real AlpaSim checkout.
The canonical local location is `workspace/alpasim/`.

```bash
export ALPASIM_ROOT=/abs/path/to/workspace/alpasim
./scripts/bootstrap_alpasim_runtime.sh
./scripts/check_alpasim_readiness.py --scene-preset front_camera_30scene_merged
./.venv/bin/python scripts/fetch_checkpoints.py
./.venv/bin/python scripts/run_alpasim_transfer_matrix.py \
  --mode both \
  --scene-presets front_camera_30scene_merged \
  --matrix-dir runs/alpasim_transfer_matrix_30scene_axis \
  --allow-existing-matrix-dir \
  --continue-on-error
```

The matrix currently runs these models in order:

1. `token_dagger_iter2`
2. `token_dagger_iter2_clamped`
3. `token_dagger_iter2_axis_constrained_clamped`
4. `token_dagger_iter2_hybrid_clamped`
5. `token_dagger_srcdecay`

Refresh the paired analysis after runs complete:

```bash
./.venv/bin/python scripts/analyze_alpasim_transfer_matrix.py \
  runs/alpasim_transfer_matrix_30scene_axis \
  --scene-preset front_camera_30scene_merged \
  --output-json artifacts/corl2027/alpasim_transfer_matrix_partial.json \
  --output-markdown artifacts/corl2027/alpasim_transfer_matrix_partial.md
```

## What To Read First

If you only have five minutes:

1. `README.md`
2. `src/minimal_shot_av/simulator/README.md`
3. `src/minimal_shot_av/model/README.md`
4. `docs/corl2027/paper.tex`
5. `artifacts/corl2027/corl_evidence_strength_audit.md`
6. `artifacts/corl2027/transfer_predictor_analysis.md`
7. `artifacts/corl2027/alpasim_transfer_matrix_partial.md` if present

## Known Boundaries

- Partial external matrices are progress indicators, not final claims.
- The AlpaSim analyzer now marks invalid `offroad`/`wrong_lane` scenes explicitly.
- Current evidence supports a strong diagnostic paper; the positive-method claim depends
  on the remaining external matrix rows.
