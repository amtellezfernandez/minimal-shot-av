# Getting Started

One page for a fresh clone: what you can run immediately, which environment to
build for which task, and where the gated data boundaries are.

## 1. Install (dataset-free core)

```bash
python3 -m venv .venv
./.venv/bin/pip install -e ".[alpasim,sim-viz]" pytest
```

Optional extras (see `pyproject.toml`): `nuplan` (nuPlan devkit stack),
`tuning` (Optuna sweeps), `audit-viz` (rerun-sdk viewers). Torch is only needed
for the learned-model tests/training paths: `pip install torch` (CPU build is fine).

## 2. Things that work with no datasets

```bash
# Quality gates + fast unit/smoke suite
./.venv/bin/python scripts/check_code_quality.py
./.venv/bin/python -m pytest tests -q

# Single Spotlight Reflex demo (internal debug simulator)
./.venv/bin/python scripts/run_demo.py \
  --policy spotlight-reflex --scenario-cluster spotlight --seed 3 \
  --artifacts-dir artifacts/demo_spotlight

# CoRL evidence audit over committed artifacts
./scripts/run_corl2027_audit.sh
```

## 3. Dataset-backed paths

| Path | Data | Obtainable? |
|---|---|---|
| nuPlan public mini | `scripts/bootstrap_nuplan_env.sh` + `scripts/fetch_nuplan_public_mini.py` | Yes — public download, fully scripted (see `docs/notes/nuplan-setup.md`) |
| WOD-E2E selector stack | Waymo Open Dataset E2E (gated) | Requires Waymo account; see `docs/notes/waymo-data-access.md` |
| AlpaSim transfer | Local AlpaSim checkout in `workspace/alpasim/` + Docker + NVIDIA runtime on x86_64 | Not redistributed; see `docs/corl2027/AUDIT.md` and `third_party/alpasim_overrides/README.md` |
| NAVSIM v2 / OneVL | NAVSIM data + GPU inference | See `LASTVLA/README.md` and `docs/notes/navsim-methodology.md` |

Published checkpoints: `scripts/fetch_checkpoints.py` pulls from the Hugging Face
release listed in `artifacts/models_manifest.json`.

## 4. Which virtualenv is which

Historically this repo used several venvs. The intended mapping:

| Env | Purpose | How to build |
|---|---|---|
| `.venv` | Default: core package, tests, demos, nuPlan mini | Section 1 above, or `scripts/bootstrap_nuplan_env.sh` |
| `.venv-nuplan` | Full nuPlan devkit stack (heavy deps) | `pip install -e ".[nuplan]"` into a fresh venv |
| `.venv-audit` | Test/audit env with torch for learned-model suites | Section 1 + `pip install torch` |
| `.venv-wod` (`.venv-v20`) | Legacy WOD/v20 GPU experiments | `scripts/bootstrap_v20_env.sh` (requires WSL GPU passthrough, see `docs/notes/wsl-gpu-setup.md`) |

If in doubt, build only `.venv` — everything in section 2 runs there.

## 5. Where things live

- `src/minimal_shot_av/` — the installable package (model / simulator / neutral / cli).
- `scripts/` — thin wrappers, one per CLI command (enforced by `tests/test_architecture_boundaries.py`).
- `tests/` — unit + smoke suite; run subsets with `scripts/run_tests.py --quick`.
- `artifacts/corl2027/` — curated, committed evidence surface (has its own README).
- `docs/corl2027/AUDIT.md` — the authoritative reproduction guide for paper claims.
- `docs/notes/` — historical working notes; **not** authoritative.
