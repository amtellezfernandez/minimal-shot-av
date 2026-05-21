#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
MATRIX_DIR="${MATRIX_DIR:-$ROOT/runs/alpasim_transfer_matrix_refresh_20260522_smoke10}"
SCENE_PRESET="${SCENE_PRESET:-front_camera_10scene_smoke}"
MODE="${MODE:-both}"
MODELS="${MODELS:-token_dagger_iter2,token_dagger_iter2_clamped,token_dagger_iter2_axis_constrained_clamped,token_dagger_iter2_hybrid_clamped,token_dagger_srcdecay}"
ALLOW_EXISTING=0
RERUN_EXISTING=0
CONTINUE_ON_ERROR=0
SCENE_LIMIT=""
SCENE_OFFSET=""
SKIP_MATRIX=0
SKIP_COMPARABLE=0
PUBLISHED_OURS_RUN=""
FRONT_OURS_RUN=""
FRONT_ALPAMAYO_RUN=""

usage() {
  cat <<'EOF'
Usage:
  scripts/refresh_corl2027_external_evidence.sh [options]

Options:
  --matrix-dir PATH
  --scene-preset NAME
  --models CSV
  --mode print|both
  --scene-limit N
  --scene-offset N
  --allow-existing
  --rerun-existing
  --continue-on-error
  --skip-matrix
  --published-ours-run PATH
  --front-ours-run PATH
  --front-alpamayo-run PATH
  --skip-comparable

Environment:
  HF_TOKEN must be set or cached for Hugging Face access.
  PYTHON_BIN can override the repo python used for orchestration.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --matrix-dir)
      MATRIX_DIR="$2"
      shift 2
      ;;
    --scene-preset)
      SCENE_PRESET="$2"
      shift 2
      ;;
    --models)
      MODELS="$2"
      shift 2
      ;;
    --mode)
      MODE="$2"
      shift 2
      ;;
    --scene-limit)
      SCENE_LIMIT="$2"
      shift 2
      ;;
    --scene-offset)
      SCENE_OFFSET="$2"
      shift 2
      ;;
    --allow-existing)
      ALLOW_EXISTING=1
      shift
      ;;
    --rerun-existing)
      RERUN_EXISTING=1
      shift
      ;;
    --continue-on-error)
      CONTINUE_ON_ERROR=1
      shift
      ;;
    --skip-matrix)
      SKIP_MATRIX=1
      shift
      ;;
    --published-ours-run)
      PUBLISHED_OURS_RUN="$2"
      shift 2
      ;;
    --front-ours-run)
      FRONT_OURS_RUN="$2"
      shift 2
      ;;
    --front-alpamayo-run)
      FRONT_ALPAMAYO_RUN="$2"
      shift 2
      ;;
    --skip-comparable)
      SKIP_COMPARABLE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing repo python: $PYTHON_BIN" >&2
  exit 1
fi

if [[ -f "$ROOT/.env.alpasim_hf" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.env.alpasim_hf"
fi

echo "[1/4] AlpaSim readiness"
"$PYTHON_BIN" "$ROOT/scripts/check_alpasim_readiness.py" --skip-scene-artifacts

if [[ "$SKIP_MATRIX" -eq 0 ]]; then
  echo
  echo "[2/4] Fresh AlpaSim transfer matrix"
  cmd=(
    "$ROOT/scripts/run_alpasim_transfer_matrix_repo.sh"
    --mode "$MODE"
    --scene-presets "$SCENE_PRESET"
    --models "$MODELS"
    --matrix-dir "$MATRIX_DIR"
  )
  if [[ "$ALLOW_EXISTING" -eq 1 ]]; then
    cmd+=(--allow-existing-matrix-dir)
  fi
  if [[ "$RERUN_EXISTING" -eq 1 ]]; then
    cmd+=(--rerun-existing)
  fi
  if [[ "$CONTINUE_ON_ERROR" -eq 1 ]]; then
    cmd+=(--continue-on-error)
  fi
  if [[ -n "$SCENE_LIMIT" ]]; then
    cmd+=(--scene-limit "$SCENE_LIMIT")
  fi
  if [[ -n "$SCENE_OFFSET" ]]; then
    cmd+=(--scene-offset "$SCENE_OFFSET")
  fi
  "${cmd[@]}"
fi

echo
echo "[3/4] Refresh matrix analysis and CoRL audit artifacts"
"$PYTHON_BIN" "$ROOT/scripts/analyze_alpasim_transfer_matrix.py" \
  "$MATRIX_DIR" \
  --baseline-model token_dagger_iter2 \
  --scene-preset "$SCENE_PRESET" \
  --output-json "$ROOT/artifacts/corl2027/alpasim_matrix10_analysis.json" \
  --output-markdown "$ROOT/artifacts/corl2027/alpasim_matrix10_analysis.md"

"$PYTHON_BIN" "$ROOT/scripts/audit_corl_evidence_strength.py"
"$PYTHON_BIN" "$ROOT/scripts/analyze_transfer_predictors.py"

if [[ "$SKIP_COMPARABLE" -eq 0 ]]; then
  if [[ -n "$PUBLISHED_OURS_RUN" && -n "$FRONT_OURS_RUN" && -n "$FRONT_ALPAMAYO_RUN" ]]; then
    echo
    echo "[4/4] Refresh comparable benchmark reports"
    "$PYTHON_BIN" "$ROOT/scripts/produce_alpasim_comparable_reports.py" \
      --published-ours-run "$PUBLISHED_OURS_RUN" \
      --front-ours-run "$FRONT_OURS_RUN" \
      --front-alpamayo-run "$FRONT_ALPAMAYO_RUN"
  else
    echo
    echo "[4/4] Comparable benchmark reports skipped (missing run paths)"
  fi
else
  echo
  echo "[4/4] Comparable benchmark reports skipped (--skip-comparable)"
fi

echo
echo "Primary refreshed outputs:"
echo "  - artifacts/corl2027/alpasim_matrix10_analysis.md"
echo "  - artifacts/corl2027/corl_evidence_strength_audit.md"
echo "  - artifacts/corl2027/transfer_predictor_analysis.md"
