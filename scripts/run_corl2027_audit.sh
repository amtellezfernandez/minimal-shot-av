#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  echo "Missing repo venv: $ROOT/.venv/bin/python" >&2
  exit 1
fi

echo "[1/3] Evidence strength audit"
"$ROOT/.venv/bin/python" "$ROOT/scripts/audit_corl_evidence_strength.py"

echo
echo "[2/3] Transfer predictor analysis"
"$ROOT/.venv/bin/python" "$ROOT/scripts/analyze_transfer_predictors.py"

MATRIX_DIR="$ROOT/runs/alpasim_transfer_matrix_30scene_axis"
if [[ -d "$MATRIX_DIR" ]]; then
  echo
  echo "[3/3] Refresh paired AlpaSim transfer analysis"
  "$ROOT/.venv/bin/python" "$ROOT/scripts/analyze_alpasim_transfer_matrix.py" \
    "$MATRIX_DIR" \
    --scene-preset front_camera_30scene_merged \
    --output-json "$ROOT/artifacts/alpasim_transfer_matrix_partial.json" \
    --output-markdown "$ROOT/artifacts/alpasim_transfer_matrix_partial.md"
else
  echo
  echo "[3/3] Skipped paired AlpaSim transfer analysis (no matrix dir at $MATRIX_DIR)"
fi

echo
echo "Primary audit outputs:"
echo "  - artifacts/corl_evidence_strength_audit.md"
echo "  - artifacts/transfer_predictor_analysis.md"
echo "  - artifacts/alpasim_transfer_matrix_partial.md (if matrix data exists)"
