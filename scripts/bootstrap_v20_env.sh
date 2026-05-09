#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/.venv-v20/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  if [[ -x "$ROOT/.venv-wod/bin/python" ]]; then
    ln -s .venv-wod "$ROOT/.venv-v20"
  else
    uv venv "$ROOT/.venv-v20"
  fi
fi

env UV_CACHE_DIR="$ROOT/.uv-cache" uv pip install \
  --python "$ROOT/.venv-v20/bin/python" \
  -e "$ROOT" \
  "transformers==4.51.0" \
  "diffusers==0.33.1" \
  "accelerate==1.4.0" \
  "pytest==9.0.3"

"$ROOT/.venv-v20/bin/python" "$ROOT/scripts/check_v20_env.py"
"$ROOT/.venv-v20/bin/python" "$ROOT/scripts/check_cuda_preflight.py" \
  --output "$ROOT/artifacts/cuda_preflight.json"
