#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="$HOME/.local/bin:$PATH"
export ALPASIM_ROOT="${ALPASIM_ROOT:-$ROOT/alpasim}"

if [[ -f "$ROOT/.env.alpasim_hf" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.env.alpasim_hf"
fi

"$ROOT/scripts/bootstrap_alpasim_runtime.sh"
"$ROOT/.venv/bin/python" "$ROOT/scripts/run_alpasim_transfer_matrix.py" "$@"
