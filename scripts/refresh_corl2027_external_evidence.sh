#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${PYTHON_BIN:-$ROOT/.venv/bin/python}" "$ROOT/scripts/refresh_corl2027_external_evidence.py" "$@"
