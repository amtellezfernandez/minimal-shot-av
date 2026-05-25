#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UV_BIN="${UV_BIN:-$(command -v uv || true)}"
VENV_DIR="${VENV_DIR:-$ROOT/.venv}"

if [[ -z "$UV_BIN" ]]; then
  echo "uv is required. Install it first, e.g. python3 -m pip install --user uv" >&2
  exit 1
fi

mkdir -p "$ROOT/.uv-cache"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  env UV_CACHE_DIR="$ROOT/.uv-cache" "$UV_BIN" venv "$VENV_DIR"
fi

env UV_CACHE_DIR="$ROOT/.uv-cache" "$UV_BIN" pip install \
  --python "$VENV_DIR/bin/python" \
  -e "$ROOT[nuplan]"

"$VENV_DIR/bin/python" - <<'PY'
from nuplan.database.nuplan_db.nuplan_db_utils import get_lidarpc_sensor_data
from nuplan.database.nuplan_db.nuplan_scenario_queries import get_scenarios_from_db

print("nuplan imports ok")
print("sensor source:", get_lidarpc_sensor_data())
print("scenario query:", get_scenarios_from_db.__name__)
PY

cat <<EOF
Repo nuPlan environment is ready at $VENV_DIR

Next steps:
  $VENV_DIR/bin/python scripts/fetch_nuplan_public_mini.py --db-count 5
  PYTHONPATH=src $VENV_DIR/bin/python scripts/run_nuplan_maneuvertoken_rollout.py \\
    --nuplan-db-file $ROOT/workspace/nuplan/public_mini \\
    --limit 50
  PYTHONPATH=src $VENV_DIR/bin/python scripts/train_nuplan_maneuvertoken_selector.py \\
    --nuplan-db-file $ROOT/workspace/nuplan/public_mini \\
    --limit 200
EOF
