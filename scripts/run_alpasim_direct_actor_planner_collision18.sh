#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT/.uv-cache}"

PRESET="${PRESET:-front_camera_collision18}"
BASELINE_MODEL="${BASELINE_MODEL:-token_dagger_iter2_axis_constrained_clamped}"
DIRECT_MODEL="${DIRECT_MODEL:-direct_actor_planner_oracle}"

ALPASIM_ROOT="${ALPASIM_ROOT:-$ROOT/alpasim}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
ALPASIM_PYTHON="${ALPASIM_PYTHON:-$ALPASIM_ROOT/.venv/bin/python}"

BASELINE_30_DIR="${BASELINE_30_DIR:-$ROOT/runs/alpasim_axis_constrained_30scene_eval}"
MATRIX_DIR="${MATRIX_DIR:-$ROOT/runs/alpasim_direct_actor_planner_collision18}"
BASELINE_BATCH_DIR="$MATRIX_DIR/${BASELINE_MODEL}__${PRESET}"
DIRECT_BATCH_DIR="$MATRIX_DIR/${DIRECT_MODEL}__${PRESET}"

ORACLE_PROXY_JSON="${ORACLE_PROXY_JSON:-$ROOT/artifacts/alpasim_oracle_actor_proxy_30scene.json}"
ANALYSIS_JSON="${ANALYSIS_JSON:-$ROOT/artifacts/alpasim_direct_actor_planner_collision18_analysis.json}"
ANALYSIS_MD="${ANALYSIS_MD:-$ROOT/artifacts/alpasim_direct_actor_planner_collision18_analysis.md}"

TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-900}"
TOPOLOGY="${TOPOLOGY:-1gpu}"
MAX_BATCH_PASSES="${MAX_BATCH_PASSES:-4}"
POLL_SECONDS="${POLL_SECONDS:-60}"

log() {
  printf '[direct-actor-planner] %s\n' "$*"
}

ensure_hf_token() {
  if [[ -n "${HF_TOKEN:-}" ]]; then
    export HF_TOKEN
    return
  fi
  local token
  token="$("$ALPASIM_PYTHON" - <<'PY'
from huggingface_hub import get_token
print(get_token() or "")
PY
)"
  token="${token//$'\r'/}"
  token="${token//$'\n'/}"
  if [[ -z "$token" ]]; then
    log "No cached Hugging Face token available; cannot restore gated AlpaSim scene artifacts."
    exit 1
  fi
  export HF_TOKEN="$token"
}

expected_scene_count() {
  "$PYTHON_BIN" - <<'PY' "$ROOT" "$PRESET"
from pathlib import Path
import sys
import yaml

root = Path(sys.argv[1])
preset = sys.argv[2]
path = root / "src" / "minimal_shot_av" / "simulator" / "alpasim_scene_presets" / f"{preset}.yaml"
payload = yaml.safe_load(path.read_text(encoding="utf-8"))
print(len(payload.get("scenes", {}).get("scene_ids", [])))
PY
}

count_completed_runs() {
  "$PYTHON_BIN" - <<'PY' "$1"
from pathlib import Path
import sys

batch_dir = Path(sys.argv[1])
if not batch_dir.exists():
    print(0)
    raise SystemExit
count = 0
for run_dir in sorted(path for path in batch_dir.iterdir() if path.is_dir()):
    aggregate = run_dir / "aggregate"
    if any(
        candidate.is_file()
        for candidate in (
            aggregate / "metrics_unprocessed.parquet",
            aggregate / "metrics_results.parquet",
            aggregate / "metrics_results.txt",
        )
    ):
        count += 1
print(count)
PY
}

wait_for_active_batch() {
  local batch_dir="$1"
  local batch_base
  batch_base="$(basename "$batch_dir")"
  while pgrep -af "run_alpasim_scene_batch.py" | grep -F -- "$batch_base" >/dev/null 2>&1; do
    log "Waiting for active batch process on ${batch_dir}"
    sleep "$POLL_SECONDS"
  done
}

complete_batch() {
  local batch_dir="$1"
  local model="$2"
  local expected
  expected="$(expected_scene_count)"

  mkdir -p "$batch_dir"
  for pass in $(seq 1 "$MAX_BATCH_PASSES"); do
    wait_for_active_batch "$batch_dir"
    local completed
    completed="$(count_completed_runs "$batch_dir")"
    log "Batch ${model} pass ${pass}: ${completed}/${expected} scenes completed"
    if [[ "$completed" == "$expected" ]]; then
      return 0
    fi

    HF_TOKEN="$HF_TOKEN" "$PYTHON_BIN" "$ROOT/scripts/run_alpasim_scene_batch.py" \
      --mode both \
      --model "$model" \
      --scene-preset "$PRESET" \
      --batch-dir "$batch_dir" \
      --allow-existing-batch-dir \
      --continue-on-error \
      --alpasim-root "$ALPASIM_ROOT" \
      --timeout "$TIMEOUT_SECONDS" \
      --topology "$TOPOLOGY" \
      --oracle-actor-proxy "$ORACLE_PROXY_JSON"
  done

  local completed
  completed="$(count_completed_runs "$batch_dir")"
  log "Batch ${model} incomplete after ${MAX_BATCH_PASSES} passes: ${completed}/${expected}"
  exit 1
}

link_collision18_baseline() {
  mkdir -p "$MATRIX_DIR"
  if [[ -L "$BASELINE_BATCH_DIR" || -d "$BASELINE_BATCH_DIR" ]]; then
    return
  fi
  "$PYTHON_BIN" - <<'PY' "$ROOT" "$BASELINE_30_DIR" "$BASELINE_BATCH_DIR"
from pathlib import Path
import shutil
import sys
import yaml

root = Path(sys.argv[1])
source = Path(sys.argv[2])
target = Path(sys.argv[3])
preset = yaml.safe_load(
    (root / "src" / "minimal_shot_av" / "simulator" / "alpasim_scene_presets" / "front_camera_collision18.yaml")
    .read_text(encoding="utf-8")
)
scene_ids = preset["scenes"]["scene_ids"]
target.mkdir(parents=True, exist_ok=True)
for index, scene_id in enumerate(scene_ids, start=1):
    matches = sorted(source.glob(f"*_{{scene_id}}".format(scene_id=scene_id)))
    if not matches:
        raise SystemExit(f"Missing baseline source scene: {scene_id}")
    link = target / f"{index:03d}_{scene_id}"
    if link.exists() or link.is_symlink():
        continue
    link.symlink_to(matches[0].resolve(), target_is_directory=True)
PY
}

run_analysis() {
  log "Running paired raw analysis"
  "$PYTHON_BIN" "$ROOT/scripts/analyze_alpasim_transfer_matrix.py" \
    "$MATRIX_DIR" \
    --baseline-model "$BASELINE_MODEL" \
    --scene-preset "$PRESET" \
    --output-json "$ANALYSIS_JSON" \
    --output-markdown "$ANALYSIS_MD"
}

main() {
  if [[ ! -f "$ORACLE_PROXY_JSON" ]]; then
    log "Missing oracle actor proxy: $ORACLE_PROXY_JSON"
    exit 1
  fi
  ensure_hf_token
  link_collision18_baseline
  complete_batch "$DIRECT_BATCH_DIR" "$DIRECT_MODEL"
  run_analysis
  log "Done"
}

main "$@"
