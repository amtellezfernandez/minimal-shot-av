#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT/.uv-cache}"

PRESET="${PRESET:-front_camera_30scene_merged}"
BASELINE_MODEL="${BASELINE_MODEL:-token_dagger_iter2_axis_constrained_clamped}"
ORACLE_MODEL="${ORACLE_MODEL:-token_dagger_iter2_axis_constrained_oracle_actor_clamped}"

ALPASIM_ROOT="${ALPASIM_ROOT:-$ROOT/workspace/alpasim}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
ALPASIM_PYTHON="${ALPASIM_PYTHON:-$ALPASIM_ROOT/.venv/bin/python}"

BASELINE_STAGING_DIR="${BASELINE_STAGING_DIR:-$ROOT/runs/alpasim_axis_constrained_30scene_eval}"
MATRIX_DIR="${MATRIX_DIR:-$ROOT/runs/alpasim_actor_blindness_30scene_raw}"
BASELINE_MATRIX_DIR="$MATRIX_DIR/${BASELINE_MODEL}__${PRESET}"
ORACLE_BATCH_DIR="${ORACLE_BATCH_DIR:-$MATRIX_DIR/${ORACLE_MODEL}__${PRESET}}"
CORL_ARTIFACTS_DIR="${CORL_ARTIFACTS_DIR:-$ROOT/artifacts/corl2027}"

ORACLE_PROXY_JSON="${ORACLE_PROXY_JSON:-$ROOT/artifacts/alpasim_oracle_actor_proxy_30scene.json}"
ANALYSIS_JSON="${ANALYSIS_JSON:-$CORL_ARTIFACTS_DIR/alpasim_actor_blindness_30scene_raw_analysis.json}"
ANALYSIS_MD="${ANALYSIS_MD:-$CORL_ARTIFACTS_DIR/alpasim_actor_blindness_30scene_raw_analysis.md}"
COLLISION_AUDIT_JSON="${COLLISION_AUDIT_JSON:-$CORL_ARTIFACTS_DIR/alpasim_collision_surface_30scene_audit.json}"
COLLISION_AUDIT_MD="${COLLISION_AUDIT_MD:-$CORL_ARTIFACTS_DIR/alpasim_collision_surface_30scene_audit.md}"
COUNTERFACTUAL_AUDIT_JSON="${COUNTERFACTUAL_AUDIT_JSON:-$CORL_ARTIFACTS_DIR/alpasim_candidate_counterfactual_30scene.json}"
COUNTERFACTUAL_AUDIT_MD="${COUNTERFACTUAL_AUDIT_MD:-$CORL_ARTIFACTS_DIR/alpasim_candidate_counterfactual_30scene.md}"

TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-900}"
TOPOLOGY="${TOPOLOGY:-1gpu}"
MAX_BATCH_PASSES="${MAX_BATCH_PASSES:-6}"
POLL_SECONDS="${POLL_SECONDS:-60}"

log() {
  printf '[actor-blindness] %s\n' "$*"
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
import yaml
import sys

root = Path(sys.argv[1])
preset = sys.argv[2]
path = root / "src" / "minimal_shot_av" / "simulator" / "alpasim_scene_presets" / f"{preset}.yaml"
payload = yaml.safe_load(path.read_text(encoding="utf-8"))
scene_ids = payload.get("scenes", {}).get("scene_ids", [])
print(len(scene_ids))
PY
}

count_completed_runs() {
  "$PYTHON_BIN" - <<'PY' "$1"
from pathlib import Path
import sys

batch_dir = Path(sys.argv[1])
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
  local oracle_proxy="${3:-}"
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

    local cmd=(
      "$PYTHON_BIN"
      "$ROOT/scripts/run_alpasim_scene_batch.py"
      --mode both
      --model "$model"
      --scene-preset "$PRESET"
      --batch-dir "$batch_dir"
      --allow-existing-batch-dir
      --continue-on-error
      --alpasim-root "$ALPASIM_ROOT"
      --timeout "$TIMEOUT_SECONDS"
      --topology "$TOPOLOGY"
    )
    if [[ -n "$oracle_proxy" ]]; then
      cmd+=(--oracle-actor-proxy "$oracle_proxy")
    fi

    log "Launching completion pass ${pass} for ${model}"
    HF_TOKEN="$HF_TOKEN" "${cmd[@]}"
  done

  local completed
  completed="$(count_completed_runs "$batch_dir")"
  log "Batch ${model} incomplete after ${MAX_BATCH_PASSES} passes: ${completed}/${expected}"
  exit 1
}

link_baseline_into_matrix() {
  mkdir -p "$MATRIX_DIR"
  if [[ -L "$BASELINE_MATRIX_DIR" || -d "$BASELINE_MATRIX_DIR" ]]; then
    return
  fi
  ln -s "$BASELINE_STAGING_DIR" "$BASELINE_MATRIX_DIR"
}

build_oracle_proxy() {
  log "Building 30-scene world-frame oracle proxy from ${BASELINE_STAGING_DIR}"
  "$ALPASIM_PYTHON" "$ROOT/scripts/build_alpasim_oracle_actor_proxy.py" \
    --run-dir "$BASELINE_STAGING_DIR" \
    --output "$ORACLE_PROXY_JSON" \
    --alpasim-root "$ALPASIM_ROOT"
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

run_collision_surface_audit() {
  log "Running first-impact collision-surface audit"
  "$PYTHON_BIN" "$ROOT/scripts/audit_alpasim_collision_surface.py" \
    --baseline-run-dir "$BASELINE_STAGING_DIR" \
    --candidate-run-dir "$ORACLE_BATCH_DIR" \
    --output-json "$COLLISION_AUDIT_JSON" \
    --output-markdown "$COLLISION_AUDIT_MD"
}

run_candidate_counterfactual_audit() {
  log "Running selector-side candidate-counterfactual audit"
  "$PYTHON_BIN" "$ROOT/scripts/audit_alpasim_candidate_counterfactuals.py" \
    --baseline-run-dir "$BASELINE_STAGING_DIR" \
    --candidate-run-dir "$ORACLE_BATCH_DIR" \
    --oracle-actor-proxy "$ORACLE_PROXY_JSON" \
    --output-json "$COUNTERFACTUAL_AUDIT_JSON" \
    --output-markdown "$COUNTERFACTUAL_AUDIT_MD"
}

main() {
  ensure_hf_token
  link_baseline_into_matrix
  complete_batch "$BASELINE_STAGING_DIR" "$BASELINE_MODEL"
  build_oracle_proxy
  complete_batch "$ORACLE_BATCH_DIR" "$ORACLE_MODEL" "$ORACLE_PROXY_JSON"
  run_analysis
  run_collision_surface_audit
  run_candidate_counterfactual_audit
  log "Done"
}

main "$@"
