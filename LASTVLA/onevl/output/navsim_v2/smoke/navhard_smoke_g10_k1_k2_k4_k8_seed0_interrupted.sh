#!/usr/bin/env bash
set -euo pipefail
ROOT="$HOME/dev/minimal-shot-av-codex/LASTVLA/onevl"
cd "$ROOT"
REPO_PYTHON="$ROOT/../../.venv/bin/python"
NAVSIM_V2_REPO="$HOME/dev/navsim-v2"
OPENSCENE_DATA_ROOT="$NAVSIM_V2_REPO/data"
NAVSIM_EXP_ROOT="$HOME/dev/navsim_v2_workspace"
NUPLAN_MAPS_ROOT="$HOME/dev/minimal-shot-av/workspace/nuplan/maps/maps"
MODEL_PATH="models/OneVL_NAVSIM"
if [[ -f venv/onevl/bin/activate ]]; then
  source venv/onevl/bin/activate
fi
INFER_PYTHON="${INFER_PYTHON:-python}"
TORCH_SITE_PACKAGES="$ROOT/venv/onevl/lib/python3.12/site-packages"
MAX_GROUPS=10
SEED=0
TEMPERATURE=0.8
TOP_P=0.95
OUT_DIR="output/navsim_v2/smoke"
RUN_STEM="navhard_smoke_g${MAX_GROUPS}"
DATASET_JSON="test_data/navsim_v2_navhard_smoke_g${MAX_GROUPS}.json"
SCENE_FILTER_YAML="$NAVSIM_V2_REPO/navsim/planning/script/config/common/train_test_split/scene_filter/navhard_two_stage.yaml"
TRAIN_TEST_SPLIT_YAML="$NAVSIM_V2_REPO/navsim/planning/script/config/common/train_test_split/navhard_two_stage.yaml"
METRIC_CACHE="$NAVSIM_EXP_ROOT/metric_cache_navhard_two_stage_smoke_g${MAX_GROUPS}"
GREEDY_JSON="$OUT_DIR/${RUN_STEM}_greedy_seed${SEED}.json"
SAMPLE_JSON="$OUT_DIR/${RUN_STEM}_sample7_seed${SEED}_t${TEMPERATURE}_p${TOP_P}.json"
mkdir -p "$OUT_DIR" "$NAVSIM_EXP_ROOT"
echo "[RUN] build dataset g${MAX_GROUPS}"
PYTHONPATH="$NAVSIM_V2_REPO" "$REPO_PYTHON" scripts/build_navsim_v2_onevl_dataset.py \
  --scene-filter-yaml "$SCENE_FILTER_YAML" \
  --navsim-logs-dir "$OPENSCENE_DATA_ROOT/navsim_logs/test" \
  --original-sensor-root "$OPENSCENE_DATA_ROOT/sensor_blobs/test" \
  --synthetic-scenes-dir "$OPENSCENE_DATA_ROOT/navhard_two_stage/synthetic_scene_pickles" \
  --synthetic-sensor-root "$OPENSCENE_DATA_ROOT/navhard_two_stage/sensor_blobs" \
  --output-json "$DATASET_JSON" \
  --train-test-split-yaml "$TRAIN_TEST_SPLIT_YAML" \
  --max-groups "$MAX_GROUPS"
echo "[RUN] greedy g${MAX_GROUPS} seed=${SEED}"
"$INFER_PYTHON" infer_onevl.py \
  --model_path "$MODEL_PATH" \
  --test_set_path "$DATASET_JSON" \
  --image_base_path "" \
  --output_path "$GREEDY_JSON" \
  --device cuda:0 \
  --num_latent 2 \
  --num_latent_vis 4 \
  --max_new_tokens 1024 \
  --answer_prefix "[" \
  --prefix_k 0 \
  --candidate_mode greedy \
  --num_candidates 1 \
  --seed "$SEED"
echo "[RUN] sample7 g${MAX_GROUPS} seed=${SEED}"
"$INFER_PYTHON" infer_onevl.py \
  --model_path "$MODEL_PATH" \
  --test_set_path "$DATASET_JSON" \
  --image_base_path "" \
  --output_path "$SAMPLE_JSON" \
  --device cuda:0 \
  --num_latent 2 \
  --num_latent_vis 4 \
  --max_new_tokens 1024 \
  --answer_prefix "[" \
  --prefix_k 0 \
  --candidate_mode sample \
  --num_candidates 7 \
  --temperature "$TEMPERATURE" \
  --top_p "$TOP_P" \
  --seed "$SEED"
for K in 1 2 4 8; do
  HYBRID_JSON="$OUT_DIR/${RUN_STEM}_hybrid_k${K}_seed${SEED}_t${TEMPERATURE}_p${TOP_P}.json"
  SUBMISSION_DIR="$OUT_DIR/submissions_g${MAX_GROUPS}_k${K}_seed${SEED}"
  SCORE_DIR="$OUT_DIR/scores_g${MAX_GROUPS}_k${K}_seed${SEED}"
  ORACLE_JSON="$OUT_DIR/${RUN_STEM}_hybrid_k${K}_seed${SEED}_oracle.json"
  echo "[RUN] build/eval g${MAX_GROUPS} K=${K} seed=${SEED}"
  if [[ "$K" == "1" ]]; then
    cp "$GREEDY_JSON" "$HYBRID_JSON"
  else
    "$REPO_PYTHON" scripts/build_corrected_hybrid_bank.py \
      --greedy-json "$GREEDY_JSON" \
      --sample-json "$SAMPLE_JSON" \
      --sample-count "$((K - 1))" \
      --output-json "$HYBRID_JSON"
  fi
  extra=()
  if [[ "$K" != "1" ]]; then
    extra+=(--skip-metric-cache)
  fi
  PYTHONPATH="$NAVSIM_V2_REPO:$TORCH_SITE_PACKAGES" "$REPO_PYTHON" scripts/navsim_v2_eval_pipeline.py \
    --candidate-json "$HYBRID_JSON" \
    --navsim-repo "$NAVSIM_V2_REPO" \
    --openscene-data-root "$OPENSCENE_DATA_ROOT" \
    --navsim-exp-root "$NAVSIM_EXP_ROOT" \
    --nuplan-maps-root "$NUPLAN_MAPS_ROOT" \
    --metric-cache-path "$METRIC_CACHE" \
    --submission-dir "$SUBMISSION_DIR" \
    --score-dir "$SCORE_DIR" \
    --oracle-summary-out "$ORACLE_JSON" \
    --python-bin "$REPO_PYTHON" \
    --split navhard_two_stage \
    "${extra[@]}"
  echo "[DONE] g${MAX_GROUPS} K=${K} seed=${SEED}"
done
echo "[DONE] g${MAX_GROUPS} K=1,2,4,8 seed=${SEED}"
