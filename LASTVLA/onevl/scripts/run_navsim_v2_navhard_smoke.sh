#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_PYTHON="${REPO_PYTHON:-${ROOT_DIR}/../../.venv/bin/python}"
NAVSIM_V2_REPO="${NAVSIM_V2_REPO:-${HOME}/dev/navsim-v2}"
OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-${NAVSIM_V2_REPO}/data}"
NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-${HOME}/dev/navsim_v2_workspace}"
MODEL_PATH="${MODEL_PATH:-models/OneVL_NAVSIM}"

MAX_STAGE_ONE="${MAX_STAGE_ONE:-2}"
MAX_STAGE_TWO="${MAX_STAGE_TWO:-2}"
K="${K:-4}"
SEED="${SEED:-0}"
TEMPERATURE="${TEMPERATURE:-0.8}"
TOP_P="${TOP_P:-0.95}"

SCENE_FILTER_YAML="${NAVSIM_V2_REPO}/navsim/planning/script/config/common/train_test_split/scene_filter/navhard_two_stage.yaml"
DATASET_JSON="test_data/navsim_v2_navhard_smoke_s1_${MAX_STAGE_ONE}_s2_${MAX_STAGE_TWO}.json"
OUT_DIR="output/navsim_v2/smoke"
GREEDY_JSON="${OUT_DIR}/navhard_smoke_greedy_seed${SEED}.json"
SAMPLE_JSON="${OUT_DIR}/navhard_smoke_sample$((K - 1))_seed${SEED}_t${TEMPERATURE}_p${TOP_P}.json"
HYBRID_JSON="${OUT_DIR}/navhard_smoke_hybrid_k${K}_seed${SEED}_t${TEMPERATURE}_p${TOP_P}.json"
METRIC_CACHE="${NAVSIM_EXP_ROOT}/metric_cache_navhard_two_stage_smoke_s1_${MAX_STAGE_ONE}_s2_${MAX_STAGE_TWO}"
SUBMISSION_DIR="${OUT_DIR}/submissions_k${K}_seed${SEED}"
SCORE_DIR="${OUT_DIR}/scores_k${K}_seed${SEED}"
ORACLE_JSON="${OUT_DIR}/navhard_smoke_hybrid_k${K}_seed${SEED}_oracle.json"

if [[ "${K}" -lt 1 ]]; then
  echo "K must be >= 1" >&2
  exit 2
fi

mkdir -p "${OUT_DIR}" "${NAVSIM_EXP_ROOT}"

"${REPO_PYTHON}" scripts/build_navsim_v2_onevl_dataset.py \
  --scene-filter-yaml "${SCENE_FILTER_YAML}" \
  --navsim-logs-dir "${OPENSCENE_DATA_ROOT}/navsim_logs/test" \
  --original-sensor-root "${OPENSCENE_DATA_ROOT}/sensor_blobs/test" \
  --synthetic-scenes-dir "${OPENSCENE_DATA_ROOT}/navhard_two_stage/synthetic_scene_pickles" \
  --synthetic-sensor-root "${OPENSCENE_DATA_ROOT}/navhard_two_stage/sensor_blobs" \
  --output-json "${DATASET_JSON}" \
  --max-stage-one "${MAX_STAGE_ONE}" \
  --max-stage-two "${MAX_STAGE_TWO}"

"${REPO_PYTHON}" infer_onevl.py \
  --model_path "${MODEL_PATH}" \
  --test_set_path "${DATASET_JSON}" \
  --image_base_path "" \
  --output_path "${GREEDY_JSON}" \
  --device cuda:0 \
  --num_latent 2 \
  --num_latent_vis 4 \
  --max_new_tokens 1024 \
  --answer_prefix "[" \
  --prefix_k 0 \
  --candidate_mode greedy \
  --num_candidates 1 \
  --seed "${SEED}"

if [[ "${K}" -gt 1 ]]; then
  "${REPO_PYTHON}" infer_onevl.py \
    --model_path "${MODEL_PATH}" \
    --test_set_path "${DATASET_JSON}" \
    --image_base_path "" \
    --output_path "${SAMPLE_JSON}" \
    --device cuda:0 \
    --num_latent 2 \
    --num_latent_vis 4 \
    --max_new_tokens 1024 \
    --answer_prefix "[" \
    --prefix_k 0 \
    --candidate_mode sample \
    --num_candidates "$((K - 1))" \
    --temperature "${TEMPERATURE}" \
    --top_p "${TOP_P}" \
    --seed "${SEED}"

  "${REPO_PYTHON}" scripts/build_corrected_hybrid_bank.py \
    --greedy-json "${GREEDY_JSON}" \
    --sample-json "${SAMPLE_JSON}" \
    --sample-count "$((K - 1))" \
    --output-json "${HYBRID_JSON}"
else
  cp "${GREEDY_JSON}" "${HYBRID_JSON}"
fi

PYTHONPATH="${NAVSIM_V2_REPO}" "${REPO_PYTHON}" scripts/navsim_v2_eval_pipeline.py \
  --candidate-json "${HYBRID_JSON}" \
  --navsim-repo "${NAVSIM_V2_REPO}" \
  --openscene-data-root "${OPENSCENE_DATA_ROOT}" \
  --navsim-exp-root "${NAVSIM_EXP_ROOT}" \
  --metric-cache-path "${METRIC_CACHE}" \
  --submission-dir "${SUBMISSION_DIR}" \
  --score-dir "${SCORE_DIR}" \
  --oracle-summary-out "${ORACLE_JSON}" \
  --python-bin "${REPO_PYTHON}" \
  --split navhard_two_stage
