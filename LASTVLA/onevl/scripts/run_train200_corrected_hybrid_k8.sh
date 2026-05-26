#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${ROOT_DIR}"

source venv/onevl/bin/activate
REPO_PYTHON="${ROOT_DIR}/../../.venv/bin/python"
NAVSIM_PYTHONPATH="${HOME}/dev/navsim"

mkdir -p output/navsim/sweeps

for seed in 0 1 2 3; do
  sample_json="output/navsim/sweeps/train200_sample7_seed${seed}_t08_p095_candidates.json"
  hybrid_json="output/navsim/sweeps/train200_hybrid_k8_seed${seed}_t08_p095_candidates.json"
  submission_dir="output/navsim/sweeps/train200_hybrid_k8_seed${seed}_t08_p095_submissions"
  score_dir="output/navsim/sweeps/train200_hybrid_k8_seed${seed}_t08_p095_scores"
  token_map="output/navsim/sweeps/train200_hybrid_k8_seed${seed}_t08_p095_token_map.json"
  oracle_json="output/navsim/sweeps/train200_hybrid_k8_seed${seed}_t08_p095_oracle.json"

  if [[ ! -f "${sample_json}" ]]; then
    python infer_onevl.py \
      --model_path models/OneVL_NAVSIM \
      --test_set_path test_data/navsim_test200_balanced.json \
      --image_base_path "" \
      --output_path "${sample_json}" \
      --device cuda:0 \
      --num_latent 2 \
      --num_latent_vis 4 \
      --max_new_tokens 1024 \
      --answer_prefix "[" \
      --prefix_k 0 \
      --candidate_mode sample \
      --num_candidates 7 \
      --temperature 0.8 \
      --top_p 0.95 \
      --seed "${seed}"
  fi

  if [[ ! -f "${hybrid_json}" ]]; then
    python scripts/build_corrected_hybrid_bank.py \
      --greedy-json output/navsim/sweeps/train200_greedy_candidates.json \
      --sample-json "${sample_json}" \
      --output-json "${hybrid_json}" \
      --sample-count 7
  fi

  if [[ ! -f "${oracle_json}" ]]; then
    PYTHONPATH="${NAVSIM_PYTHONPATH}" "${REPO_PYTHON}" scripts/navsim_eval_pipeline.py \
      --subset-json test_data/navsim_test200_balanced.json \
      --candidate-json "${hybrid_json}" \
      --navsim-repo "${HOME}/dev/navsim" \
      --navsim-logs-dir "${HOME}/dev/navsim/download/test_navsim_logs" \
      --metric-cache-path "${HOME}/dev/navsim_workspace/exp/metric_cache_test200_balanced_frame1" \
      --submission-dir "${submission_dir}" \
      --score-dir "${score_dir}" \
      --token-map-out "${token_map}" \
      --oracle-summary-out "${oracle_json}" \
      --python-bin "${REPO_PYTHON}" \
      --skip-metric-cache
  fi
done
