#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${ROOT_DIR}"

source venv/onevl/bin/activate

REPO_PYTHON="${ROOT_DIR}/../../.venv/bin/python"
NAVSIM_PYTHONPATH="${HOME}/dev/navsim"
SEEDS="${SEEDS:-0 1 2}"
K_VALUES="${K_VALUES:-2 4 8 16 32}"

mkdir -p output/navsim/sweeps

for k in ${K_VALUES}; do
  sample_count=$((k - 1))
  for seed in ${SEEDS}; do
    sample_json="output/navsim/sweeps/holdout100_sample${sample_count}_seed${seed}_t08_p095_candidates.json"
    hybrid_json="output/navsim/sweeps/holdout100_hybrid_k${k}_seed${seed}_t08_p095_candidates.json"
    submission_dir="output/navsim/sweeps/holdout100_hybrid_k${k}_seed${seed}_t08_p095_submissions"
    score_dir="output/navsim/sweeps/holdout100_hybrid_k${k}_seed${seed}_t08_p095_scores"
    token_map="output/navsim/sweeps/holdout100_hybrid_k${k}_seed${seed}_t08_p095_token_map.json"
    oracle_json="output/navsim/sweeps/holdout100_hybrid_k${k}_seed${seed}_t08_p095_oracle.json"

    if [[ ! -f "${sample_json}" ]]; then
      python infer_onevl.py \
        --model_path models/OneVL_NAVSIM \
        --test_set_path test_data/navsim_test100_balanced_holdout.json \
        --image_base_path "" \
        --output_path "${sample_json}" \
        --device cuda:0 \
        --num_latent 2 \
        --num_latent_vis 4 \
        --max_new_tokens 1024 \
        --answer_prefix "[" \
        --prefix_k 0 \
        --candidate_mode sample \
        --num_candidates "${sample_count}" \
        --temperature 0.8 \
        --top_p 0.95 \
        --seed "${seed}"
    fi

    if [[ ! -f "${hybrid_json}" ]]; then
      python scripts/build_corrected_hybrid_bank.py \
        --greedy-json output/navsim/sweeps/holdout100_greedy_candidates.json \
        --sample-json "${sample_json}" \
        --output-json "${hybrid_json}" \
        --sample-count "${sample_count}"
    fi

    if [[ ! -f "${oracle_json}" ]]; then
      PYTHONPATH="${NAVSIM_PYTHONPATH}" "${REPO_PYTHON}" scripts/navsim_eval_pipeline.py \
        --subset-json test_data/navsim_test100_balanced_holdout.json \
        --candidate-json "${hybrid_json}" \
        --navsim-repo "${HOME}/dev/navsim" \
        --navsim-logs-dir "${HOME}/dev/navsim/download/test_navsim_logs" \
        --metric-cache-path "${HOME}/dev/navsim_workspace/exp/metric_cache_test100_balanced_holdout_frame1_packaged" \
        --submission-dir "${submission_dir}" \
        --score-dir "${score_dir}" \
        --token-map-out "${token_map}" \
        --oracle-summary-out "${oracle_json}" \
        --python-bin "${REPO_PYTHON}" \
        --skip-metric-cache
    fi
  done
done
