#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${ROOT_DIR}"

source venv/onevl/bin/activate

mkdir -p output/navsim/sweeps

train_candidate_args=()
train_token_args=()
train_score_args=()
for seed in 0 1 2 3; do
  train_candidate_args+=("output/navsim/sweeps/train200_hybrid_k8_seed${seed}_t08_p095_candidates.json")
  train_token_args+=("output/navsim/sweeps/train200_hybrid_k8_seed${seed}_t08_p095_token_map.json")
  train_score_args+=("output/navsim/sweeps/train200_hybrid_k8_seed${seed}_t08_p095_scores")
done

for seed in 0 1 2 3; do
  summary_json="output/navsim/sweeps/train200_pooled_k8_to_holdout100_seed${seed}_transfer.json"
  model_json="output/navsim/sweeps/train200_pooled_k8_to_holdout100_seed${seed}_ridge_model.json"
  python scripts/transfer_navsim_ridge_reranker.py \
    --train-candidate-json "${train_candidate_args[@]}" \
    --train-token-map "${train_token_args[@]}" \
    --train-score-dir "${train_score_args[@]}" \
    --test-candidate-json "output/navsim/sweeps/holdout100_hybrid_k8_seed${seed}_t08_p095_candidates.json" \
    --test-token-map "output/navsim/test100_balanced_holdout_token_map.json" \
    --test-score-dir "output/navsim/sweeps/holdout100_hybrid_k8_seed${seed}_t08_p095_scores" \
    --summary-out "${summary_json}" \
    --model-out "${model_json}"

  no_order_summary_json="output/navsim/sweeps/train200_pooled_k8_to_holdout100_seed${seed}_zero_perception_transfer.json"
  no_order_model_json="output/navsim/sweeps/train200_pooled_k8_to_holdout100_seed${seed}_zero_perception_ridge_model.json"
  python scripts/transfer_navsim_ridge_reranker.py \
    --train-candidate-json "${train_candidate_args[@]}" \
    --train-token-map "${train_token_args[@]}" \
    --train-score-dir "${train_score_args[@]}" \
    --test-candidate-json "output/navsim/sweeps/holdout100_hybrid_k8_seed${seed}_t08_p095_candidates.json" \
    --test-token-map "output/navsim/test100_balanced_holdout_token_map.json" \
    --test-score-dir "output/navsim/sweeps/holdout100_hybrid_k8_seed${seed}_t08_p095_scores" \
    --summary-out "${no_order_summary_json}" \
    --model-out "${no_order_model_json}" \
    --drop-order-features
done
