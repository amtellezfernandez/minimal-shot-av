#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${ROOT_DIR}"

POLL_SECONDS="${POLL_SECONDS:-60}"
K_VALUES="${K_VALUES:-16 32}"
SEEDS="${SEEDS:-0 1 2}"

while true; do
  missing=0
  for seed in 0 1 2 3; do
    oracle_json="output/navsim/sweeps/train200_hybrid_k8_seed${seed}_t08_p095_oracle.json"
    if [[ ! -f "${oracle_json}" ]]; then
      missing=1
      break
    fi
  done
  if [[ "${missing}" -eq 0 ]]; then
    break
  fi
  sleep "${POLL_SECONDS}"
done

K_VALUES="${K_VALUES}" SEEDS="${SEEDS}" exec ./scripts/run_holdout100_corrected_hybrid_k_sweep.sh
