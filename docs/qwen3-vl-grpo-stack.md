# Qwen3-VL WOD-E2E GRPO Stack

This stack is an optional offline research path around
`Qwen/Qwen3-VL-8B-Instruct`. It is not the strict zero-shot submission path and
it is not the online simulator policy. The online policy remains the fast
trajectory decoder plus RFS trust-region selector.

## Position

The idea is viable if the VLM is used for SFT/GRPO experiments and later
distillation. It is not viable as a real-time online planner because VLM
decoding is too slow compared with the optimized simulator loop.

For SoTA Commission framing, keep this clearly separated from the strict
minimal-shot claim. Any use of WOD-E2E labels for SFT/GRPO must be declared as
preference-calibrated training, not zero-shot operation.

## Pipeline

1. Build WOD-E2E preference frames with the official parser.
2. Build Qwen3-VL SFT records:

   ```bash
   PYTHONPATH=.wod-protos .venv-wod/bin/python scripts/build_qwen3_vl_sft_dataset.py \
     --val-dir waymo_open_dataset_end_to_end_camera_v_1_0_0/val \
     --output artifacts/qwen3_vl_wod_sft.jsonl
   ```

3. Run reward dry-run:

   ```bash
   .venv-qwen/bin/python scripts/train_qwen3_vl_grpo.py \
     --dataset artifacts/qwen3_vl_wod_sft.jsonl \
     --dry-run
   ```

4. On the training cluster, install optional dependencies:

   ```bash
   scripts/setup_qwen3_vl_stack.sh
   ```

   This creates `.venv-qwen` and clones helper fine-tuning code under
   `external/qwen2_5_vl_finetune`.

5. Use TRL GRPO with rewards from `scripts/train_qwen3_vl_grpo.py`. The current
   non-dry-run path validates the training dependencies and writes a launch plan;
   the full GPU job should run on the cluster with local model weights.

## Prompt Policy

The prompt emits compact causal fields and does not ask the model to emit
hidden chain-of-thought. In leaderboard-facing materials, describe this as
structured scene/hazard output rather than branded causal-reasoning traces:

- critical objects
- interaction summary
- intent
- maneuver
- `trajectory_64wp_10hz`

The trajectory is resampled to WOD format with
`minimal_shot_av.trajectory_resampling.resample_64wp_10hz_to_20wp_4hz`.

## Runtime Policy

Do not run Qwen3-VL-8B in the simulator loop. Distill its outputs into a compact
trajectory decoder and keep:

- `RFS trust-region selector`
- `WOD preference ranker`
- forecast-aware clearance checks
- safety filter
