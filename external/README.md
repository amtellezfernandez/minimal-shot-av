# External Training Repos

These repos are optional training helpers. The runtime simulator path does not
depend on them.

- `qwen2_5_vl_finetune`: reference VLM fine-tuning scripts that can be adapted
  to Qwen3-VL once the local CUDA/Transformers stack supports the selected
  model checkpoint.

Model weights are not downloaded by this setup script. Use Hugging Face login
and local cluster storage for `Qwen/Qwen3-VL-8B-Instruct`.
