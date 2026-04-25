#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required because this environment does not provide pip." >&2
  exit 1
fi

uv venv .venv-qwen
uv pip install --python .venv-qwen/bin/python -e '.[qwen-grpo]'

mkdir -p external
if [ ! -d external/qwen2_5_vl_finetune ]; then
  git clone https://github.com/2U1/Qwen2-VL-Finetune.git external/qwen2_5_vl_finetune
fi

cat > external/README.md <<'EOF'
# External Training Repos

These repos are optional training helpers. The runtime simulator path does not
depend on them.

- `qwen2_5_vl_finetune`: reference VLM fine-tuning scripts that can be adapted
  to Qwen3-VL once the local CUDA/Transformers stack supports the selected
  model checkpoint.

Model weights are not downloaded by this setup script. Use Hugging Face login
and local cluster storage for `Qwen/Qwen3-VL-8B-Instruct`.
EOF

echo "Qwen3-VL optional training stack installed in .venv-qwen."
