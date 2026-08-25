# Artifacts

Large data/checkpoints were moved off the workstation on 2026-08-25 to private Hugging Face datasets:

- `artifacts/` → https://huggingface.co/datasets/amtellezfernandez/minimal-shot-av-artifacts
- `runs/` → https://huggingface.co/datasets/amtellezfernandez/minimal-shot-av-runs
- `workspace/` → https://huggingface.co/datasets/amtellezfernandez/minimal-shot-av-workspace

Restore with: `hf download amtellezfernandez/<name> --repo-type dataset --local-dir <dir>`
