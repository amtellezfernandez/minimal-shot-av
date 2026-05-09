# WSL GPU Setup

This repo uses `.venv-v20` for WOD/v20 training. CUDA is ready only when all of these are true inside Ubuntu:

- `/dev/dxg` exists.
- `/usr/lib/wsl/lib/libcuda.so` exists.
- `nvidia-smi` works inside Ubuntu.
- `.venv-v20/bin/python scripts/check_cuda_preflight.py --require-cuda` exits `0`.

Current diagnosis on this machine: Windows sees the RTX GPU, but Ubuntu does not expose `/dev/dxg`, so WSL GPU passthrough is blocked below Python.

Run this from Windows PowerShell:

```powershell
cd \\wsl$\Ubuntu\home\amdev\sota\minimal-shot-av
.\scripts\setup_wsl_gpu_windows.ps1
```

Then reopen Ubuntu and run:

```bash
cd /home/amdev/sota/minimal-shot-av
.venv-v20/bin/python scripts/check_cuda_preflight.py --require-cuda
.venv-v20/bin/python scripts/check_v20_env.py
```

Do not install a Linux NVIDIA kernel driver inside WSL. WSL uses the Windows NVIDIA driver and exposes it through `/dev/dxg` plus `/usr/lib/wsl/lib`.
