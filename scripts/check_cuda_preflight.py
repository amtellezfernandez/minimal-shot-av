#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Record local CUDA/GPU availability for WOD-E2E training.")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "cuda_preflight.json")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    report = cuda_preflight_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.require_cuda and not report["torch_cuda_available"]:
        return 2
    return 0


def cuda_preflight_report() -> dict[str, Any]:
    torch_info = _torch_info()
    nvidia_smi_path = _first_existing_command(("nvidia-smi", "/usr/lib/wsl/lib/nvidia-smi"))
    nvidia_smi = _command(
        [nvidia_smi_path, "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader"]
    )
    dev_dxg_present = Path("/dev/dxg").exists()
    wsl_libcuda_present = Path("/usr/lib/wsl/lib/libcuda.so").exists()
    wsl_detected = _is_wsl()
    report = {
        "schema": "cuda_preflight_v1",
        "platform": platform.platform(),
        "python": sys.executable,
        "wsl_detected": wsl_detected,
        "nvidia_smi_ok": nvidia_smi["returncode"] == 0,
        "nvidia_smi_path": nvidia_smi_path,
        "nvidia_smi_returncode": nvidia_smi["returncode"],
        "nvidia_smi_stdout": nvidia_smi["stdout"],
        "nvidia_smi_stderr": nvidia_smi["stderr"],
        "dev_dxg_present": dev_dxg_present,
        "wsl_libcuda_present": wsl_libcuda_present,
        "ld_library_path_has_wsl": "/usr/lib/wsl/lib" in os.environ.get("LD_LIBRARY_PATH", "").split(":"),
        **torch_info,
    }
    report.update(_diagnose_cuda(report))
    return report


def _torch_info() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {
            "torch_import_ok": False,
            "torch_error": f"{type(exc).__name__}: {exc}",
            "torch_version": None,
            "torch_cuda_available": False,
            "torch_cuda_device_count": 0,
            "torch_cuda_devices": [],
        }
    devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "total_memory_bytes": int(props.total_memory),
                }
            )
    return {
        "torch_import_ok": True,
        "torch_error": None,
        "torch_version": str(torch.__version__),
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "torch_cuda_devices": devices,
    }


def _command(command: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(command, check=False, text=True, capture_output=True, timeout=15)
    except Exception as exc:
        return {"returncode": 127, "stdout": "", "stderr": f"{type(exc).__name__}: {exc}"}
    return {
        "returncode": int(completed.returncode),
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def _first_existing_command(candidates: tuple[str, ...]) -> str:
    for candidate in candidates:
        if "/" in candidate and Path(candidate).exists():
            return candidate
        if "/" not in candidate:
            result = _command(["/usr/bin/env", "bash", "-lc", f"command -v {candidate}"])
            if result["returncode"] == 0 and result["stdout"]:
                return str(result["stdout"].splitlines()[0])
    return candidates[0]


def _is_wsl() -> bool:
    try:
        version = Path("/proc/version").read_text(encoding="utf-8", errors="ignore").lower()
    except OSError:
        return False
    return "microsoft" in version or "wsl" in version


def _diagnose_cuda(report: dict[str, Any]) -> dict[str, Any]:
    if report["torch_cuda_available"]:
        return {
            "status": "cuda_ready",
            "diagnosis": "PyTorch can see CUDA.",
            "next_actions": [],
        }
    if report["wsl_detected"] and report["wsl_libcuda_present"] and not report["dev_dxg_present"]:
        return {
            "status": "wsl_gpu_device_missing",
            "diagnosis": (
                "WSL CUDA libraries are mounted, but /dev/dxg is absent. "
                "The Windows host driver is not exposing the GPU device to this running distro."
            ),
            "next_actions": [
                "From Windows PowerShell, run scripts/setup_wsl_gpu_windows.ps1.",
                "Run wsl --shutdown, then start Ubuntu again.",
                "Back in this repo, run .venv-v20/bin/python scripts/check_cuda_preflight.py --require-cuda.",
                "Do not install a Linux NVIDIA kernel driver inside WSL; WSL uses the Windows driver.",
            ],
        }
    if report["wsl_detected"] and not report["wsl_libcuda_present"]:
        return {
            "status": "wsl_cuda_libraries_missing",
            "diagnosis": "This looks like WSL, but /usr/lib/wsl/lib/libcuda.so is missing.",
            "next_actions": [
                "Update WSL from Windows PowerShell with wsl --update.",
                "Install or update the Windows NVIDIA driver.",
                "Run wsl --shutdown and start Ubuntu again.",
            ],
        }
    if not report["nvidia_smi_ok"]:
        return {
            "status": "nvidia_smi_failed",
            "diagnosis": "nvidia-smi is not working in this environment.",
            "next_actions": [
                "Verify the host GPU driver with nvidia-smi outside this shell.",
                "Install a CUDA-enabled PyTorch build after the host driver is visible.",
            ],
        }
    return {
        "status": "torch_cuda_unavailable",
        "diagnosis": "nvidia-smi works, but PyTorch cannot use CUDA.",
        "next_actions": [
            "Install a CUDA-enabled torch build into .venv-v20.",
            "Re-run scripts/check_v20_env.py and scripts/check_cuda_preflight.py.",
        ],
    }


if __name__ == "__main__":
    raise SystemExit(main())
