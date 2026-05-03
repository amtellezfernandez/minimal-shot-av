#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
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
    nvidia_smi = _command(["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader"])
    return {
        "schema": "cuda_preflight_v1",
        "nvidia_smi_ok": nvidia_smi["returncode"] == 0,
        "nvidia_smi_returncode": nvidia_smi["returncode"],
        "nvidia_smi_stdout": nvidia_smi["stdout"],
        "nvidia_smi_stderr": nvidia_smi["stderr"],
        "dev_dxg_present": Path("/dev/dxg").exists(),
        "wsl_libcuda_present": Path("/usr/lib/wsl/lib/libcuda.so").exists(),
        **torch_info,
    }


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


if __name__ == "__main__":
    raise SystemExit(main())
