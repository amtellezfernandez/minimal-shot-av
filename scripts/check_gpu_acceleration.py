#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys


def main() -> int:
    report: dict[str, object] = {
        "dev_dxg": os.path.exists("/dev/dxg"),
        "nvidia_smi": shutil.which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi",
        "ld_library_path_has_wsl": "/usr/lib/wsl/lib" in os.environ.get("LD_LIBRARY_PATH", "").split(":"),
    }
    report["nvidia_smi_ok"] = _nvidia_smi_ok(str(report["nvidia_smi"]))
    try:
        import torch
    except ModuleNotFoundError:
        report["torch"] = None
        report["cuda_available"] = False
    else:
        report["torch"] = torch.__version__
        report["torch_cuda"] = torch.version.cuda
        report["cuda_available"] = bool(torch.cuda.is_available())
        report["cuda_device_count"] = int(torch.cuda.device_count())
        if torch.cuda.is_available():
            report["cuda_device_0"] = torch.cuda.get_device_name(0)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("cuda_available") else 1


def _nvidia_smi_ok(path: str) -> bool:
    try:
        result = subprocess.run(
            [path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


if __name__ == "__main__":
    raise SystemExit(main())
