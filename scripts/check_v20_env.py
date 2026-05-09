#!/usr/bin/env python3
from __future__ import annotations

import importlib
import json
import sys


REQUIRED_MODULES = (
    "numpy",
    "torch",
    "tensorflow",
    "sklearn",
    "transformers",
    "diffusers",
    "accelerate",
    "minimal_shot_av",
)


def main() -> int:
    report = {
        "schema": "v20_environment_check_v1",
        "python": sys.executable,
        "ok": True,
        "modules": {},
    }
    for module_name in REQUIRED_MODULES:
        try:
            module = importlib.import_module(module_name)
            report["modules"][module_name] = {
                "available": True,
                "version": str(getattr(module, "__version__", "unknown")),
            }
        except Exception as exc:  # pragma: no cover - exact import failures vary by host.
            report["ok"] = False
            report["modules"][module_name] = {
                "available": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
    try:
        import torch

        report["torch_cuda_available"] = bool(torch.cuda.is_available())
        report["torch_cuda_device_count"] = int(torch.cuda.device_count())
    except Exception:
        report["torch_cuda_available"] = False
        report["torch_cuda_device_count"] = 0
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
