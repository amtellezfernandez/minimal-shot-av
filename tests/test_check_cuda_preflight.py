from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_cuda_preflight.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("check_cuda_preflight", SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class CheckCudaPreflightTests(unittest.TestCase):
    def test_diagnosis_reports_missing_wsl_gpu_device(self) -> None:
        module = _load_module()
        diagnosis = module._diagnose_cuda(
            {
                "torch_cuda_available": False,
                "wsl_detected": True,
                "wsl_libcuda_present": True,
                "dev_dxg_present": False,
                "nvidia_smi_ok": False,
            }
        )

        self.assertEqual("wsl_gpu_device_missing", diagnosis["status"])
        self.assertTrue(any("/dev/dxg" in action for action in [diagnosis["diagnosis"]]))
        self.assertTrue(any("setup_wsl_gpu_windows.ps1" in action for action in diagnosis["next_actions"]))

    def test_diagnosis_reports_cuda_ready(self) -> None:
        module = _load_module()
        diagnosis = module._diagnose_cuda({"torch_cuda_available": True})

        self.assertEqual("cuda_ready", diagnosis["status"])
        self.assertEqual([], diagnosis["next_actions"])


if __name__ == "__main__":
    unittest.main()
