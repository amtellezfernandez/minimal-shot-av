from __future__ import annotations

import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_minor_runtime_constraints.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("audit_minor_runtime_constraints", SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AuditMinorRuntimeConstraintsTests(unittest.TestCase):
    def test_report_validates_closed_loop_runtime_budget(self) -> None:
        module = _load_module()

        report = module.build_report(seed_start=1, seed_end=1, target_step_ms=1000.0)

        self.assertTrue(report["valid"])
        self.assertEqual("minor_runtime_constraints_audit_v1", report["schema"])
        self.assertEqual("closed_loop_simulator_control_runtime", report["contract"])
        self.assertGreaterEqual(report["run_count"], 15)
        self.assertGreater(report["total_steps"], 0)
        self.assertLess(report["p95_step_latency_ms"], 1000.0)
        self.assertTrue(report["gates"]["p95_step_latency_within_budget"])

    def test_report_rejects_invalid_seed_window(self) -> None:
        module = _load_module()

        with self.assertRaisesRegex(ValueError, "seed_end"):
            module.build_report(seed_start=2, seed_end=1)

    def test_main_writes_report(self) -> None:
        with TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "minor_runtime_constraints.json"

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--seed-start",
                    "1",
                    "--seed-end",
                    "1",
                    "--target-step-ms",
                    "1000",
                    "--output",
                    str(output),
                ],
                cwd=ROOT,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            self.assertTrue(output.exists())


if __name__ == "__main__":
    unittest.main()
