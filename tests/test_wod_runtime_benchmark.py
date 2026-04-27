from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "benchmark_wod_runtime.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("benchmark_wod_runtime", SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class WodRuntimeBenchmarkTests(unittest.TestCase):
    def test_benchmark_reports_online_latency_without_text_runtime(self) -> None:
        module = _load_module()

        report = module.benchmark_runtime(frame_count=8, warmup=1, residual_modes=1, target_ms=1000.0)

        self.assertEqual("online_wod_non_text_runtime", report["benchmark_type"])
        self.assertEqual(8, report["frames"])
        self.assertEqual("contextual", report["selector_features"])
        self.assertTrue(report["meets_target"])
        self.assertGreater(report["candidate_count_mean"], 0)
        self.assertLess(report["total_ms"]["p95"], 1000.0)
        self.assertIn("generation_ms", report)
        self.assertIn("selection_ms", report)

    def test_benchmark_rejects_empty_frame_count(self) -> None:
        module = _load_module()

        with self.assertRaisesRegex(ValueError, "frame_count"):
            module.benchmark_runtime(frame_count=0)


if __name__ == "__main__":
    unittest.main()
