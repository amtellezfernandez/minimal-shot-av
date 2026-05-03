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

    def test_fast_slow_benchmark_separates_control_and_refresh_latency(self) -> None:
        module = _load_module()

        report = module.benchmark_fast_slow_runtime(
            frame_count=12,
            warmup=1,
            slow_residual_modes=2,
            fast_residual_modes=1,
            slow_refresh_frames=3,
            target_ms=1000.0,
        )

        self.assertEqual("online_wod_fast_slow_runtime", report["benchmark_type"])
        self.assertEqual("fast_reflex_slow_deliberation", report["architecture"])
        self.assertEqual(
            "synchronous fast loop excludes asynchronous slow refresh latency",
            report["control_loop_contract"],
        )
        self.assertTrue(report["meets_target"])
        self.assertEqual(1, report["fast_residual_modes"])
        self.assertEqual(2, report["slow_residual_modes"])
        self.assertEqual(3, report["slow_refresh_frames"])
        self.assertGreater(report["candidate_count_mean"], 0)
        self.assertLess(report["total_ms"]["p95"], 1000.0)
        self.assertIn("fast_generation_ms", report)
        self.assertIn("async_slow_refresh_ms", report)
        self.assertIsNotNone(report["async_slow_refresh_ms"])
        self.assertGreaterEqual(report["stale_cache_frame_rate"], 0.0)

    def test_benchmark_rejects_empty_frame_count(self) -> None:
        module = _load_module()

        with self.assertRaisesRegex(ValueError, "frame_count"):
            module.benchmark_runtime(frame_count=0)

    def test_fast_slow_rejects_invalid_refresh_interval(self) -> None:
        module = _load_module()

        with self.assertRaisesRegex(ValueError, "slow_refresh_frames"):
            module.benchmark_fast_slow_runtime(frame_count=8, slow_refresh_frames=0)


if __name__ == "__main__":
    unittest.main()
