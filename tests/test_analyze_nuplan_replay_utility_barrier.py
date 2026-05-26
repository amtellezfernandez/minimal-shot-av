from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

from tests.test_train_nuplan_replay_calibrated_selector import _replay_report


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_nuplan_replay_utility_barrier.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class AnalyzeNuPlanReplayUtilityBarrierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_module(SCRIPT, "analyze_nuplan_replay_utility_barrier")

    def test_utility_barrier_row_measures_progress_retention(self) -> None:
        scene = _replay_report()["scenes"][0]

        row = self.module.utility_barrier_row(scene, near_miss_threshold_m=1.0)

        self.assertIsNotNone(row)
        assert row is not None
        self.assertEqual("maintain", row["proxy_token"])
        self.assertEqual("slow_yield", row["best_replay_safe_token"])
        self.assertEqual(0.5, row["progress_retention"])
        self.assertFalse(row["intervention_like"])

    def test_report_summarizes_retention_thresholds(self) -> None:
        report = self.module.analyze_utility_barrier(
            _replay_report(),
            near_miss_threshold_m=1.0,
            retention_thresholds=[0.25, 0.75],
        )

        self.assertEqual(4, report["recoverable_proxy_failure_count"])
        by_threshold = {
            row["min_progress_retention"]: row
            for row in report["retention_summary"]
        }
        self.assertEqual(4, by_threshold[0.25]["recoverable_count"])
        self.assertEqual(0, by_threshold[0.75]["recoverable_count"])
        self.assertEqual(4, report["retention_bin_token_histogram"][2]["count"])
        self.assertIn("maintain", report["proxy_token_histogram"])
        self.assertIn("slow_yield", report["oracle_token_histogram"])


if __name__ == "__main__":
    unittest.main()
