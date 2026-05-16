from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "eval_internal_proxy_transfer.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("eval_internal_proxy_transfer", SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class EvalInternalProxyTransferTests(unittest.TestCase):
    def test_named_perturbations_include_expected_axes(self) -> None:
        module = _load_module()
        perturbations = module._named_perturbations()
        self.assertIn("clean", perturbations)
        self.assertIn("latency_3", perturbations)
        self.assertEqual(1.5, perturbations["lane_error_x1.5"].lane_error_scale)
        self.assertEqual(12.0, perturbations["heading_bias_p12"].lane_heading_bias_deg)

    def test_step_rng_is_deterministic_per_seed_tick_and_channel(self) -> None:
        module = _load_module()
        first = module._step_rng(17, 3, 1).gauss(0.0, 1.0)
        second = module._step_rng(17, 3, 1).gauss(0.0, 1.0)
        different = module._step_rng(17, 4, 1).gauss(0.0, 1.0)
        self.assertEqual(first, second)
        self.assertNotEqual(first, different)

    def test_aggregate_reports_axis_conflicts_and_deltas(self) -> None:
        module = _load_module()
        rows = [
            {
                "perturbation": "clean",
                "agents": {
                    "raw_iter2": {
                        "pass": True,
                        "collision": False,
                        "lane_violation_any": False,
                        "offroad_proxy_any": False,
                        "min_clearance_m": 1.0,
                        "total_progress_m": 10.0,
                        "mean_abs_jerk": 0.2,
                        "max_lane_error_m": 2.0,
                    },
                    "clamped_iter2": {
                        "pass": True,
                        "collision": True,
                        "lane_violation_any": False,
                        "offroad_proxy_any": False,
                        "min_clearance_m": 2.0,
                        "total_progress_m": 12.0,
                        "mean_abs_jerk": 0.1,
                        "max_lane_error_m": 1.0,
                    },
                },
            },
            {
                "perturbation": "clean",
                "agents": {
                    "raw_iter2": {
                        "pass": True,
                        "collision": False,
                        "lane_violation_any": False,
                        "offroad_proxy_any": False,
                        "min_clearance_m": 1.5,
                        "total_progress_m": 9.0,
                        "mean_abs_jerk": 0.3,
                        "max_lane_error_m": 3.0,
                    },
                    "clamped_iter2": {
                        "pass": True,
                        "collision": False,
                        "lane_violation_any": True,
                        "offroad_proxy_any": False,
                        "min_clearance_m": 2.5,
                        "total_progress_m": 7.0,
                        "mean_abs_jerk": 0.1,
                        "max_lane_error_m": 4.0,
                    },
                },
            },
        ]
        report = module._aggregate(rows)
        clean = report["clean"]
        self.assertEqual(2, clean["task_count"])
        cmp = clean["comparisons_vs_raw_iter2"]["clamped_iter2"]
        self.assertAlmostEqual(1.0, cmp["mean_deltas"]["min_clearance_m"])
        self.assertAlmostEqual(0.0, cmp["mean_deltas"]["total_progress_m"])
        self.assertEqual(1, cmp["axis_conflicts"]["progress_up_collision_up"])
        self.assertEqual(1, cmp["axis_conflicts"]["clearance_up_progress_down"])
        self.assertEqual(0.5, clean["summary"]["clamped_iter2"]["collision"])


if __name__ == "__main__":
    unittest.main()
