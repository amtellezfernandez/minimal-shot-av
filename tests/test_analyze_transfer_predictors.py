from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_transfer_predictors.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("analyze_transfer_predictors", SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class AnalyzeTransferPredictorsTests(unittest.TestCase):
    def test_predictor_confirms_tradeoff_and_marks_missing_axis_constrained_external_run(self) -> None:
        module = _load_module()
        internal_proxy = {
            "by_perturbation": {
                "route_offset": {
                    "summary": {
                        "raw_iter2": {
                            "pass": 0.9,
                            "collision": 0.1,
                            "lane_violation_any": 0.5,
                            "offroad_proxy_any": 0.2,
                            "total_progress_m": 10.0,
                        },
                        "clamped_iter2": {
                            "pass": 0.8,
                            "collision": 0.2,
                            "lane_violation_any": 0.1,
                            "offroad_proxy_any": 0.0,
                            "total_progress_m": 11.0,
                        },
                        "axis_constrained_iter2": {
                            "pass": 1.0,
                            "collision": 0.0,
                            "lane_violation_any": 0.0,
                            "offroad_proxy_any": 0.0,
                            "total_progress_m": 12.0,
                        },
                    }
                }
            }
        }
        alpasim_analysis = {
            "per_model_on_common_scenes": {
                "token_dagger_iter2": {
                    "raw_collision_any": {"rate": 0.6},
                    "raw_offroad": {"rate": 0.9},
                    "raw_wrong_lane": {"rate": 0.7},
                    "raw_progress": {"mean": 0.1},
                    "raw_dist_to_gt_trajectory": {"mean": 10.0},
                },
                "token_dagger_iter2_clamped": {
                    "raw_collision_any": {"rate": 0.7},
                    "raw_offroad": {"rate": 0.5},
                    "raw_wrong_lane": {"rate": 0.2},
                    "raw_progress": {"mean": 0.4},
                    "raw_dist_to_gt_trajectory": {"mean": 5.0},
                },
            }
        }

        report = module.build_predictor_report(
            internal_proxy=internal_proxy,
            alpasim_analysis=alpasim_analysis,
        )
        rows = {row["internal_variant"]: row for row in report["rows"]}

        self.assertEqual("confirmed", rows["clamped_iter2"]["status"])
        self.assertEqual("tradeoff", rows["clamped_iter2"]["external_outcome"]["class"])
        self.assertEqual("needs_external_run", rows["axis_constrained_iter2"]["status"])
        self.assertEqual(1, report["summary"]["confirmed_predictions"])
        self.assertEqual(2, report["summary"]["missing_external_runs"])
        self.assertFalse(report["summary"]["ready_for_claim"])


if __name__ == "__main__":
    unittest.main()
