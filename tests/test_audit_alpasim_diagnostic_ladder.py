from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_alpasim_diagnostic_ladder.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("audit_alpasim_diagnostic_ladder", SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _analysis(scene_count: int, collision_base: float, collision_candidate: float, improved: int = 0) -> dict:
    return {
        "comparisons_vs_baseline": {
            "candidate": {
                "scene_count": scene_count,
                "binary_metrics": {
                    "raw_collision_any": {
                        "baseline_rate": collision_base,
                        "candidate_rate": collision_candidate,
                        "delta_rate": collision_candidate - collision_base,
                        "improved_scenes": improved,
                        "worsened_scenes": 0,
                        "scene_count": scene_count,
                    }
                },
                "continuous_metrics": {
                    "raw_progress": {
                        "mean_delta": 0.1,
                        "median_delta": 0.1,
                        "positive_count": scene_count,
                        "negative_count": 0,
                    }
                },
                "axis_conflicts": {},
            }
        }
    }


class AlpaSimDiagnosticLadderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_module()

    def test_localizes_residual_to_controller_when_upstream_rungs_are_not_sufficient(self) -> None:
        report = self.module.build_ladder_report(
            matrix_analysis=_analysis(10, 0.6, 0.7),
            actor_completion_analysis=_analysis(30, 0.6, 0.6),
            collision_surface={
                "summary": {
                    "common_scene_count": 30,
                    "baseline_raw_collision_count": 18,
                    "candidate_raw_collision_count": 18,
                    "baseline_first_collision": {"structured_hazards_zero": 18},
                    "candidate_first_collision": {"proxy_hits": 17, "logged_at_impact": 17},
                }
            },
            candidate_counterfactual={
                "summary": {
                    "baseline_collision_scene_count": 18,
                    "actor_axis_proxy_actionable_safe_frame_count": 25,
                    "actor_axis_proxy_selected_safe_frame_count": 20,
                    "actor_axis_proxy_missed_safe_frame_count": 5,
                    "actor_axis_proxy_safe_at_impact_count": 0,
                    "bucket_counts": {
                        "actor_axis_safe_candidate_missed_by_baseline_selector": 2,
                        "mixed_actor_axis_safe_candidate_intermittently_missed": 2,
                        "baseline_selected_actor_axis_safe_candidate_but_collision_persisted": 12,
                    },
                }
            },
            action_space_upper_bound={
                "summary": {
                    "baseline_collision_scene_count": 18,
                    "token_candidate_set": {
                        "scene_count_with_proxy_safe_actionable_frame": 18,
                        "proxy_safe_actionable_frame_count": 986,
                        "selected_proxy_safe_actionable_frame_count": 889,
                    },
                    "direct_grid": {
                        "scene_count_with_proxy_safe_actionable_frame": 18,
                        "scene_count_with_selected_proxy_safe_actionable_frame": 18,
                        "scene_count_with_cost_missed_proxy_safe_actionable_frame": 0,
                        "selected_proxy_safe_actionable_frame_count": 962,
                        "cost_missed_proxy_safe_actionable_frame_count": 0,
                    },
                }
            },
            direct_grid_cost_replay=_analysis(18, 1.0, 16 / 18, improved=2),
            direct_grid_clearance_replay=_analysis(18, 1.0, 16 / 18, improved=2),
        )

        self.assertEqual("controller_execution_or_proxy_mismatch", report["offline_localization"]["top_rung"])
        statuses = report["conclusion"]["rung_statuses"]
        self.assertEqual("visibility_real_but_not_sufficient", statuses["actor_visibility"])
        self.assertEqual(
            "token_safe_candidate_present_in_all_collision_scenes",
            statuses["token_candidate_feasibility"],
        )
        self.assertEqual("selector_miss_secondary", statuses["token_selector_ranking"])
        self.assertEqual(
            "direct_grid_cost_not_open_loop_bottleneck",
            statuses["direct_grid_cost_ranking"],
        )

    def test_trajectory_discrepancy_preserves_per_axis_signed_delta(self) -> None:
        discrepancy = self.module._paired_axis_discrepancies(_analysis(10, 0.6, 0.7))
        axis = discrepancy["candidate"]["axes"]["raw_collision_any"]
        self.assertEqual(0.6, axis["baseline_rate"])
        self.assertEqual(0.7, axis["candidate_rate"])
        self.assertAlmostEqual(0.1, axis["signed_delta"])


if __name__ == "__main__":
    unittest.main()
