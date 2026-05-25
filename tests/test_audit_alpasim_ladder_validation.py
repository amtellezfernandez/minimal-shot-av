from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_alpasim_ladder_validation.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("audit_alpasim_ladder_validation", SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class AlpaSimLadderValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_module()

    def test_reports_clearance_divergence_and_selector_subset(self) -> None:
        action_space = {
            "scenes": [
                {
                    "scene": "001_clip",
                    "clipgt_id": "clip",
                    "baseline_first_collision": {"frame_index": 10},
                    "bucket": {"direct_selected_proxy_safe_actionable_frame_count": 2},
                    "frames": [
                        _frame(lead=8, selected_clearance=1.2, closest_distance=2.5, radius=1.0),
                        _frame(lead=5, selected_clearance=0.8, closest_distance=1.5, radius=1.0),
                        _frame(lead=0, selected_clearance=-0.2, closest_distance=1.1, radius=1.0),
                    ],
                }
            ]
        }
        counterfactual = {
            "scenes": [
                {
                    "scene": "001_clip",
                    "bucket": {"label": "actor_axis_safe_candidate_missed_by_baseline_selector"},
                }
            ]
        }

        report = self.module.build_validation_report(
            action_space=action_space,
            counterfactual=counterfactual,
            actionable_lead_frames=5,
            realized_near_thresholds=(1.0,),
            proxy_clearance_thresholds=(0.0,),
        )

        self.assertEqual(
            1,
            report["summary"]["clearance_divergence"]["1.0"]["scene_count"],
        )
        self.assertEqual(
            "selector_ranking_secondary",
            report["scenes"][0]["assigned_mode"],
        )
        self.assertTrue(report["scenes"][0]["proxy_clearance_erosion"]["0.0"]["eroded_before_impact"])


def _frame(*, lead: int, selected_clearance: float, closest_distance: float, radius: float) -> dict:
    return {
        "status": "ok",
        "lead_frames": lead,
        "speed_mps": 2.0,
        "hazard_count": 3,
        "closest_hazard": {"distance_m": closest_distance, "radius": radius},
        "direct_grid": {
            "selected_collision_free_proxy": selected_clearance >= 0.0,
            "selected_min_clearance_m": selected_clearance,
        },
    }


if __name__ == "__main__":
    unittest.main()
