from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_corl_evidence_strength.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("audit_corl_evidence_strength", SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class AuditCorlEvidenceStrengthTests(unittest.TestCase):
    def test_classifies_tradeoff_when_axes_disagree(self) -> None:
        module = _load_module()
        outcome = module._classify_axis_delta(
            {"collision": 0.5, "progress": 0.2},
            {"collision": 0.7, "progress": 0.8},
            axes={"collision": "lower", "progress": "higher"},
        )
        self.assertEqual("tradeoff", outcome["class"])
        self.assertEqual(["progress"], outcome["improved_axes"])
        self.assertEqual(["collision"], outcome["worsened_axes"])

    def test_build_audit_marks_diagnostic_not_best_paper_when_external_scale_is_small(self) -> None:
        module = _load_module()
        internal_proxy = {
            "config": {"tasks": 288},
            "by_perturbation": {
                "latency": {
                    "summary": {
                        "raw_iter2": {
                            "pass": 0.6,
                            "collision": 0.4,
                            "lane_violation_any": 0.8,
                            "offroad_proxy_any": 0.2,
                        },
                        "hybrid_veto_iter2": {
                            "pass": 0.8,
                            "collision": 0.2,
                            "lane_violation_any": 0.3,
                            "offroad_proxy_any": 0.2,
                        },
                    }
                },
                "route": {
                    "summary": {
                        "raw_iter2": {
                            "pass": 0.9,
                            "collision": 0.1,
                            "lane_violation_any": 0.4,
                            "offroad_proxy_any": 0.0,
                        },
                        "hybrid_veto_iter2": {
                            "pass": 0.8,
                            "collision": 0.2,
                            "lane_violation_any": 0.1,
                            "offroad_proxy_any": 0.0,
                        },
                    }
                },
            },
        }
        alpasim = {
            "all_models_common_scene_count": 10,
            "per_model_on_common_scenes": {
                "token_dagger_iter2": {
                    "raw_collision_any": {"rate": 0.6},
                    "raw_offroad": {"rate": 0.9},
                    "raw_wrong_lane": {"rate": 0.7},
                    "raw_progress": {"mean": 0.1},
                    "raw_dist_to_gt_trajectory": {"mean": 10.0},
                },
                "token_dagger_iter2_hybrid_clamped": {
                    "raw_collision_any": {"rate": 0.8},
                    "raw_offroad": {"rate": 0.2},
                    "raw_wrong_lane": {"rate": 0.7},
                    "raw_progress": {"mean": 0.8},
                    "raw_dist_to_gt_trajectory": {"mean": 20.0},
                },
            },
        }
        wod = {
            "rows": [
                {"name": "Scalar / geometry only", "combined_ranker_mean_rfs": 7.0},
                {"name": "Cosmos 64d nonlinear head", "combined_ranker_mean_rfs": 7.5},
            ]
        }
        report = module.build_audit(
            internal_proxy=internal_proxy,
            alpasim_analysis=alpasim,
            wod_grounding=wod,
            min_best_paper_alpasim_scenes=30,
        )
        self.assertEqual("strong_diagnostic", report["conclusion"]["level"])
        self.assertFalse(report["gates"]["best_paper_external_scale"]["pass"])
        self.assertFalse(report["gates"]["positive_method_uniform_dominance"]["pass"])
        self.assertEqual(1, report["internal_proxy"]["tradeoff_count"])
        self.assertEqual(1, report["alpasim"]["tradeoff_count"])


if __name__ == "__main__":
    unittest.main()
