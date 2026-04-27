from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_production_av_readiness.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("audit_production_av_readiness", SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ProductionAvReadinessTests(unittest.TestCase):
    def test_readiness_audit_rejects_local_cv_only_evidence(self) -> None:
        module = _load_module()
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            trajectory = root / "trajectory.json"
            trajectory.write_text(
                json.dumps(
                    {
                        "benchmark_type": "segment_grouped_cross_validation",
                        "combined_ranker_mean_rfs": 7.755584,
                        "combined_oracle_mean_rfs": 9.142997,
                        "combined_ranker_mean_normalized_rfs": 8.087251,
                        "reference_ceiling_mean_rfs": 9.601253,
                    }
                ),
                encoding="utf-8",
            )

            report = module.production_readiness_report(
                trajectory_report=trajectory,
                leaderboard_results=root / "missing_leaderboard.json",
                closed_loop_evidence=root / "missing_closed_loop.json",
                safety_case=root / "missing_safety.json",
                integration_manifest=root / "missing_integration.json",
            )

        self.assertFalse(report["production_ready"])
        self.assertEqual("research_harness_only", report["disposition"])
        self.assertFalse(report["gates"]["score_near_normalized_ceiling"])
        self.assertFalse(report["gates"]["score_near_oracle"])
        self.assertFalse(report["gates"]["generalization_beyond_local_frame_cache"])
        self.assertAlmostEqual(
            1.387413,
            report["trajectory_metrics"]["rfs_gap_to_oracle"],
            places=5,
        )
        self.assertTrue(report["trajectory_metrics"]["label_metric_profile"]["local_cv_only"])
        self.assertTrue(
            any(
                "hidden-test labels" in caution
                for caution in report["trajectory_metrics"]["label_metric_profile"]["cautions"]
            )
        )
        self.assertTrue(any("routing remains far from candidate oracle" in item for item in report["blockers"]))
        self.assertTrue(any(action["gate"] == "score_near_oracle" for action in report["next_actions"]))
        score_action = next(
            action for action in report["next_actions"] if action["gate"] == "score_near_normalized_ceiling"
        )
        self.assertEqual("score_near_normalized_ceiling", score_action["gate"])
        self.assertIn("label_metric_profile", score_action)

    def test_readiness_audit_reports_missing_sotif_runs_per_level(self) -> None:
        module = _load_module()
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            trajectory = root / "trajectory.json"
            trajectory.write_text(
                json.dumps(
                    {
                        "combined_ranker_mean_rfs": 8.0,
                        "combined_oracle_mean_rfs": 9.0,
                        "combined_ranker_mean_normalized_rfs": 8.5,
                    }
                ),
                encoding="utf-8",
            )
            closed_loop = root / "closed_loop.json"
            closed_loop.write_text(
                json.dumps(
                    {
                        "schema": "compass_sotif_evidence_package_v0",
                        "evidence_profile": "sotif-v0",
                        "evidence_status": "insufficient_evidence_level_sample_size",
                        "claim_type": "simulation_evidence_package_not_legal_certification",
                        "statistical_evidence": {
                            "official_level_evidence": [
                                {
                                    "level": 3,
                                    "name": "adversarial",
                                    "ranked_runs": 4,
                                    "minimum_ranked_runs": 381,
                                    "sample_size_valid": False,
                                    "success_rate": {"lower": 0.51},
                                    "collision_rate": {"upper": 0.49},
                                    "quality_thresholds_met": True,
                                    "quality": {"safety_score": 8.7, "worst_safety_score": 5.5},
                                }
                            ]
                        },
                    }
                ),
                encoding="utf-8",
            )

            report = module.production_readiness_report(
                trajectory_report=trajectory,
                leaderboard_results=root / "missing_leaderboard.json",
                closed_loop_evidence=closed_loop,
                safety_case=root / "missing_safety.json",
                integration_manifest=root / "missing_integration.json",
            )

        closed_loop_evidence = report["evidence"]["closed_loop"]
        self.assertEqual(377, closed_loop_evidence["total_missing_ranked_runs"])
        self.assertEqual(377, closed_loop_evidence["official_level_evidence"][0]["missing_ranked_runs"])
        action = next(action for action in report["next_actions"] if action["gate"] == "closed_loop_driving_validation")
        self.assertEqual(377, action["total_missing_ranked_runs"])

    def test_readiness_audit_next_actions_follow_configured_thresholds(self) -> None:
        module = _load_module()
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            trajectory = root / "trajectory.json"
            trajectory.write_text(
                json.dumps(
                    {
                        "benchmark_type": "segment_grouped_cross_validation",
                        "frames": 479,
                        "combined_ranker_mean_rfs": 8.0,
                        "combined_oracle_mean_rfs": 10.0,
                        "combined_ranker_mean_normalized_rfs": 8.4,
                        "combined_oracle_mean_normalized_rfs": 9.5,
                        "reference_ceiling_mean_rfs": 10.0,
                    }
                ),
                encoding="utf-8",
            )

            report = module.production_readiness_report(
                trajectory_report=trajectory,
                leaderboard_results=root / "missing_leaderboard.json",
                closed_loop_evidence=root / "missing_closed_loop.json",
                safety_case=root / "missing_safety.json",
                integration_manifest=root / "missing_integration.json",
                min_normalized_score=8.8,
                min_oracle_capture=0.9,
            )

        score_action = next(
            action for action in report["next_actions"] if action["gate"] == "score_near_normalized_ceiling"
        )
        oracle_action = next(action for action in report["next_actions"] if action["gate"] == "score_near_oracle")
        self.assertEqual(8.8, score_action["target"])
        self.assertAlmostEqual(0.4, score_action["gap_to_target"])
        self.assertEqual(0.9, oracle_action["target"])
        self.assertAlmostEqual(0.1, oracle_action["gap"])
        self.assertAlmostEqual(0.884211, oracle_action["normalized_oracle_capture_ratio"], places=5)

    def test_readiness_audit_accepts_complete_evidence_package(self) -> None:
        module = _load_module()
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            trajectory = root / "trajectory.json"
            trajectory.write_text(
                json.dumps(
                    {
                        "combined_ranker_mean_rfs": 9.6,
                        "combined_oracle_mean_rfs": 9.8,
                        "combined_ranker_mean_normalized_rfs": 9.7,
                        "reference_ceiling_mean_rfs": 9.9,
                    }
                ),
                encoding="utf-8",
            )
            leaderboard = root / "leaderboard.json"
            leaderboard.write_text(
                json.dumps(
                    {
                        "schema": "wod_e2e_leaderboard_results_v1",
                        "results": [
                            {
                                "confirmed_hidden_test": True,
                                "hidden_test_rfs": 9.4,
                                "submission_sha256": "abc123",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            closed_loop = root / "closed_loop.json"
            closed_loop.write_text(
                json.dumps(
                    {
                        "schema": "compass_sotif_evidence_package_v0",
                        "evidence_profile": "sotif-v0",
                        "evidence_status": "compass_evidence_threshold_met",
                        "claim_type": "simulation_evidence_package_not_legal_certification",
                    }
                ),
                encoding="utf-8",
            )
            safety = root / "safety.json"
            safety.write_text(
                json.dumps(
                    {
                        "schema": "av_safety_case_v1",
                        "approved_for_public_road_deployment": True,
                        "sections": [
                            "operational_design_domain",
                            "hazard_analysis",
                            "verification_validation",
                            "residual_risk_acceptance",
                            "fallback_and_minimal_risk_condition",
                        ],
                    }
                ),
                encoding="utf-8",
            )
            integration = root / "integration.json"
            integration.write_text(
                json.dumps(
                    {
                        "schema": "av_stack_integration_manifest_v1",
                        "production_av_stack": True,
                        "components": {
                            "perception": "validated",
                            "prediction": "validated",
                            "planning": "validated",
                            "control": "validated",
                            "safety_monitor": "validated",
                        },
                        "closed_loop_interface": "validated",
                        "fault_injection": "validated",
                        "real_time_budget": "validated",
                    }
                ),
                encoding="utf-8",
            )

            report = module.production_readiness_report(
                trajectory_report=trajectory,
                leaderboard_results=leaderboard,
                closed_loop_evidence=closed_loop,
                safety_case=safety,
                integration_manifest=integration,
            )

        self.assertTrue(report["production_ready"])
        self.assertEqual("production_claim_evidence_complete", report["disposition"])
        self.assertTrue(all(report["gates"].values()))
        self.assertEqual([], report["blockers"])


if __name__ == "__main__":
    unittest.main()
