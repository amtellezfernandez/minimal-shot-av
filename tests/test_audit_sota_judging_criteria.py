from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_sota_judging_criteria import build_report


class SotaJudgingCriteriaAuditTest(unittest.TestCase):
    def test_build_report_passes_with_complete_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wod = root / "wod.json"
            breakthrough = root / "breakthrough.json"
            sim = root / "sim.json"
            runtime = root / "runtime.json"
            minimal = root / "minimal.json"
            notebook = root / "analysis.ipynb"

            wod.write_text(
                json.dumps(
                    {
                        "frames": 479,
                        "kinematic_constant_velocity_mean_rfs": 7.0,
                        "combined_ranker_mean_rfs": 7.7,
                        "combined_oracle_mean_rfs": 9.1,
                        "combined_ranker_regret_to_oracle": 1.4,
                        "slices": {"speed:slow": {}, "intent:2": {}},
                    }
                ),
                encoding="utf-8",
            )
            breakthrough.write_text(
                json.dumps(
                    {
                        "passed": False,
                        "failures": [{"id": "worst_slice_regret_worse"}],
                    }
                ),
                encoding="utf-8",
            )
            sim.write_text(
                json.dumps(_sim_payload()),
                encoding="utf-8",
            )
            runtime.write_text(
                json.dumps(
                    {
                        "beats_all_shared_metrics": True,
                        "comparisons": [
                            {
                                "metric": "p95_total_latency_ms",
                                "subject_value": 1.4,
                                "baseline_value": 14.0,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            minimal.write_text(json.dumps(_minimal_shot_payload()), encoding="utf-8")
            notebook.write_text("{}", encoding="utf-8")

            report = build_report(
                wod_report=wod,
                breakthrough_audit=breakthrough,
                sim_eval=sim,
                runtime_report=runtime,
                notebook=notebook,
                minimal_shot_audit=minimal,
            )

        self.assertTrue(report["valid"])
        self.assertEqual("pass", report["criteria"]["technical_excellence"]["status"])
        self.assertEqual("pass", report["criteria"]["novelty"]["status"])
        self.assertEqual("pass", report["criteria"]["feasibility"]["status"])
        self.assertEqual("pass", report["criteria"]["adherence_to_brief"]["status"])
        self.assertEqual("pass", report["criteria"]["minimal_shot_integrity"]["status"])

    def test_missing_notebook_fails_adherence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wod = root / "wod.json"
            breakthrough = root / "breakthrough.json"
            sim = root / "sim.json"
            runtime = root / "runtime.json"
            minimal = root / "minimal.json"
            notebook = root / "missing.ipynb"

            wod.write_text(
                json.dumps(
                    {
                        "frames": 479,
                        "kinematic_constant_velocity_mean_rfs": 7.0,
                        "combined_ranker_mean_rfs": 7.7,
                        "combined_oracle_mean_rfs": 9.1,
                        "combined_ranker_regret_to_oracle": 1.4,
                        "slices": {},
                    }
                ),
                encoding="utf-8",
            )
            breakthrough.write_text(json.dumps({"passed": False, "failures": [{"id": "x"}]}), encoding="utf-8")
            sim.write_text(
                json.dumps(_sim_payload()),
                encoding="utf-8",
            )
            runtime.write_text(
                json.dumps(
                    {
                        "beats_all_shared_metrics": True,
                        "comparisons": [{"metric": "p95_total_latency_ms", "subject_value": 1.4}],
                    }
                ),
                encoding="utf-8",
            )
            minimal.write_text(json.dumps(_minimal_shot_payload()), encoding="utf-8")

            report = build_report(
                wod_report=wod,
                breakthrough_audit=breakthrough,
                sim_eval=sim,
                runtime_report=runtime,
                notebook=notebook,
                minimal_shot_audit=minimal,
            )

        self.assertFalse(report["valid"])
        self.assertEqual("fail", report["criteria"]["adherence_to_brief"]["status"])
        self.assertIn(
            "analysis_notebook_present",
            report["criteria"]["adherence_to_brief"]["failed_checks"],
        )

    def test_build_report_passes_with_conservative_improvement_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wod = root / "wod.json"
            breakthrough = root / "breakthrough.json"
            sim = root / "sim.json"
            runtime = root / "runtime.json"
            minimal = root / "minimal.json"
            notebook = root / "analysis.ipynb"

            wod.write_text(
                json.dumps(
                    {
                        "frames": 479,
                        "kinematic_constant_velocity_mean_rfs": 7.022,
                        "combined_ranker_mean_rfs": 7.659,
                        "combined_oracle_mean_rfs": 9.098,
                        "combined_ranker_regret_to_oracle": 1.439,
                        "slices": {"speed:slow": {}, "intent:3": {}},
                    }
                ),
                encoding="utf-8",
            )
            breakthrough.write_text(json.dumps({"passed": True, "failures": []}), encoding="utf-8")
            sim.write_text(
                json.dumps(_sim_payload()),
                encoding="utf-8",
            )
            runtime.write_text(
                json.dumps(
                    {
                        "beats_all_shared_metrics": True,
                        "comparisons": [
                            {
                                "metric": "p95_total_latency_ms",
                                "subject_value": 1.4,
                                "baseline_value": 14.0,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            minimal.write_text(json.dumps(_minimal_shot_payload()), encoding="utf-8")
            notebook.write_text("{}", encoding="utf-8")

            report = build_report(
                wod_report=wod,
                breakthrough_audit=breakthrough,
                sim_eval=sim,
                runtime_report=runtime,
                notebook=notebook,
                minimal_shot_audit=minimal,
            )

        self.assertTrue(report["valid"])
        self.assertEqual("pass", report["criteria"]["novelty"]["status"])
        self.assertEqual("pass", report["criteria"]["adherence_to_brief"]["status"])

    def test_episode_dependent_primary_claim_fails_minimal_shot_integrity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wod = root / "wod.json"
            breakthrough = root / "breakthrough.json"
            sim = root / "sim.json"
            runtime = root / "runtime.json"
            minimal = root / "minimal.json"
            notebook = root / "analysis.ipynb"

            wod.write_text(
                json.dumps(
                    {
                        "frames": 479,
                        "kinematic_constant_velocity_mean_rfs": 7.0,
                        "combined_ranker_mean_rfs": 8.0,
                        "combined_oracle_mean_rfs": 9.2,
                        "combined_ranker_regret_to_oracle": 1.2,
                        "slices": {"speed:slow": {}},
                    }
                ),
                encoding="utf-8",
            )
            breakthrough.write_text(json.dumps({"passed": True, "failures": []}), encoding="utf-8")
            sim.write_text(
                json.dumps(_sim_payload()),
                encoding="utf-8",
            )
            runtime.write_text(
                json.dumps(
                    {
                        "beats_all_shared_metrics": True,
                        "comparisons": [{"metric": "p95_total_latency_ms", "subject_value": 1.4}],
                    }
                ),
                encoding="utf-8",
            )
            payload = _minimal_shot_payload()
            payload["valid"] = False
            payload["checks"]["active_policy_has_no_episode_lookup"] = False
            minimal.write_text(json.dumps(payload), encoding="utf-8")
            notebook.write_text("{}", encoding="utf-8")

            report = build_report(
                wod_report=wod,
                breakthrough_audit=breakthrough,
                sim_eval=sim,
                runtime_report=runtime,
                notebook=notebook,
                minimal_shot_audit=minimal,
            )

        self.assertFalse(report["valid"])
        self.assertEqual("fail", report["criteria"]["minimal_shot_integrity"]["status"])
        self.assertEqual("fail", report["criteria"]["technical_excellence"]["status"])


def _minimal_shot_payload() -> dict:
    checks = {
        "active_policy_has_no_episode_lookup": True,
        "randomized_long_tail_coverage": True,
        "closed_loop_zero_collision": True,
        "closed_loop_no_near_miss": True,
        "wod_evidence_declared_auxiliary": True,
        "validation_preferences_not_primary_claim": True,
        "no_strict_zero_shot_wod_claim": True,
    }
    return {"schema": "minimal_shot_claim_audit_v1", "valid": True, "checks": checks}


def _sim_payload() -> dict:
    summary = [
        {
            "runs": 50,
            "success_rate": 1.0,
            "success_rate_ci95_low": 0.929,
            "success_rate_ci95_high": 1.0,
            "benchmark_pass_rate": 1.0,
            "benchmark_pass_rate_ci95_low": 0.929,
            "benchmark_pass_rate_ci95_high": 1.0,
            "trajectory_safety_pass_rate": 1.0,
            "collision_rate": 0.0,
            "near_miss_rate": 0.0,
            "avg_min_clearance": 2.0,
            "avg_intervention_rate": 0.1,
        }
        for _ in range(11)
    ]
    runs = [
        {
            "suite": "wod",
            "cluster": f"cluster_{index % 11}",
            "policy": "spotlight-reflex",
            "success": True,
            "benchmark_pass": True,
            "trajectory_safety_pass": True,
            "trajectory_safety_event_count": 0,
            "max_collision_risk": 0.2,
        }
        for index in range(550)
    ]
    return {
        "runs": runs,
        "summary": summary,
        "curriculum": {
            "generator": "closed_loop_procedural_curriculum",
            "cluster_count": 11,
            "suite_count": 1,
            "topology_count": 3,
            "primary_hazard_type_count": 11,
        },
        "statistics": {
            "unit": "closed-loop rollout",
            "confidence_interval": "Wilson score interval, 95%",
            "by_policy_suite": [{"policy": "spotlight-reflex", "suite": "wod", "pass_at_1": 1.0}],
        },
    }


if __name__ == "__main__":
    unittest.main()
