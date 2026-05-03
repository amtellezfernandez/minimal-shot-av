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
                json.dumps(
                    {
                        "summary": [
                            {
                                "runs": 50,
                                "success_rate": 1.0,
                                "benchmark_pass_rate": 1.0,
                                "collision_rate": 0.0,
                                "near_miss_rate": 0.0,
                                "avg_min_clearance": 2.0,
                                "avg_intervention_rate": 0.1,
                            }
                            for _ in range(11)
                        ]
                    }
                ),
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
            notebook.write_text("{}", encoding="utf-8")

            report = build_report(
                wod_report=wod,
                breakthrough_audit=breakthrough,
                sim_eval=sim,
                runtime_report=runtime,
                notebook=notebook,
            )

        self.assertTrue(report["valid"])
        self.assertEqual("pass", report["criteria"]["technical_excellence"]["status"])
        self.assertEqual("pass", report["criteria"]["novelty"]["status"])
        self.assertEqual("pass", report["criteria"]["feasibility"]["status"])
        self.assertEqual("pass", report["criteria"]["adherence_to_brief"]["status"])

    def test_missing_notebook_fails_adherence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wod = root / "wod.json"
            breakthrough = root / "breakthrough.json"
            sim = root / "sim.json"
            runtime = root / "runtime.json"
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
                json.dumps(
                    {
                        "summary": [
                            {
                                "runs": 50,
                                "success_rate": 1.0,
                                "benchmark_pass_rate": 1.0,
                                "collision_rate": 0.0,
                                "near_miss_rate": 0.0,
                                "avg_min_clearance": 2.0,
                                "avg_intervention_rate": 0.1,
                            }
                            for _ in range(11)
                        ]
                    }
                ),
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

            report = build_report(
                wod_report=wod,
                breakthrough_audit=breakthrough,
                sim_eval=sim,
                runtime_report=runtime,
                notebook=notebook,
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
                json.dumps(
                    {
                        "summary": [
                            {
                                "runs": 50,
                                "success_rate": 1.0,
                                "benchmark_pass_rate": 1.0,
                                "collision_rate": 0.0,
                                "near_miss_rate": 0.0,
                                "avg_min_clearance": 2.0,
                                "avg_intervention_rate": 0.1,
                            }
                            for _ in range(11)
                        ]
                    }
                ),
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
            notebook.write_text("{}", encoding="utf-8")

            report = build_report(
                wod_report=wod,
                breakthrough_audit=breakthrough,
                sim_eval=sim,
                runtime_report=runtime,
                notebook=notebook,
            )

        self.assertTrue(report["valid"])
        self.assertEqual("pass", report["criteria"]["novelty"]["status"])
        self.assertEqual("pass", report["criteria"]["adherence_to_brief"]["status"])


if __name__ == "__main__":
    unittest.main()
