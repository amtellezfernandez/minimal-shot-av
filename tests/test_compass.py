from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minimal_shot_av.simulator.compass import (
    CompassRun,
    DrivingQualityConfig,
    compass_profile_by_name,
    load_compass_profile,
    _ladder_summary,
    _driving_quality_scores,
    _policy_trace,
    _sanity_scenario,
    evaluate_compass,
    evaluate_ladder,
)
from minimal_shot_av.simulator.compositional_scenarios import generate_compositional_scenario
from minimal_shot_av.simulator.policy import Rollout, StepRecord, run_spotlight_reflex_policy
from minimal_shot_av.simulator.oracle import run_oracle_policy
from minimal_shot_av.simulator.wod_scenarios import WOD_E2E_CLUSTERS


class CompassTests(unittest.TestCase):
    def test_oracle_checks_simple_sanity_scenario(self) -> None:
        check = run_oracle_policy(_sanity_scenario(seed=1))
        self.assertTrue(check.solvable)
        self.assertTrue(check.rollout.success)
        self.assertIn("required_decision", check.reasoning_trace)

    def test_compass_report_contains_composite_scores(self) -> None:
        report = evaluate_compass("spotlight-reflex", "gauntlet", range(1, 3))
        self.assertEqual(report["summary"]["runs"], 2)
        self.assertIn("compass_score", report["summary"])
        self.assertIn("avg_generalization_score", report["summary"])
        self.assertIn("avg_safety_score", report["summary"])
        self.assertIn("avg_route_quality_score", report["summary"])
        self.assertIn("avg_comfort_score", report["summary"])
        self.assertIn("oracle_trace", report["runs"][0])
        self.assertIn("safety_score", report["runs"][0])
        self.assertIn("route_quality_score", report["runs"][0])
        self.assertIn("comfort_score", report["runs"][0])
        self.assertGreaterEqual(report["summary"]["compass_score"], 0.0)
        self.assertLessEqual(report["summary"]["compass_score"], 10.0)

    def test_ladder_report_separates_official_score_from_frontier_probe(self) -> None:
        report = evaluate_ladder("spotlight-reflex", range(1, 2))
        self.assertIn("official_compass_score", report["summary"])
        self.assertIn("frontier_probe", report["summary"])
        self.assertEqual(len(report["summary"]["levels"]), 5)
        official_levels = [level for level in report["summary"]["levels"] if level["official"]]
        frontier_levels = [level for level in report["summary"]["levels"] if not level["official"]]
        self.assertEqual([level["level"] for level in official_levels], [0, 1, 2, 3])
        self.assertEqual([level["level"] for level in frontier_levels], [4])
        self.assertIn(2, report["summary"]["official_weights"])
        self.assertIn("official_coverage_weight", report["summary"])
        self.assertLessEqual(report["summary"]["official_coverage_weight"], 1.0)
        self.assertIn("score_valid", report["summary"])
        self.assertIn("minimum_required_coverage_weight", report["summary"])
        self.assertEqual(
            report["summary"]["reasoning_score_source"],
            "diagnostic_rollout_trace_overlap_v0_not_official",
        )
        self.assertIn("official_score_formula", report["summary"])
        self.assertIn("sample_size_valid", report["summary"])

    def test_compass_profiles_make_scoring_assumptions_explicit(self) -> None:
        default_profile = compass_profile_by_name("compass-v0")
        smoke_profile = compass_profile_by_name("smoke")
        self.assertNotEqual(default_profile.official_level_weights, smoke_profile.official_level_weights)
        self.assertLess(smoke_profile.min_runs_per_official_level, default_profile.min_runs_per_official_level)

        report = evaluate_ladder("spotlight-reflex", range(1, 2), smoke_profile)
        self.assertEqual(report["summary"]["benchmark_profile"], "smoke")
        self.assertEqual(report["summary"]["minimum_runs_per_official_level"], 1)

    def test_loads_custom_compass_profile_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            profile_path = Path(temp_dir) / "compass-profile.json"
            profile_path.write_text(
                json.dumps(
                    {
                        "name": "unit-compass",
                        "official_level_weights": {"0": 0.25, "1": 0.25, "2": 0.25, "3": 0.25},
                        "score_weights": {
                            "safety": 1.0,
                            "route_quality": 0.0,
                            "comfort": 0.0,
                            "recovery": 0.0,
                            "generalization": 0.0,
                        },
                        "driving_quality": {"progress_target_m_per_step": 1.2},
                        "scenario_generation": {
                            "name": "unit-scenario-generator",
                            "suite_pressure": {
                                "compositional": 1.0,
                                "hidden": 1.2,
                                "adversarial": 2.5,
                                "gauntlet": 3.0,
                            },
                            "hazard_counts": {
                                "compositional": [1],
                                "hidden": [1],
                                "adversarial": [3],
                                "gauntlet": [4],
                            },
                            "ambient_base_count": 2,
                            "corridor_clearance": {"default": 2.4},
                            "difficulty": {"base": 0.2, "per_hazard": 0.2},
                        },
                        "oracle": {"horizon_steps": 3, "lookahead_samples": 12},
                        "min_official_coverage_weight": 0.50,
                        "min_runs_per_official_level": 1,
                    }
                ),
                encoding="utf-8",
            )

            profile = load_compass_profile(profile_path)

        self.assertEqual(profile.name, "unit-compass")
        self.assertEqual(profile.score_weights.safety, 1.0)
        self.assertEqual(profile.driving_quality.progress_target_m_per_step, 1.2)
        self.assertEqual(profile.scenario_generation.name, "unit-scenario-generator")
        self.assertEqual(profile.scenario_generation.hazard_counts["adversarial"], (3,))
        self.assertEqual(profile.scenario_generation.ambient_base_count, 2)
        self.assertEqual(profile.oracle.horizon_steps, 3)
        self.assertIn("gauntlet", profile.suite_penalties)

        report = evaluate_ladder("spotlight-reflex", range(1, 2), profile)
        self.assertEqual(report["summary"]["benchmark_profile"], "unit-compass")
        self.assertEqual(report["summary"]["scenario_generation_profile"]["name"], "unit-scenario-generator")
        self.assertEqual(report["summary"]["oracle_config"]["horizon_steps"], 3)
        self.assertEqual(report["summary"]["official_weights"], {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})
        self.assertEqual(
            report["summary"]["official_score_formula"],
            "1.00*safety + 0.00*route_quality + 0.00*comfort + 0.00*recovery + 0.00*generalization normalized_by 1.00",
        )

    def test_driving_quality_penalizes_poor_driving_without_collision(self) -> None:
        rollout = Rollout(
            success=True,
            collision=False,
            reached_goal=True,
            steps=[
                StepRecord(
                    t=index,
                    x=float(index),
                    y=0.0,
                    lane_error=3.5,
                    min_obstacle_distance=0.4,
                    uncertainty=0.0,
                    collision_risk=0.0,
                    action_mode="crawl",
                    speed=0.05,
                    intervention=True,
                    goal_distance=10.0 - index * 0.05,
                    progress=0.05,
                    comfort_cost=1.0,
                    active_actor_count=0,
                    stall=index % 2 == 0,
                )
                for index in range(10)
            ],
        )

        scores = _driving_quality_scores(rollout)

        self.assertLess(scores["safety_score"], 5.0)
        self.assertLess(scores["route_quality_score"], 5.0)
        self.assertLess(scores["comfort_score"], 5.0)

    def test_custom_compass_profile_validates_official_weights(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            profile_path = Path(temp_dir) / "bad-compass-profile.json"
            profile_path.write_text(
                json.dumps({"name": "bad", "official_level_weights": {"0": 1.0}}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "missing official level"):
                load_compass_profile(profile_path)

    def test_repository_stress_profile_loads(self) -> None:
        profile = load_compass_profile(ROOT / "configs" / "compass_stress.json")

        self.assertEqual(profile.name, "compass-stress-v0")
        self.assertEqual(profile.scenario_generation.name, "compositional-stress-v0")
        self.assertGreater(profile.oracle.horizon_steps, compass_profile_by_name("compass-v0").oracle.horizon_steps)

    def test_oracle_gap_policy_successes_remain_ranked(self) -> None:
        runs = [
            CompassRun(
                level=0,
                level_name="sanity",
                suite="sanity",
                cluster="oracle_gap_case",
                seed=seed,
                policy="spotlight-reflex",
                solvable=False,
                success=True,
                collision=False,
                safety_score=8.0,
                route_quality_score=8.0,
                comfort_score=8.0,
                reasoning_quality=0.0,
                reasoning_score_source="diagnostic_rollout_trace_overlap_v0_not_official",
                recovery_rate=8.0,
                generalization_score=8.0,
                compass_score=8.0,
                oracle_trace="feasibility_check_failed",
                policy_trace="policy_modes=progress",
                failure_axes="",
                oracle_failed_policy_succeeded=True,
                oracle_failed_policy_failed=False,
            )
            for seed in range(1, 4)
        ]
        summary = _ladder_summary(runs)
        level_zero = next(level for level in summary["levels"] if level["level"] == 0)

        self.assertEqual(level_zero["oracle_gap_runs"], 3)
        self.assertEqual(level_zero["excluded_unsolvable"], 0)
        self.assertEqual(level_zero["ranked_runs"]["runs"], 3)
        self.assertEqual(level_zero["ranked_runs"]["compass_score"], 8.0)

    def test_wod_level_evaluates_every_cluster_for_each_seed(self) -> None:
        report = evaluate_compass("spotlight-reflex", "wod", range(1, 2))
        clusters = {run["cluster"] for run in report["runs"]}
        self.assertEqual(len(report["runs"]), len(WOD_E2E_CLUSTERS))
        self.assertEqual(clusters, set(WOD_E2E_CLUSTERS))

    def test_policy_trace_does_not_leak_scenario_tags(self) -> None:
        scenario = generate_compositional_scenario(seed=1, suite="gauntlet")
        rollout = run_spotlight_reflex_policy(scenario)
        trace = _policy_trace(rollout)
        self.assertNotIn(str(scenario.tags["primary_hazard_type"]), trace)
        self.assertNotIn(str(scenario.tags["hazard_composition"]), trace)
        self.assertNotIn(str(scenario.tags["intended_decision"]), trace)
        self.assertIn("policy_modes=", trace)

    def test_compass_cli_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "report.json"
            env = os.environ.copy()
            env["PYTHONPATH"] = str(SRC)
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "minimal_shot_av.simulator.compass",
                    "run",
                    "--policy",
                    "spotlight-reflex",
                    "--suite",
                    "compositional",
                    "--seed-start",
                    "1",
                    "--seed-end",
                    "1",
                    "--output",
                    str(output),
                ],
                cwd=ROOT,
                env=env,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            payload = json.loads(output.read_text())

        self.assertEqual(payload["summary"]["runs"], 1)
        self.assertIn("reasoning_quality", payload["runs"][0])
        self.assertEqual(
            payload["runs"][0]["reasoning_score_source"],
            "diagnostic_rollout_trace_overlap_v0_not_official",
        )
        self.assertIn("oracle_failed_policy_succeeded", payload["runs"][0])
        self.assertIn("oracle_failed_policy_failed", payload["runs"][0])
        self.assertIn("oracle_failed_policy_failed", payload["summary"])

    def test_benchmark_cli_is_ladder_alias(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "report.json"
            env = os.environ.copy()
            env["PYTHONPATH"] = str(SRC)
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "minimal_shot_av.simulator.compass",
                    "benchmark",
                    "--policy",
                    "spotlight-reflex",
                    "--seed-start",
                    "1",
                    "--seed-end",
                    "1",
                    "--output",
                    str(output),
                ],
                cwd=ROOT,
                env=env,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            payload = json.loads(output.read_text())

        self.assertIn("official_compass_score", payload["summary"])
        self.assertIn("frontier_probe", payload["summary"])


if __name__ == "__main__":
    unittest.main()
