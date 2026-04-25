from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minimal_shot_av.certification import (
    DEFAULT_MIN_RANKED_RUNS_PER_LEVEL,
    EvidenceThresholds,
    OddSpec,
    load_odd_spec,
    load_thresholds,
    profile_by_name,
    _minimum_zero_event_trials,
    _nearest_rank_percentile,
    _proportion_interval,
    generate_evidence_report,
)
from minimal_shot_av.compass import _driving_quality_scores
from minimal_shot_av.policy import Rollout, StepRecord


class CertificationEvidenceTests(unittest.TestCase):
    def test_default_odd_spec_is_machine_readable_and_bounded(self) -> None:
        odd = OddSpec()
        self.assertIn("no legal certification claim", odd.exclusions)
        self.assertEqual(odd.speed_range_kph, (0, 120))
        self.assertGreater(len(odd.road_topologies), 1)
        self.assertGreater(len(odd.actor_and_hazard_types), 1)

    def test_profiles_make_certification_assumptions_explicit(self) -> None:
        sotif = profile_by_name("sotif-v0")
        smoke = profile_by_name("smoke")
        self.assertEqual(sotif.thresholds.max_collision_rate_upper_bound, 0.01)
        self.assertGreater(smoke.thresholds.max_collision_rate_upper_bound, sotif.thresholds.max_collision_rate_upper_bound)
        self.assertIn("not a legal certification", sotif.description)

    def test_wilson_interval_bounds_proportion(self) -> None:
        interval = _proportion_interval(successes=94, trials=100, confidence=0.95)
        self.assertLess(interval.lower, interval.estimate)
        self.assertGreater(interval.upper, interval.estimate)
        self.assertEqual(interval.method, "wilson_score")

    def test_zero_collision_minimum_matches_one_percent_threshold(self) -> None:
        minimum = _minimum_zero_event_trials(max_upper_bound=0.01, confidence=0.95)
        self.assertGreater(minimum, DEFAULT_MIN_RANKED_RUNS_PER_LEVEL)
        interval = _proportion_interval(successes=0, trials=minimum, confidence=0.95)
        self.assertLessEqual(interval.upper, 0.01)

    def test_nearest_rank_percentile_uses_lower_tail(self) -> None:
        self.assertEqual(_nearest_rank_percentile([9.0, 1.0, 8.0, 7.0, 6.0], 0.10), 1.0)
        self.assertEqual(_nearest_rank_percentile([9.0, 1.0, 8.0, 7.0, 6.0], 0.50), 7.0)

    def test_evidence_report_does_not_claim_legal_certification(self) -> None:
        report = generate_evidence_report("spotlight-reflex", range(1, 2), profile_name="sotif-v0")
        self.assertEqual(report["schema"], "compass_sotif_evidence_package_v0")
        self.assertEqual(report["claim_type"], "simulation_evidence_package_not_legal_certification")
        self.assertEqual(report["evidence_profile"], "sotif-v0")
        self.assertEqual(report["compass_profile"], "compass-v0")
        self.assertIn("compass_summary", report)
        self.assertIn("statistical_evidence", report)
        self.assertIn("official_level_evidence", report["statistical_evidence"])
        self.assertIn("quality_summary", report["statistical_evidence"])
        self.assertIn("quality", report["statistical_evidence"]["official_level_evidence"][0])
        self.assertIn("worst_compass_score", report["statistical_evidence"]["quality_summary"])
        self.assertIn("p10_route_quality_score", report["statistical_evidence"]["quality_summary"])
        self.assertIn("sotif_alignment", report)
        self.assertIn("evidence_failures", report)
        self.assertEqual(report["sotif_alignment"]["iso_21448_section_9_validation"], "partial_simulation_only")
        self.assertIn("validate abstract-sim findings in sensor-realistic AlpaSim tier", report["remaining_gaps"])

    def test_evidence_report_gates_on_per_level_sample_size(self) -> None:
        report = generate_evidence_report("spotlight-reflex", range(1, 4))
        level_evidence = report["statistical_evidence"]["official_level_evidence"]
        self.assertEqual(len(level_evidence), 4)
        self.assertTrue(any(not level["sample_size_valid"] for level in level_evidence))
        self.assertEqual(report["evidence_status"], "insufficient_evidence_level_sample_size")
        self.assertTrue(all(level["minimum_ranked_runs"] > 30 for level in level_evidence))
        self.assertTrue(
            any(failure["code"] == "insufficient_per_level_collision_confidence" for failure in report["evidence_failures"])
        )
        self.assertIn("increase ranked runs per official evidence level", report["remaining_gaps"])

    def test_evidence_report_accepts_custom_thresholds(self) -> None:
        report = generate_evidence_report(
            "spotlight-reflex",
            range(1, 2),
            thresholds=EvidenceThresholds(
                min_success_rate_lower_bound=0.0,
                max_collision_rate_upper_bound=0.99,
                min_compass_score=0.0,
                min_safety_score=0.0,
                min_route_quality_score=0.0,
                min_comfort_score=0.0,
                min_worst_compass_score=0.0,
                min_worst_safety_score=0.0,
                min_worst_route_quality_score=0.0,
                min_worst_comfort_score=0.0,
                min_ranked_runs_per_official_level=1,
            ),
            profile_name="unit-test",
        )
        self.assertEqual(report["evidence_profile"], "unit-test")
        self.assertEqual(report["thresholds"]["max_collision_rate_upper_bound"], 0.99)

    def test_quality_scores_fail_poor_driving_even_without_collision(self) -> None:
        rollout = Rollout(
            success=True,
            collision=False,
            reached_goal=True,
            steps=[
                StepRecord(
                    t=index,
                    x=float(index),
                    y=0.0,
                    lane_error=3.8,
                    min_obstacle_distance=0.3,
                    uncertainty=0.0,
                    collision_risk=0.0,
                    action_mode="poor",
                    speed=0.05,
                    intervention=True,
                    goal_distance=10.0,
                    progress=0.04,
                    comfort_cost=1.1,
                    active_actor_count=0,
                    stall=True,
                )
                for index in range(8)
            ],
        )

        scores = _driving_quality_scores(rollout)

        self.assertLess(scores["safety_score"], 8.0)
        self.assertLess(scores["route_quality_score"], 6.5)
        self.assertLess(scores["comfort_score"], 6.0)

    def test_loads_odd_and_threshold_json_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            odd_path = temp_path / "odd.json"
            thresholds_path = temp_path / "thresholds.json"
            odd_path.write_text(
                json.dumps({"name": "yard", "speed_range_kph": [0, 20], "road_topologies": ["industrial_yard"]}),
                encoding="utf-8",
            )
            thresholds_path.write_text(
                json.dumps({"min_success_rate_lower_bound": 0.75, "max_collision_rate_upper_bound": 0.25}),
                encoding="utf-8",
            )

            odd = load_odd_spec(odd_path)
            thresholds = load_thresholds(thresholds_path)

        self.assertEqual(odd.name, "yard")
        self.assertEqual(odd.speed_range_kph, (0, 20))
        self.assertEqual(odd.road_topologies, ("industrial_yard",))
        self.assertEqual(thresholds.min_success_rate_lower_bound, 0.75)
        self.assertEqual(thresholds.max_collision_rate_upper_bound, 0.25)

    def test_evidence_cli_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "evidence.json"
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "minimal_shot_av.certification",
                    "--policy",
                    "spotlight-reflex",
                    "--profile",
                    "smoke",
                    "--seed-start",
                    "1",
                    "--seed-end",
                    "1",
                    "--output",
                    str(output),
                ],
                cwd=ROOT,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env={"PYTHONPATH": str(SRC)},
            )
            payload = json.loads(output.read_text())

        self.assertEqual(payload["schema"], "compass_sotif_evidence_package_v0")
        self.assertEqual(payload["policy"], "spotlight-reflex")
        self.assertEqual(payload["evidence_profile"], "smoke")
        self.assertEqual(payload["compass_profile"], "smoke")
        self.assertIn("official_ranked_runs", payload["statistical_evidence"])
        self.assertIn("official_level_evidence", payload["statistical_evidence"])
        self.assertIn("quality_summary", payload["statistical_evidence"])
        self.assertIn("evidence_failures", payload)

    def test_evidence_cli_accepts_custom_compass_profile_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output = temp_path / "evidence.json"
            compass_profile = temp_path / "compass-profile.json"
            compass_profile.write_text(
                json.dumps(
                    {
                        "name": "unit-compass",
                        "official_level_weights": {"0": 0.25, "1": 0.25, "2": 0.25, "3": 0.25},
                        "min_official_coverage_weight": 0.50,
                        "min_runs_per_official_level": 1,
                    }
                ),
                encoding="utf-8",
            )
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "minimal_shot_av.certification",
                    "--policy",
                    "spotlight-reflex",
                    "--profile",
                    "smoke",
                    "--compass-profile-json",
                    str(compass_profile),
                    "--seed-start",
                    "1",
                    "--seed-end",
                    "1",
                    "--output",
                    str(output),
                ],
                cwd=ROOT,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env={"PYTHONPATH": str(SRC)},
            )
            payload = json.loads(output.read_text())

        self.assertEqual(payload["evidence_profile"], "smoke")
        self.assertEqual(payload["compass_profile"], "unit-compass")
        self.assertEqual(payload["compass_summary"]["benchmark_profile"], "unit-compass")


if __name__ == "__main__":
    unittest.main()
