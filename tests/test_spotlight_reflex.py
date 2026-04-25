from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minimal_shot_av.environment import Obstacle, Scenario, generate_scenario
from minimal_shot_av.perception import perceive_scene
from minimal_shot_av.policy import run_policy, run_spotlight_reflex_policy
from minimal_shot_av.spotlight_reflex import (
    RFS_3S_INDEX,
    RFS_5S_INDEX,
    ManeuverCandidate,
    RfsReference,
    generate_maneuver_candidates,
    score_candidate,
    select_maneuver,
    speed_scale,
    trust_region_score,
)
from minimal_shot_av.world_model import update_world_state


def straight_trajectory(lateral_offset: float = 0.0, longitudinal_offset: float = 0.0) -> list[tuple[float, float]]:
    return [(float(index + 1) + longitudinal_offset, lateral_offset) for index in range(20)]


class RfsGeometryTests(unittest.TestCase):
    def test_speed_scale_is_clipped_piecewise_linear(self) -> None:
        self.assertEqual(speed_scale(0.0), 0.5)
        self.assertEqual(speed_scale(1.4), 0.5)
        self.assertEqual(speed_scale(11.0), 1.0)
        self.assertEqual(speed_scale(15.0), 1.0)
        expected_midpoint = 0.5 + 0.5 * (6.2 - 1.4) / (11.0 - 1.4)
        self.assertAlmostEqual(speed_scale(6.2), expected_midpoint)

    def test_candidate_inside_5s_trust_region_gets_full_reference_score(self) -> None:
        reference = straight_trajectory()
        candidate = straight_trajectory(lateral_offset=1.7)
        score, inside = trust_region_score(candidate, reference, 91.0, 11.0, RFS_5S_INDEX)
        self.assertTrue(inside)
        self.assertEqual(score, 91.0)

    def test_candidate_outside_trust_region_decays_and_floors(self) -> None:
        reference = straight_trajectory()
        candidate = straight_trajectory(lateral_offset=100.0)
        score, inside = trust_region_score(candidate, reference, 91.0, 11.0, RFS_5S_INDEX)
        self.assertFalse(inside)
        self.assertEqual(score, 4.0)

    def test_3s_and_5s_indices_use_distinct_thresholds(self) -> None:
        reference = straight_trajectory()
        candidate = straight_trajectory()
        candidate[RFS_3S_INDEX] = (reference[RFS_3S_INDEX][0], 0.6)
        candidate[RFS_5S_INDEX] = (reference[RFS_5S_INDEX][0], 0.6)

        score_3s, inside_3s = trust_region_score(candidate, reference, 80.0, 1.4, RFS_3S_INDEX)
        score_5s, inside_5s = trust_region_score(candidate, reference, 80.0, 1.4, RFS_5S_INDEX)

        self.assertFalse(inside_3s)
        self.assertLess(score_3s, 80.0)
        self.assertTrue(inside_5s)
        self.assertEqual(score_5s, 80.0)

    def test_invalid_trust_region_index_raises(self) -> None:
        with self.assertRaises(ValueError):
            trust_region_score(straight_trajectory(), straight_trajectory(), 80.0, 5.0, 7)

    def test_score_candidate_selects_best_3s_and_5s_references(self) -> None:
        candidate = ManeuverCandidate("candidate", straight_trajectory(lateral_offset=0.4))
        references = [
            RfsReference("low", straight_trajectory(lateral_offset=3.0), 70.0),
            RfsReference("high", straight_trajectory(lateral_offset=0.4), 95.0),
        ]
        score = score_candidate(candidate, references, 11.0)
        self.assertEqual(score.reference_3s_label, "high")
        self.assertEqual(score.reference_5s_label, "high")
        self.assertEqual(score.combined_score, 95.0)


class ManeuverLibraryTests(unittest.TestCase):
    def test_every_candidate_has_20_finite_points(self) -> None:
        candidates = generate_maneuver_candidates((0.0, 0.0), (1.0, 0.0), 2.0)
        self.assertEqual(len(candidates), 9)
        for candidate in candidates:
            self.assertEqual(len(candidate.trajectory), 20)
            for point in candidate.trajectory:
                self.assertTrue(math.isfinite(point[0]))
                self.assertTrue(math.isfinite(point[1]))

    def test_stop_trajectory_remains_at_current_point(self) -> None:
        stop = next(candidate for candidate in generate_maneuver_candidates((2.0, -1.0), (1.0, 0.0), 2.0) if candidate.name == "stop")
        self.assertTrue(all(point == (2.0, -1.0) for point in stop.trajectory))

    def test_lateral_maneuvers_have_expected_sign(self) -> None:
        candidates = {candidate.name: candidate for candidate in generate_maneuver_candidates((0.0, 0.0), (1.0, 0.0), 2.0)}
        self.assertGreater(candidates["nudge_left"].trajectory[-1][1], 0.0)
        self.assertLess(candidates["nudge_right"].trajectory[-1][1], 0.0)
        self.assertGreater(candidates["evasive_left"].trajectory[-1][1], candidates["nudge_left"].trajectory[-1][1])
        self.assertLess(candidates["evasive_right"].trajectory[-1][1], candidates["nudge_right"].trajectory[-1][1])

    def test_selector_prefers_lateral_avoidance_under_obstacle_pressure(self) -> None:
        scenario = Scenario(
            width=40.0,
            height=20.0,
            lane_center=[(0.0, 0.0), (30.0, 0.0)],
            lane_half_width=5.0,
            obstacles=[Obstacle(x=4.0, y=1.0, radius=1.0)],
            start=(0.0, 0.0),
            goal=(30.0, 0.0),
            seed=123,
        )
        position = scenario.start
        perception = perceive_scene(scenario, position)
        world_state = update_world_state(scenario, position, perception)
        selection = select_maneuver(scenario, position, world_state, perception, 1.25)
        self.assertIn(selection.candidate.name, {"nudge_right", "evasive_right"})


class DemoIntegrationTests(unittest.TestCase):
    def test_demo_artifacts_include_spotlight_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_demo.py"),
                    "--seed",
                    "1",
                    "--policy",
                    "spotlight-reflex",
                    "--artifacts-dir",
                    temp_dir,
                ],
                cwd=ROOT,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            payload = json.loads((Path(temp_dir) / "latest_rollout.json").read_text())

        self.assertEqual(payload["architecture"]["policy"], "spotlight-reflex")
        self.assertIn("exact RFS trust-region", payload["architecture"]["planner"])
        self.assertTrue(payload["rollout"]["success"])
        self.assertFalse(payload["rollout"]["collision"])
        self.assertTrue(payload["rollout"]["reached_goal"])
        first_step = payload["rollout"]["steps"][0]
        self.assertEqual(first_step["candidate_count"], 9)
        self.assertIsInstance(first_step["selected_maneuver"], str)
        self.assertIsInstance(first_step["rfs_score"], float)
        self.assertIsInstance(first_step["rfs_3s_reference"], str)
        self.assertIsInstance(first_step["rfs_5s_reference"], str)

    def test_baseline_and_spotlight_demos_run_on_fixed_seed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            for policy in ("baseline", "spotlight-reflex"):
                subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "scripts" / "run_demo.py"),
                        "--seed",
                        "2",
                        "--policy",
                        policy,
                        "--artifacts-dir",
                        str(Path(temp_dir) / policy),
                    ],
                    cwd=ROOT,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                self.assertTrue((Path(temp_dir) / policy / "latest_rollout.json").exists())

    def test_no_collision_regression_on_deterministic_simulator_seeds(self) -> None:
        for seed in (1, 2, 3):
            baseline = run_policy(generate_scenario(seed))
            spotlight = run_spotlight_reflex_policy(generate_scenario(seed))
            self.assertFalse(baseline.collision, f"baseline collided on seed {seed}")
            self.assertTrue(spotlight.success, f"spotlight-reflex did not succeed on seed {seed}")
            self.assertFalse(spotlight.collision, f"spotlight-reflex collided on seed {seed}")
            self.assertTrue(spotlight.reached_goal, f"spotlight-reflex did not reach the goal on seed {seed}")


if __name__ == "__main__":
    unittest.main()
