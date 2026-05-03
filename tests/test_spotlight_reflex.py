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

from minimal_shot_av.simulator.environment import Actor, Obstacle, Scenario, generate_scenario, scenario_at_tick
from minimal_shot_av.simulator.perception import ScenePerception, perceive_scene
from minimal_shot_av.simulator.planner import PlannedAction
from minimal_shot_av.simulator.policy import run_policy, run_spotlight_reflex_policy
from minimal_shot_av.simulator.policy import _blocking_obstacle_row
from minimal_shot_av.simulator.trajectory_selector import (
    TrajectoryCandidate,
    TrajectoryReference,
    score_candidate,
    speed_scale,
    trajectory_region_score,
)
from minimal_shot_av.simulator.safety import apply_safety_filter
from minimal_shot_av.simulator.spotlight_reflex import (
    DEFAULT_SPOTLIGHT_CONFIG,
    SimulatorBackedScoreConfig,
    SpotlightReflexConfig,
    _min_obstacle_clearance,
    _min_obstacle_clearance_with_config,
    generate_maneuver_candidates,
    select_maneuver,
)
from minimal_shot_av.simulator.world_model import WorldState, update_world_state


SELECTOR_3S_INDEX = DEFAULT_SPOTLIGHT_CONFIG.selector.index_3s
SELECTOR_5S_INDEX = DEFAULT_SPOTLIGHT_CONFIG.selector.index_5s


def straight_trajectory(lateral_offset: float = 0.0, longitudinal_offset: float = 0.0) -> list[tuple[float, float]]:
    return [(float(index + 1) + longitudinal_offset, lateral_offset) for index in range(20)]


class TrajectorySelectorGeometryTests(unittest.TestCase):
    def test_speed_scale_is_clipped_piecewise_linear(self) -> None:
        self.assertEqual(speed_scale(0.0), 0.5)
        self.assertEqual(speed_scale(1.4), 0.5)
        self.assertEqual(speed_scale(11.0), 1.0)
        self.assertEqual(speed_scale(15.0), 1.0)
        expected_midpoint = 0.5 + 0.5 * (6.2 - 1.4) / (11.0 - 1.4)
        self.assertAlmostEqual(speed_scale(6.2), expected_midpoint)

    def test_candidate_inside_5s_selection_region_gets_full_reference_score(self) -> None:
        reference = straight_trajectory()
        candidate = straight_trajectory(lateral_offset=1.7)
        score, inside = trajectory_region_score(candidate, reference, 91.0, 11.0, SELECTOR_5S_INDEX)
        self.assertTrue(inside)
        self.assertEqual(score, 91.0)

    def test_candidate_outside_selection_region_decays_and_floors(self) -> None:
        reference = straight_trajectory()
        candidate = straight_trajectory(lateral_offset=100.0)
        score, inside = trajectory_region_score(candidate, reference, 91.0, 11.0, SELECTOR_5S_INDEX)
        self.assertFalse(inside)
        self.assertEqual(score, 4.0)

    def test_3s_and_5s_indices_use_distinct_thresholds(self) -> None:
        reference = straight_trajectory()
        candidate = straight_trajectory()
        candidate[SELECTOR_3S_INDEX] = (reference[SELECTOR_3S_INDEX][0], 0.6)
        candidate[SELECTOR_5S_INDEX] = (reference[SELECTOR_5S_INDEX][0], 0.6)

        score_3s, inside_3s = trajectory_region_score(candidate, reference, 80.0, 1.4, SELECTOR_3S_INDEX)
        score_5s, inside_5s = trajectory_region_score(candidate, reference, 80.0, 1.4, SELECTOR_5S_INDEX)

        self.assertFalse(inside_3s)
        self.assertLess(score_3s, 80.0)
        self.assertTrue(inside_5s)
        self.assertEqual(score_5s, 80.0)

    def test_invalid_selection_region_index_raises(self) -> None:
        with self.assertRaises(ValueError):
            trajectory_region_score(straight_trajectory(), straight_trajectory(), 80.0, 5.0, 7)

    def test_score_candidate_selects_best_3s_and_5s_references(self) -> None:
        candidate = TrajectoryCandidate("candidate", straight_trajectory(lateral_offset=0.4))
        references = [
            TrajectoryReference("low", straight_trajectory(lateral_offset=3.0), 70.0),
            TrajectoryReference("high", straight_trajectory(lateral_offset=0.4), 95.0),
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
        candidates = generate_maneuver_candidates((2.0, -1.0), (1.0, 0.0), 2.0)
        stop = next(candidate for candidate in candidates if candidate.name == "stop")
        self.assertTrue(all(point == (2.0, -1.0) for point in stop.trajectory))

    def test_lateral_maneuvers_have_expected_sign(self) -> None:
        generated_candidates = generate_maneuver_candidates((0.0, 0.0), (1.0, 0.0), 2.0)
        candidates = {candidate.name: candidate for candidate in generated_candidates}
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

    def test_selector_explains_selected_maneuver_and_alternatives(self) -> None:
        scenario = Scenario(
            width=40.0,
            height=20.0,
            lane_center=[(0.0, 0.0), (30.0, 0.0)],
            lane_half_width=5.0,
            obstacles=[Obstacle(x=4.0, y=1.0, radius=1.0)],
            start=(0.0, 0.0),
            goal=(30.0, 0.0),
            seed=124,
        )
        perception = perceive_scene(scenario, scenario.start)
        world_state = update_world_state(scenario, scenario.start, perception)

        selection = select_maneuver(scenario, scenario.start, world_state, perception, 1.25)
        metadata = selection.to_metadata()

        self.assertGreater(selection.effective_score, selection.score.combined_score - 1000.0)
        self.assertGreaterEqual(len(selection.decision_reasons), 6)
        self.assertIn("3s_reference=", selection.decision_reasons[0])
        self.assertIn("action_clearance=", " ".join(selection.decision_reasons))
        self.assertEqual(metadata["decision_reasons"], list(selection.decision_reasons))
        summaries = metadata["top_candidate_summaries"]
        self.assertIsInstance(summaries, list)
        self.assertGreaterEqual(len(summaries), 1)
        self.assertEqual(summaries[0]["candidate"], selection.candidate.name)
        self.assertIn("effective_score", summaries[0])
        self.assertIn("action_clearance_m", summaries[0])
        self.assertIn("reasons", summaries[0])

    def test_forecast_clearance_keeps_static_obstacle_with_same_label_as_moving_actor(self) -> None:
        scenario = Scenario(
            width=20.0,
            height=20.0,
            lane_center=[(0.0, 0.0), (10.0, 0.0)],
            lane_half_width=4.0,
            obstacles=[Obstacle(x=1.0, y=0.0, radius=1.0, kind="barrier", label="shared_role")],
            start=(0.0, 0.0),
            goal=(10.0, 0.0),
            seed=7,
            actors=[
                Actor(
                    actor_id="moving_0",
                    kind="vehicle",
                    x=0.0,
                    y=10.0,
                    width=1.0,
                    length=1.0,
                    heading=-math.pi / 2.0,
                    speed=1.0,
                    vx=0.0,
                    vy=-1.0,
                    behavior="linear",
                    role="shared_role",
                )
            ],
        )
        active = scenario_at_tick(scenario, 0)

        self.assertLess(_min_obstacle_clearance([(1.0, 0.0)], active), 0.0)

    def test_default_clearance_does_not_use_privileged_actor_forecast(self) -> None:
        scenario = Scenario(
            width=20.0,
            height=20.0,
            lane_center=[(0.0, 0.0), (10.0, 0.0)],
            lane_half_width=4.0,
            obstacles=[],
            start=(0.0, 0.0),
            goal=(10.0, 0.0),
            seed=8,
            actors=[
                Actor(
                    actor_id="crossing_0",
                    kind="vehicle",
                    x=0.0,
                    y=1.0,
                    width=1.0,
                    length=1.0,
                    heading=-math.pi / 2.0,
                    speed=4.0,
                    vx=0.0,
                    vy=-4.0,
                    behavior="linear",
                    role="crossing_actor",
                )
            ],
        )
        active = scenario_at_tick(scenario, 0)

        self.assertGreater(_min_obstacle_clearance([(0.0, 0.0)], active), 0.0)

    def test_privileged_forecast_clearance_uses_future_position_for_moving_actor(self) -> None:
        scenario = Scenario(
            width=20.0,
            height=20.0,
            lane_center=[(0.0, 0.0), (10.0, 0.0)],
            lane_half_width=4.0,
            obstacles=[],
            start=(0.0, 0.0),
            goal=(10.0, 0.0),
            seed=8,
            actors=[
                Actor(
                    actor_id="crossing_0",
                    kind="vehicle",
                    x=0.0,
                    y=1.0,
                    width=1.0,
                    length=1.0,
                    heading=-math.pi / 2.0,
                    speed=4.0,
                    vx=0.0,
                    vy=-4.0,
                    behavior="linear",
                    role="crossing_actor",
                )
            ],
        )
        active = scenario_at_tick(scenario, 0)
        config = SpotlightReflexConfig(
            scoring=SimulatorBackedScoreConfig(use_privileged_actor_forecast=True)
        )

        self.assertLess(_min_obstacle_clearance_with_config([(0.0, 0.0)], active, config), 0.0)

    def test_forecast_clearance_replaces_current_moving_actor_obstacle(self) -> None:
        scenario = Scenario(
            width=20.0,
            height=20.0,
            lane_center=[(0.0, 0.0), (10.0, 0.0)],
            lane_half_width=4.0,
            obstacles=[],
            start=(0.0, 0.0),
            goal=(10.0, 0.0),
            seed=9,
            actors=[
                Actor(
                    actor_id="departing_0",
                    kind="vehicle",
                    x=0.0,
                    y=0.0,
                    width=1.0,
                    length=1.0,
                    heading=math.pi / 2.0,
                    speed=40.0,
                    vx=0.0,
                    vy=40.0,
                    behavior="linear",
                    role="departing_actor",
                )
            ],
        )
        active = scenario_at_tick(scenario, 0)

        self.assertLess(_min_obstacle_clearance([(0.0, 0.0)], active), 0.0)

    def test_lane_recovery_steers_toward_world_model_target(self) -> None:
        action = PlannedAction(direction=(1.0, 0.0), speed=1.0, mode="planned", score=0.0)
        world_state = WorldState(
            position=(0.0, 0.0),
            target_point=(3.0, 4.0),
            progress_fraction=0.0,
            collision_risk=0.0,
            goal_distance=10.0,
            uncertainty=0.1,
        )
        perception = ScenePerception(
            lane_index=0,
            lane_point=(0.0, 0.0),
            lane_error=4.0,
            lane_heading=(1.0, 0.0),
            corridor_margin=0.1,
            free_space_confidence=0.9,
            uncertainty=0.1,
            visible_obstacles=[],
        )

        safe_action = apply_safety_filter(action, world_state, perception)

        self.assertEqual(safe_action.mode, "lane_recovery")
        self.assertAlmostEqual(safe_action.direction[0], 0.6)
        self.assertAlmostEqual(safe_action.direction[1], 0.8)

    def test_blocking_obstacle_row_is_label_free(self) -> None:
        scenario = Scenario(
            width=40.0,
            height=30.0,
            lane_center=[(0.0, 0.0), (20.0, 0.0), (40.0, 0.0)],
            lane_half_width=6.0,
            obstacles=[
                Obstacle(x=20.0, y=-4.0, radius=0.9, kind="unknown", label="alpha"),
                Obstacle(x=20.3, y=0.0, radius=0.9, kind="unseen", label="beta"),
                Obstacle(x=19.8, y=4.0, radius=0.9, kind="novel", label="gamma"),
            ],
            start=(0.0, 0.0),
            goal=(40.0, 0.0),
            seed=14,
            cluster="not_intersection",
            tags={"generator": "not_wod"},
        )

        row = _blocking_obstacle_row(scenario)

        self.assertIsNotNone(row)
        assert row is not None
        self.assertAlmostEqual(row[0], 20.03333333333333)


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
        self.assertIn("simulator-native trajectory selector", payload["architecture"]["planner"])
        self.assertTrue(payload["rollout"]["success"])
        self.assertFalse(payload["rollout"]["collision"])
        self.assertTrue(payload["rollout"]["reached_goal"])
        first_step = payload["rollout"]["steps"][0]
        self.assertEqual(first_step["candidate_count"], 9)
        self.assertGreaterEqual(first_step["reference_count"], 1)
        self.assertIsInstance(first_step["selected_maneuver"], str)
        self.assertIsInstance(first_step["selector_score"], float)
        self.assertIsInstance(first_step["selector_effective_score"], float)
        self.assertIsInstance(first_step["selector_3s_reference"], str)
        self.assertIsInstance(first_step["selector_5s_reference"], str)
        self.assertIsInstance(first_step["decision_reason"], str)
        self.assertGreater(len(first_step["decision_reason"]), 0)
        self.assertIsInstance(first_step["decision_reasons"], list)
        self.assertGreaterEqual(len(first_step["decision_reasons"]), 6)
        self.assertIsInstance(first_step["top_candidate_summaries"], list)
        self.assertGreaterEqual(len(first_step["top_candidate_summaries"]), 1)
        self.assertIn("effective_score", first_step["top_candidate_summaries"][0])

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
