from __future__ import annotations

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minimal_shot_av.model.kinematic_candidates import (
    kinematic_candidate_payloads,
    kinematic_trajectories,
    write_kinematic_candidate_jsonl,
)
from minimal_shot_av.model.wod_e2e import WodE2EPreferenceFrame


def sample_frame() -> WodE2EPreferenceFrame:
    return WodE2EPreferenceFrame(
        frame_name="frame-a",
        past_trajectory=[(-2.0, 0.0), (-1.0, 0.0), (0.0, 0.0)],
        future_trajectory=[(float(index), 0.0) for index in range(1, 21)],
        intent=1,
        init_speed_mps=4.0,
        references=[],
    )


class KinematicCandidateTests(unittest.TestCase):
    def test_constant_velocity_candidate_uses_past_ego_step(self) -> None:
        candidates = dict(kinematic_trajectories(sample_frame().past_trajectory))

        self.assertEqual(candidates["constant_velocity"][0], (1.0, 0.0))
        self.assertEqual(candidates["constant_velocity"][-1], (20.0, 0.0))

    def test_payloads_are_wod_candidate_records(self) -> None:
        payloads = kinematic_candidate_payloads([sample_frame()])

        self.assertEqual(len(payloads), 4)
        self.assertEqual(payloads[0]["frame_name"], "frame-a")
        self.assertEqual(payloads[0]["candidate_index"], 0)
        self.assertEqual(len(payloads[0]["trajectory_20wp_4hz"]), 20)

    def test_writes_jsonl(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidates.jsonl"
            count = write_kinematic_candidate_jsonl([sample_frame()], path)
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(count, 4)
        self.assertEqual([row["candidate_index"] for row in rows], [0, 1, 2, 3])

    def test_expanded_profile_adds_speed_and_stop_candidates(self) -> None:
        candidates = dict(kinematic_trajectories(sample_frame().past_trajectory, profile="expanded"))

        self.assertIn("speed_50pct", candidates)
        self.assertIn("stop_by_3s", candidates)
        self.assertGreater(len(candidates), 4)
        self.assertLess(candidates["speed_50pct"][-1][0], candidates["constant_velocity"][-1][0])

    def test_reflex_profile_adds_training_free_evasive_hypotheses(self) -> None:
        candidates = dict(kinematic_trajectories(sample_frame().past_trajectory, profile="reflex"))

        self.assertIn("lane_change_left_3m", candidates)
        self.assertIn("lane_change_right_3m", candidates)
        self.assertIn("avoid_left_return", candidates)
        self.assertIn("yield_then_go", candidates)
        self.assertGreater(candidates["lane_change_left_3m"][-1][1], 2.5)
        self.assertLess(candidates["lane_change_right_3m"][-1][1], -2.5)
        self.assertAlmostEqual(candidates["avoid_left_return"][-1][1], 0.0, places=6)
        self.assertLess(candidates["yield_creep"][-1][0], candidates["constant_velocity"][-1][0])

    def test_internnav_profile_adds_intent_waypoint_hypotheses(self) -> None:
        left = dict(
            kinematic_trajectories(
                sample_frame().past_trajectory,
                profile="internnav",
                intent=2,
                init_speed_mps=4.0,
            )
        )
        right = dict(
            kinematic_trajectories(
                sample_frame().past_trajectory,
                profile="internnav",
                intent=3,
                init_speed_mps=4.0,
            )
        )

        self.assertIn("internnav_s2_waypoint_progress", left)
        self.assertIn("internnav_s2_waypoint_intent", left)
        self.assertIn("internnav_s1_yield_then_track", left)
        self.assertIn("internnav_s1_yield_creep", left)
        self.assertIn("internnav_s1_late_stop_progress", left)
        self.assertGreater(left["internnav_s2_waypoint_progress"][-1][1], 1.0)
        self.assertGreater(left["internnav_s2_waypoint_intent"][-1][1], 2.5)
        self.assertLess(right["internnav_s2_waypoint_progress"][-1][1], -1.0)
        self.assertLess(right["internnav_s2_waypoint_intent"][-1][1], -2.5)
        self.assertLess(
            left["internnav_s2_waypoint_cautious"][-1][0],
            left["internnav_s2_waypoint_progress"][-1][0],
        )
        self.assertLess(
            left["internnav_s1_yield_creep"][-1][0] - left["internnav_s1_yield_creep"][-2][0],
            left["internnav_s1_yield_then_track"][-1][0] - left["internnav_s1_yield_then_track"][-2][0],
        )

    def test_payloads_pass_frame_context_to_internnav_profile(self) -> None:
        frame = WodE2EPreferenceFrame(
            frame_name="frame-left",
            past_trajectory=sample_frame().past_trajectory,
            future_trajectory=sample_frame().future_trajectory,
            intent=2,
            init_speed_mps=4.0,
            references=[],
        )
        payloads = kinematic_candidate_payloads([frame], profile="internnav")
        intent_payload = next(row for row in payloads if row["candidate_name"] == "internnav_s2_waypoint_intent")

        self.assertGreater(intent_payload["trajectory_20wp_4hz"][-1][1], 2.5)


if __name__ == "__main__":
    unittest.main()
