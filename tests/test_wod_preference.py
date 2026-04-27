from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minimal_shot_av.model.rfs_metric import RfsReference
from minimal_shot_av.model.wod_e2e import WodE2EPreferenceFrame
from minimal_shot_av.model.wod_preference import generate_preference_candidates, trajectory_features


def make_frame() -> WodE2EPreferenceFrame:
    return WodE2EPreferenceFrame(
        frame_name="frame-1",
        past_trajectory=[(-1.0, 0.0), (0.0, 0.0)],
        future_trajectory=[(float(step), 0.0) for step in range(1, 21)],
        intent=1,
        init_speed_mps=4.0,
        references=[RfsReference("ref", [(float(step), 0.0) for step in range(20)], 10.0)],
    )


class WodPreferenceCandidateTests(unittest.TestCase):
    def test_generate_preference_candidates_includes_baselines_and_maneuvers(self) -> None:
        candidates = generate_preference_candidates(make_frame())
        names = {candidate.name for candidate in candidates}

        self.assertIn("logged_future", names)
        self.assertIn("constant_velocity", names)
        self.assertIn("stop", names)
        self.assertIn("offset_left_1m", names)
        self.assertIn("offset_right_1m", names)
        self.assertIn("arc_left_0.7", names)
        self.assertIn("decel_hard", names)
        self.assertGreaterEqual(len(candidates), 25)
        for candidate in candidates:
            self.assertEqual(len(candidate.trajectory), 20)
            self.assertEqual(candidate.features["candidate_name"], candidate.name)
            self.assertIn("candidate_family", candidate.features)
            self.assertEqual(candidate.features["intent"], 1)
            self.assertEqual(candidate.features["init_speed_mps"], 4.0)

    def test_trajectory_features_groups_learned_candidate_families(self) -> None:
        trajectory = [(float(index), 0.0) for index in range(1, 21)]

        anchor_features = trajectory_features(
            trajectory,
            candidate_name="anchor_residual_3_rank0_pc1_plus",
            intent=1,
            init_speed_mps=4.0,
        )
        ridge_features = trajectory_features(
            trajectory,
            candidate_name="ridge_residual_pc1_plus",
            intent=1,
            init_speed_mps=4.0,
        )

        self.assertEqual("anchor", anchor_features["candidate_family"])
        self.assertEqual("ridge_residual", ridge_features["candidate_family"])

    def test_logged_future_candidate_preserves_future_trajectory(self) -> None:
        frame = make_frame()
        logged = next(
            candidate for candidate in generate_preference_candidates(frame) if candidate.name == "logged_future"
        )
        self.assertEqual(logged.trajectory, frame.future_trajectory)

    def test_trajectory_features_capture_endpoint_and_lateral_shape(self) -> None:
        features = trajectory_features(
            [(float(step), float(step % 2)) for step in range(1, 21)],
            candidate_name="zigzag",
            intent=3,
            init_speed_mps=2.0,
        )

        self.assertEqual(features["candidate_name"], "zigzag")
        self.assertEqual(features["intent"], 3)
        self.assertEqual(features["x_3s"], 12.0)
        self.assertEqual(features["x_5s"], 20.0)
        self.assertEqual(features["y_5s"], 0.0)
        self.assertEqual(features["max_lateral_abs"], 1.0)
        self.assertEqual(features["lateral_range"], 1.0)
        self.assertIn("x_1s", features)
        self.assertIn("y_4s", features)
        self.assertIn("mean_speed_mps", features)
        self.assertIn("mean_abs_heading_change", features)

    def test_left_intent_adds_left_biased_candidates(self) -> None:
        frame = WodE2EPreferenceFrame(
            frame_name="frame-left",
            past_trajectory=[(-1.0, 0.0), (0.0, 0.0)],
            future_trajectory=[(float(step), 0.0) for step in range(1, 21)],
            intent=2,
            init_speed_mps=4.0,
            references=[RfsReference("ref", [(float(step), 0.0) for step in range(20)], 10.0)],
        )
        names = {candidate.name for candidate in generate_preference_candidates(frame)}
        self.assertIn("intent_left_arc", names)
        self.assertIn("intent_left_wide", names)


if __name__ == "__main__":
    unittest.main()
