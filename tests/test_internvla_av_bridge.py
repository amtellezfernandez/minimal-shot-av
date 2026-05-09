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

from minimal_shot_av.model.internvla_av_bridge import (
    InternVlaNavigationCue,
    internvla_av_candidate_payloads,
    internvla_av_trajectories,
    internvla_navigation_cue_from_payload,
    internvla_navigation_payload_from_text,
    load_internvla_navigation_cues,
    wod_intent_navigation_instruction,
    write_internvla_av_candidate_jsonl,
)
from minimal_shot_av.model.wod_e2e import WodE2EPreferenceFrame


def sample_frame() -> WodE2EPreferenceFrame:
    return WodE2EPreferenceFrame(
        frame_name="segment-a-100",
        past_trajectory=[(-2.0, 0.0), (-1.0, 0.0), (0.0, 0.0)],
        future_trajectory=[(float(index), 0.0) for index in range(1, 21)],
        intent=1,
        init_speed_mps=4.0,
        references=[],
    )


class InternVlaAvBridgeTests(unittest.TestCase):
    def test_pixel_goal_left_of_image_maps_to_left_lateral_av_candidate(self) -> None:
        trajectories = dict(
            internvla_av_trajectories(
                sample_frame().past_trajectory,
                InternVlaNavigationCue(frame_name="segment-a-100", pixel_goal=(0.25, 0.65)),
                intent=1,
                init_speed_mps=4.0,
            )
        )

        self.assertIn("internvla_s2_pixel_goal", trajectories)
        self.assertIn("internvla_s2_pixel_goal_cautious", trajectories)
        self.assertGreater(trajectories["internvla_s2_pixel_goal"][-1][1], 0.5)
        self.assertLess(
            trajectories["internvla_s2_pixel_goal_cautious"][-1][0],
            trajectories["internvla_s2_pixel_goal"][-1][0],
        )

    def test_stop_output_adds_stop_and_creep_candidates(self) -> None:
        trajectories = dict(
            internvla_av_trajectories(
                sample_frame().past_trajectory,
                InternVlaNavigationCue(frame_name="segment-a-100", action="STOP", stop_probability=0.9),
                intent=1,
                init_speed_mps=4.0,
            )
        )

        self.assertIn("internvla_s2_stop", trajectories)
        self.assertIn("internvla_s1_stop_creep", trajectories)
        self.assertLess(trajectories["internvla_s2_stop"][-1][0], 10.0)

    def test_lookdown_output_adds_forward_and_cautious_av_candidates(self) -> None:
        trajectories = dict(
            internvla_av_trajectories(
                sample_frame().past_trajectory,
                InternVlaNavigationCue(frame_name="segment-a-100", action="LOOK_DOWN"),
                intent=1,
                init_speed_mps=4.0,
            )
        )

        self.assertIn("internvla_s2_lookdown_forward_track", trajectories)
        self.assertIn("internvla_s2_lookdown_intent_track", trajectories)
        self.assertIn("internvla_s2_lookdown_cautious_track", trajectories)
        self.assertIn("internvla_s1_lookdown_yield", trajectories)
        self.assertLess(trajectories["internvla_s2_lookdown_cautious_track"][-1][0], 12.0)
        self.assertGreater(
            trajectories["internvla_s2_lookdown_forward_track"][-1][0],
            trajectories["internvla_s2_lookdown_cautious_track"][-1][0],
        )

    def test_turn_arrow_output_adds_soft_track_and_probe_candidates(self) -> None:
        trajectories = dict(
            internvla_av_trajectories(
                sample_frame().past_trajectory,
                InternVlaNavigationCue(frame_name="segment-a-100", action="TURN_LEFT"),
                intent=1,
                init_speed_mps=4.0,
            )
        )

        self.assertIn("internvla_s2_view_left_track", trajectories)
        self.assertIn("internvla_s2_view_left_probe", trajectories)
        self.assertLess(
            trajectories["internvla_s2_view_left_track"][-1][1],
            trajectories["internvla_s2_view_left_probe"][-1][1],
        )

    def test_loader_accepts_qwen_style_0_to_1000_coordinates(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "cues.jsonl"
            path.write_text(
                json.dumps({"frame_name": "segment-a-100", "pixel_goal": [250, 700], "action": "GO"}) + "\n",
                encoding="utf-8",
            )
            cues = load_internvla_navigation_cues(path)

        self.assertEqual((0.25, 0.7), cues["segment-a-100"].pixel_goal)

    def test_writes_candidate_jsonl_for_matched_frames(self) -> None:
        cues = {
            "segment-a-100": InternVlaNavigationCue(
                frame_name="segment-a-100",
                pixel_goal=(0.45, 0.55),
                latency_ms=42.0,
            )
        }
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidates.jsonl"
            count = write_internvla_av_candidate_jsonl([sample_frame()], cues, path)
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(count, 3)
        self.assertEqual("internvla", rows[0]["source"])
        self.assertEqual("internvla_s2_pixel_goal", rows[0]["candidate_name"])
        self.assertEqual(20, len(rows[0]["trajectory_20wp_4hz"]))
        self.assertEqual(42.0, rows[0]["latency_ms"])

    def test_dense_profile_adds_extra_candidate_templates(self) -> None:
        pixel_goal_trajectories = dict(
            internvla_av_trajectories(
                sample_frame().past_trajectory,
                InternVlaNavigationCue(frame_name="segment-a-100", pixel_goal=(0.45, 0.55)),
                intent=1,
                init_speed_mps=4.0,
                profile="dense",
            )
        )
        forward_trajectories = dict(
            internvla_av_trajectories(
                sample_frame().past_trajectory,
                InternVlaNavigationCue(frame_name="segment-a-100", action="FORWARD"),
                intent=1,
                init_speed_mps=4.0,
                profile="dense",
            )
        )

        self.assertGreater(len(pixel_goal_trajectories), 3)
        self.assertIn("internvla_s2_pixel_goal_wide", pixel_goal_trajectories)
        self.assertIn("internvla_s2_pixel_goal_narrow", pixel_goal_trajectories)
        self.assertGreater(len(forward_trajectories), 1)
        self.assertIn("internvla_s1_forward_yield", forward_trajectories)

    def test_candidate_payloads_skip_frames_without_cues(self) -> None:
        payloads = internvla_av_candidate_payloads([sample_frame()], {})

        self.assertEqual([], payloads)

    def test_text_output_payload_preserves_image_pixel_coordinates(self) -> None:
        payload = internvla_navigation_payload_from_text(
            "segment-a-100",
            "The next waypoint is [541, 261].",
            image_size=(1024, 768),
            latency_ms=12.5,
        )

        cue = internvla_navigation_cue_from_payload(payload)

        self.assertAlmostEqual(541 / 1024, cue.pixel_goal[0])
        self.assertAlmostEqual(261 / 768, cue.pixel_goal[1])
        self.assertEqual(12.5, cue.latency_ms)

    def test_text_output_payload_maps_stop_without_coordinates(self) -> None:
        payload = internvla_navigation_payload_from_text("segment-a-100", "STOP")

        self.assertEqual("STOP", payload["action"])

    def test_text_output_payload_maps_arrow_action_symbols(self) -> None:
        right = internvla_navigation_payload_from_text("segment-a-100", "→→→→")
        left = internvla_navigation_payload_from_text("segment-a-100", "←")
        forward = internvla_navigation_payload_from_text("segment-a-100", "↑")

        self.assertEqual("TURN_RIGHT", right["action"])
        self.assertEqual("TURN_LEFT", left["action"])
        self.assertEqual("FORWARD", forward["action"])

    def test_wod_intent_instruction_distinguishes_turns(self) -> None:
        self.assertIn("left", wod_intent_navigation_instruction(2).lower())
        self.assertIn("right", wod_intent_navigation_instruction(3).lower())
        self.assertIn("forward", wod_intent_navigation_instruction(1).lower())


if __name__ == "__main__":
    unittest.main()
