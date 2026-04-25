from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minimal_shot_av.qwen3_vl_stack import (
    build_coc_style_prompt,
    build_sft_record,
    trajectory_64_to_wod20_from_response,
)
from minimal_shot_av.spotlight_reflex import RfsReference
from minimal_shot_av.trajectory_resampling import resample_64wp_10hz_to_20wp_4hz
from minimal_shot_av.wod_e2e import WodE2EPreferenceFrame


def sample_frame() -> WodE2EPreferenceFrame:
    future = [(float(index), 0.0) for index in range(1, 21)]
    return WodE2EPreferenceFrame(
        frame_name="segment-a",
        past_trajectory=[(-2.0, 0.0), (-1.0, 0.0), (0.0, 0.0)],
        future_trajectory=future,
        intent=1,
        init_speed_mps=4.0,
        references=[RfsReference("pref", future, 9.0)],
    )


class Qwen3VlStackTests(unittest.TestCase):
    def test_resamples_64_waypoints_to_wod_20_waypoints(self) -> None:
        trajectory = [(index / 10.0, 0.0) for index in range(1, 65)]
        resampled = resample_64wp_10hz_to_20wp_4hz(trajectory)

        self.assertEqual(len(resampled), 20)
        self.assertAlmostEqual(resampled[0][0], 0.25)
        self.assertAlmostEqual(resampled[-1][0], 5.0)

    def test_prompt_asks_for_compact_causation_json_not_private_cot(self) -> None:
        prompt = build_coc_style_prompt(sample_frame())

        self.assertIn("critical_objects", prompt)
        self.assertIn("Do not output private chain-of-thought", prompt)
        self.assertIn("trajectory_64wp_10hz", prompt)

    def test_sft_record_has_qwen_messages_and_rfs_references(self) -> None:
        record = build_sft_record(sample_frame())

        self.assertEqual(record["frame_name"], "segment-a")
        self.assertEqual([message["role"] for message in record["messages"]], ["system", "user", "assistant"])
        assistant = json.loads(record["messages"][-1]["content"])
        self.assertEqual(len(assistant["trajectory_64wp_10hz"]), 64)
        self.assertEqual(len(record["rfs_references"]), 1)

    def test_response_trajectory_converts_to_wod_format(self) -> None:
        response = {"trajectory_64wp_10hz": [[index / 10.0, 0.0] for index in range(1, 65)]}
        trajectory = trajectory_64_to_wod20_from_response(response)

        self.assertEqual(len(trajectory), 20)
        self.assertAlmostEqual(trajectory[3][0], 1.0)


if __name__ == "__main__":
    unittest.main()
