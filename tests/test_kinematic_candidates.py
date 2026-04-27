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


if __name__ == "__main__":
    unittest.main()
