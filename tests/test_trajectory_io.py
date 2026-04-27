from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minimal_shot_av.model.trajectory_io import (
    trajectory_64_to_wod20_from_payload,
    trajectory_to_wod20_from_payload,
    upsample_20wp_4hz_to_64wp_10hz,
    validate_wod20_trajectory,
)


class TrajectoryIoTests(unittest.TestCase):
    def test_accepts_20_waypoint_structured_payload(self) -> None:
        trajectory = trajectory_to_wod20_from_payload(
            {"trajectory_20wp_4hz": [[float(index), 0.0] for index in range(1, 21)]}
        )

        self.assertEqual(20, len(trajectory))
        self.assertEqual((1.0, 0.0), trajectory[0])
        self.assertEqual((20.0, 0.0), trajectory[-1])

    def test_resamples_64_waypoint_structured_payload_to_wod20(self) -> None:
        trajectory = trajectory_64_to_wod20_from_payload(
            {"trajectory_64wp_10hz": [[index / 10.0, 0.0] for index in range(1, 65)]}
        )

        self.assertEqual(20, len(trajectory))
        self.assertAlmostEqual(0.25, trajectory[0][0])
        self.assertAlmostEqual(5.0, trajectory[-1][0])

    def test_validator_rejects_wrong_length(self) -> None:
        with self.assertRaisesRegex(ValueError, "expected 20"):
            validate_wod20_trajectory([(0.0, 0.0)] * 19)

    def test_validator_rejects_non_finite_coordinates(self) -> None:
        trajectory = [(0.0, 0.0)] * 19 + [(math.inf, 0.0)]

        with self.assertRaisesRegex(ValueError, "not finite"):
            validate_wod20_trajectory(trajectory)

    def test_validator_accepts_stopped_first_future_point(self) -> None:
        trajectory = validate_wod20_trajectory([(0.0, 0.0)] + [(float(index), 0.0) for index in range(1, 20)])

        self.assertEqual((0.0, 0.0), trajectory[0])

    def test_upsamples_wod20_to_64_waypoints_without_text_io(self) -> None:
        trajectory = [(float(index), 0.0) for index in range(1, 21)]

        upsampled = upsample_20wp_4hz_to_64wp_10hz(trajectory)

        self.assertEqual(64, len(upsampled))
        self.assertEqual((1.0, 0.0), upsampled[0])
        self.assertEqual((20.0, 0.0), upsampled[-1])


if __name__ == "__main__":
    unittest.main()
