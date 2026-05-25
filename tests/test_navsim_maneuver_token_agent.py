from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minimal_shot_av.model.navsim_maneuver_token_agent import navsim_maneuver_token_poses


class NavsimManeuverTokenAgentTests(unittest.TestCase):
    def test_clamped_variant_reduces_route_lateral_motion(self) -> None:
        raw = navsim_maneuver_token_poses(
            speed_mps=5.0,
            command="left",
            variant="raw",
            num_poses=8,
            interval_length_s=0.5,
        )
        clamped = navsim_maneuver_token_poses(
            speed_mps=5.0,
            command="left",
            variant="clamped",
            num_poses=8,
            interval_length_s=0.5,
        )

        self.assertEqual((8, 3), raw.shape)
        self.assertLess(abs(float(clamped[-1, 1])), abs(float(raw[-1, 1])))

    def test_hard_veto_stops_when_speed_is_low(self) -> None:
        poses = navsim_maneuver_token_poses(
            speed_mps=0.2,
            command="straight",
            variant="hard_veto",
            num_poses=8,
            interval_length_s=0.5,
        )

        self.assertAlmostEqual(0.0, float(poses[-1, 0]))


if __name__ == "__main__":
    unittest.main()
