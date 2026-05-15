from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minimal_shot_av.simulator.environment import Actor, Scenario
from minimal_shot_av.simulator.interaction_features import INTERACTION_FEATURE_NAMES, candidate_interaction_features


class InteractionFeatureTests(unittest.TestCase):
    def test_candidate_interaction_features_detects_crossing_actor(self) -> None:
        scenario = Scenario(
            width=40.0,
            height=20.0,
            lane_center=[(0.0, 0.0), (20.0, 0.0)],
            lane_half_width=3.0,
            obstacles=[],
            start=(0.0, 0.0),
            goal=(20.0, 0.0),
            seed=1,
            actors=[
                Actor(
                    actor_id="ped-0",
                    kind="pedestrian",
                    x=4.0,
                    y=-2.0,
                    width=0.6,
                    length=0.6,
                    heading=0.0,
                    speed=1.0,
                    vx=0.0,
                    vy=1.5,
                    behavior="crossing",
                    role="crossing_pedestrian",
                )
            ],
        )
        features = candidate_interaction_features(
            scenario,
            (0.0, 0.0),
            [(1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)],
            point_limit=4,
            horizon_seconds=2.0,
        )
        self.assertEqual(len(INTERACTION_FEATURE_NAMES), int(features.shape[0]))
        self.assertEqual(1.0, float(features[-1]))
        self.assertLess(float(features[0]), 3.0)
        self.assertGreaterEqual(float(features[1]), 0.0)


if __name__ == "__main__":
    unittest.main()
