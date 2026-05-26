from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minimal_shot_av.model.nuplan_maneuver_token_adapter import TOKEN_ORDER
from minimal_shot_av.model.nuplan_maneuver_token_adapter import EXPANDED_TOKEN_ORDER
from minimal_shot_av.model.nuplan_maneuver_token_adapter import build_maneuver_token_candidates
from minimal_shot_av.model.nuplan_maneuver_token_adapter import extract_scalar_state
from minimal_shot_av.model.nuplan_maneuver_token_adapter import select_maneuver_token
from minimal_shot_av.model.nuplan_maneuver_token_adapter import six_scalar_vector
from minimal_shot_av.model.nuplan_maneuver_token_adapter import summarize_actors


class NuPlanManeuverTokenAdapterTests(unittest.TestCase):
    def test_extracts_six_scalar_state_from_visible_actors(self) -> None:
        scene = _scene()

        state = extract_scalar_state(scene)
        vector = six_scalar_vector(state)
        actors = summarize_actors(scene)

        self.assertTrue(state.corridor_blocked)
        self.assertGreater(state.obstacle_pressure, 0.0)
        self.assertEqual("right", state.preferred_escape_side)
        self.assertEqual(6, len(vector))
        self.assertEqual(3, actors.visible_actor_count)
        self.assertEqual(1, actors.crossing_actor_count)

    def test_builds_all_maneuver_tokens_and_prefers_open_escape_side(self) -> None:
        scene = _scene()

        candidates = build_maneuver_token_candidates(scene)
        selected = select_maneuver_token(scene)

        self.assertEqual(TOKEN_ORDER, tuple(candidate.token for candidate in candidates))
        self.assertEqual(9, len(candidates))
        self.assertIn(selected.token, {"nudge_right", "evasive_right"})
        self.assertGreater(
            next(candidate for candidate in candidates if candidate.token == "evasive_right").min_proxy_clearance_m,
            next(candidate for candidate in candidates if candidate.token == "maintain").min_proxy_clearance_m,
        )

    def test_expanded_profile_adds_generation_gap_candidates(self) -> None:
        scene = _scene()

        candidates = build_maneuver_token_candidates(scene, candidate_profile="expanded")
        by_token = {candidate.token: candidate for candidate in candidates}

        self.assertEqual(EXPANDED_TOKEN_ORDER, tuple(candidate.token for candidate in candidates))
        self.assertGreater(len(candidates), len(TOKEN_ORDER))
        self.assertIn("wide_right", by_token)
        self.assertIn("reverse_creep", by_token)
        self.assertLess(by_token["reverse_creep"].final_progress_m, 0.0)
        self.assertLess(
            by_token["wide_right"].lateral_offset_m,
            by_token["evasive_right"].lateral_offset_m,
        )


def _scene() -> dict:
    return {
        "scene_id": "hard_merge_001",
        "ego_state": {"speed_mps": 6.0},
        "route": {
            "command": "straight",
            "heading_error_rad": 0.05,
            "lane_offset_m": 0.6,
            "remaining_distance_m": 28.0,
        },
        "actors": [
            {"x_m": 5.0, "y_m": 0.5, "vx_mps": 0.0, "vy_mps": 0.0, "radius_m": 0.8, "visible": True},
            {"x_m": 4.0, "y_m": 2.1, "vx_mps": 0.0, "vy_mps": 0.0, "radius_m": 0.7, "visible": True},
            {"x_m": 8.0, "y_m": -4.0, "vx_mps": 0.0, "vy_mps": 2.0, "radius_m": 0.7, "visible": True},
        ],
    }


if __name__ == "__main__":
    unittest.main()
