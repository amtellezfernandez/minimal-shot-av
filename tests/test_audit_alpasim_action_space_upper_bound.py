from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_alpasim_action_space_upper_bound.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("audit_alpasim_action_space_upper_bound", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _frame(
    *,
    lead_frames: int,
    token_safe: bool = False,
    token_selected_safe: bool = False,
    direct_safe: bool = False,
    direct_selected_safe: bool = False,
    direct_cost_miss: bool = False,
) -> dict:
    return {
        "status": "ok",
        "lead_frames": lead_frames,
        "token_candidate_set": {
            "any_collision_free_proxy": token_safe,
            "selected_collision_free_proxy": token_selected_safe,
            "any_margin_safe_proxy": token_safe,
            "selected_margin_safe_proxy": token_selected_safe,
        },
        "direct_grid": {
            "any_collision_free_proxy": direct_safe,
            "selected_collision_free_proxy": direct_selected_safe,
            "cost_missed_collision_free_proxy": direct_cost_miss,
            "any_margin_safe_proxy": direct_safe,
            "selected_margin_safe_proxy": direct_selected_safe,
            "cost_missed_margin_safe_proxy": direct_cost_miss,
        },
    }


class AlpaSimActionSpaceUpperBoundTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_module()

    def test_classifies_direct_cost_miss_before_mixed_selector_miss(self) -> None:
        bucket = self.module._classify_scene(
            [
                _frame(
                    lead_frames=8,
                    token_safe=True,
                    token_selected_safe=False,
                    direct_safe=True,
                    direct_selected_safe=False,
                    direct_cost_miss=True,
                )
            ],
            actionable_lead_frames=5,
        )

        self.assertEqual("direct_grid_contains_proxy_safe_action_but_cost_rejects_it", bucket["label"])
        self.assertEqual(1, bucket["direct_cost_missed_proxy_safe_actionable_frame_count"])

    def test_classifies_direct_selected_safe_as_not_a_cost_failure(self) -> None:
        bucket = self.module._classify_scene(
            [
                _frame(
                    lead_frames=8,
                    token_safe=True,
                    token_selected_safe=False,
                    direct_safe=True,
                    direct_selected_safe=True,
                )
            ],
            actionable_lead_frames=5,
        )

        self.assertEqual("direct_grid_cost_selects_proxy_safe_action_before_collision", bucket["label"])
        self.assertEqual(1, bucket["direct_selected_margin_safe_actionable_frame_count"])
        self.assertEqual(0, bucket["direct_cost_missed_margin_safe_actionable_frame_count"])

    def test_classifies_no_proxy_safe_action_space(self) -> None:
        bucket = self.module._classify_scene(
            [_frame(lead_frames=8)],
            actionable_lead_frames=5,
        )

        self.assertEqual("no_proxy_safe_action_in_token_or_direct_action_space", bucket["label"])


if __name__ == "__main__":
    unittest.main()
