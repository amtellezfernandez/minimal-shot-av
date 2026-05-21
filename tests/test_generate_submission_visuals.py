from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "src" / "minimal_shot_av" / "cli" / "commands" / "generate_submission_visuals.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("generate_submission_visuals", SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GenerateSubmissionVisualsTests(unittest.TestCase):
    def test_actor_pose_respects_active_window(self) -> None:
        module = _load_module()
        actor = {
            "x": 10.0,
            "y": 20.0,
            "vx": 2.0,
            "vy": 0.0,
            "speed": 2.0,
            "heading": 0.0,
            "behavior": "linear",
            "active_from": 8,
            "active_until": 20,
        }

        self.assertIsNone(module._actor_pose_at_tick(actor, 7))
        self.assertEqual((10.0, 20.0), module._actor_pose_at_tick(actor, 8))
        self.assertEqual((11.0, 20.0), module._actor_pose_at_tick(actor, 10))
        self.assertIsNone(module._actor_pose_at_tick(actor, 21))

    def test_wrong_way_uses_tick_dt_not_raw_tick_seconds(self) -> None:
        module = _load_module()
        actor = {
            "x": 30.0,
            "y": 4.0,
            "vx": -0.8,
            "vy": 0.0,
            "speed": 0.8,
            "heading": 3.141592653589793,
            "behavior": "wrong_way",
            "active_from": 24,
            "active_until": 60,
        }

        x, y = module._actor_pose_at_tick(actor, 28)
        self.assertAlmostEqual(29.2, x, places=4)
        self.assertAlmostEqual(4.0, y, places=4)


if __name__ == "__main__":
    unittest.main()
