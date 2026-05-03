from __future__ import annotations

import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_wod_leaderboard_attack.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_wod_leaderboard_attack", SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RunWodLeaderboardAttackTests(unittest.TestCase):
    def test_can_build_matrix_requires_test_split_and_frame_list_by_default(self) -> None:
        module = _load_module()
        report = {
            "splits": {"test": {"present": True}},
            "frame_list": {"present": False},
        }

        self.assertFalse(module._can_build_matrix(report, allow_missing_frame_list=False))
        self.assertTrue(module._can_build_matrix(report, allow_missing_frame_list=True))

    def test_blocked_payload_points_to_readiness_report(self) -> None:
        module = _load_module()
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "readiness.json"
            payload = module._blocked_payload(path, {"blockers": ["missing WOD-E2E test split"]})

        self.assertTrue(payload["blocked"])
        self.assertEqual(["missing WOD-E2E test split"], payload["blockers"])
        self.assertIn("readiness.json", payload["readiness_report"])


if __name__ == "__main__":
    unittest.main()
