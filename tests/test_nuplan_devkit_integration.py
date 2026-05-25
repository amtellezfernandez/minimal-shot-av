from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

class NuPlanDevkitIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        path = ROOT / "src" / "minimal_shot_av" / "model" / "nuplan_devkit_integration.py"
        spec = importlib.util.spec_from_file_location("nuplan_devkit_integration", path)
        if spec is None or spec.loader is None:
            raise ImportError(path)
        cls.module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cls.module
        spec.loader.exec_module(cls.module)

    def test_discover_db_files_recurses_nested_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nested = root / "nested" / "deeper"
            nested.mkdir(parents=True)
            first = nested / "a.db"
            second = root / "flat.db"
            first.write_text("", encoding="utf-8")
            second.write_text("", encoding="utf-8")

            discovered = self.module._discover_db_files(data_root=root, db_files=(str(root),))

            self.assertEqual([str(second), str(first)], discovered)

    def test_route_features_fall_back_without_goal_or_future(self) -> None:
        ego = _FakeEgo(x=1.0, y=2.0, heading=0.1)

        command, heading_error_rad, route_remaining_m = self.module._route_features_from_goal(
            ego=ego,
            mission_goal=None,
            future_states=[],
        )

        self.assertEqual("straight", command)
        self.assertEqual(0.0, heading_error_rad)
        self.assertEqual(0.0, route_remaining_m)


ROOT = Path(__file__).resolve().parents[1]


class _FakeRearAxle:
    def __init__(self, *, x: float, y: float, heading: float) -> None:
        self.x = x
        self.y = y
        self.heading = heading


class _FakeEgo:
    def __init__(self, *, x: float, y: float, heading: float) -> None:
        self.rear_axle = _FakeRearAxle(x=x, y=y, heading=heading)


if __name__ == "__main__":
    unittest.main()
