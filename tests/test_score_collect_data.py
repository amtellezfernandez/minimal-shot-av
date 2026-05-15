from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "score_collect_data.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("score_collect_data", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ScoreCollectDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_module()

    def test_trajectory_to_xyh_encodes_relative_points_and_heading(self) -> None:
        rows = self.module._trajectory_to_xyh(
            (1.0, 2.0),
            [(2.0, 2.0), (2.0, 4.0)],
            point_limit=4,
        )
        expected = np.array(
            [
                [1.0, 0.0, 0.0],
                [1.0, 2.0, np.pi / 2.0],
                [1.0, 2.0, np.pi / 2.0],
                [1.0, 2.0, np.pi / 2.0],
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(rows, expected, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
