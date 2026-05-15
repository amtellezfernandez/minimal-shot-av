from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "eval_heldout_latin_hypercube.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("eval_heldout_latin_hypercube", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class HeldoutLatinHypercubeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_module()

    def test_latin_hypercube_unit_shape_and_bounds(self) -> None:
        samples = self.module._latin_hypercube_unit(7, 4, self.module.random.Random(123))
        self.assertEqual(7, len(samples))
        self.assertTrue(all(len(sample) == 4 for sample in samples))
        for sample in samples:
            for value in sample:
                self.assertGreaterEqual(value, 0.0)
                self.assertLess(value, 1.0)

    def test_build_profiles_creates_distinct_named_profiles(self) -> None:
        profiles = self.module.build_lhs_profiles(5, seed=2027)
        self.assertEqual(5, len(profiles))
        self.assertEqual(5, len({profile.name for profile in profiles}))
        for profile in profiles:
            lo, hi = profile.gauntlet_lane_half_width_cap
            self.assertLess(lo, hi)
            self.assertEqual((4,), profile.hazard_counts["gauntlet"])
            self.assertIn(profile.hazard_counts["hidden"][0], (1, 2, 3))
            self.assertGreaterEqual(profile.suite_pressure["gauntlet"], 1.35)
            self.assertLessEqual(profile.suite_pressure["gauntlet"], 3.9)


if __name__ == "__main__":
    unittest.main()
