from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "train_wod_contextual_ranker.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("train_wod_contextual_ranker", SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


try:
    trainer = _load_module()
except ModuleNotFoundError as exc:
    if exc.name != "numpy":
        raise
    trainer = None


class TrainWodContextualRankerTests(unittest.TestCase):
    def test_frame_delta_targets_are_centered_per_frame(self) -> None:
        if trainer is None:
            self.skipTest("numpy is not installed")
        rows = [
            {"frame_name": "a", "rfs_score": 4.0},
            {"frame_name": "a", "rfs_score": 6.0},
            {"frame_name": "b", "rfs_score": 10.0},
        ]

        targets = trainer.selector_targets(rows, "frame_delta")

        self.assertEqual([-1.0, 1.0, 0.0], targets.tolist())

    def test_fit_contextual_ranker_uses_contextual_features(self) -> None:
        if trainer is None:
            self.skipTest("numpy is not installed")
        base_features = {
            name: 0.0
            for name in trainer.selector_numeric_features("contextual")
        }
        rows = [
            {
                "frame_name": "a",
                "candidate_name": "ridge_mean",
                "candidate_index": 0,
                "rfs_score": 5.0,
                "features": {**base_features, "candidate_family": "ridge_mean"},
            },
            {
                "frame_name": "a",
                "candidate_name": "temporal_ridge_mean",
                "candidate_index": 1,
                "rfs_score": 6.0,
                "features": {**base_features, "candidate_family": "ridge_mean"},
            },
        ]

        ranker = trainer.fit_contextual_ranker(rows, ridge=1.0)

        self.assertIn("source_temporal_x_speed_bin_slow", ranker.numeric_features)
        self.assertEqual(["ridge_mean", "temporal_ridge_mean"], ranker.candidate_names)


if __name__ == "__main__":
    unittest.main()
