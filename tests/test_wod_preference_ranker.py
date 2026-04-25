from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "scripts" / "train_wod_preference_ranker.py"


def load_trainer_module():
    spec = importlib.util.spec_from_file_location("train_wod_preference_ranker", TRAINER)
    if spec is None or spec.loader is None:
        raise ImportError(TRAINER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


try:
    trainer = load_trainer_module()
except ModuleNotFoundError as exc:
    if exc.name != "numpy":
        raise
    trainer = None


class WodPreferenceRankerTests(unittest.TestCase):
    def test_segment_id_strips_numeric_frame_suffix(self) -> None:
        if trainer is None:
            self.skipTest("numpy is not installed")
        self.assertEqual(trainer._segment_id("abc-def-150"), "abc-def")
        self.assertEqual(trainer._segment_id("abc-def"), "abc-def")

    def test_split_frames_keeps_same_segment_on_one_side(self) -> None:
        if trainer is None:
            self.skipTest("numpy is not installed")
        rows = [
            {"frame_name": "segment-a-100"},
            {"frame_name": "segment-a-101"},
            {"frame_name": "segment-b-100"},
            {"frame_name": "segment-c-100"},
        ]
        train_frames, test_frames = trainer._split_frames(rows, 0.34, seed=3)

        self.assertTrue(train_frames)
        self.assertTrue(test_frames)
        self.assertFalse(train_frames & test_frames)
        if "segment-a-100" in train_frames:
            self.assertIn("segment-a-101", train_frames)
        if "segment-a-100" in test_frames:
            self.assertIn("segment-a-101", test_frames)

    def test_evaluate_tie_break_does_not_use_true_rfs_score(self) -> None:
        if trainer is None:
            self.skipTest("numpy is not installed")
        candidate_names = ["a", "b"]
        candidate_families: list[str] = []
        model = {
            "bias": 0.0,
            "weights": trainer.np.zeros(len(trainer.NUMERIC_FEATURES) + len(candidate_names)),
            "mean": trainer.np.zeros(len(trainer.NUMERIC_FEATURES) + len(candidate_names)),
            "scale": trainer.np.ones(len(trainer.NUMERIC_FEATURES) + len(candidate_names)),
        }
        base_features = {name: 0.0 for name in trainer.NUMERIC_FEATURES}
        rows = [
            {
                "frame_name": "segment-1-100",
                "candidate_name": "a",
                "candidate_index": 0,
                "rfs_score": 1.0,
                "features": {**base_features, "candidate_name": "a"},
            },
            {
                "frame_name": "segment-1-100",
                "candidate_name": "b",
                "candidate_index": 1,
                "rfs_score": 10.0,
                "features": {**base_features, "candidate_name": "b"},
            },
            {
                "frame_name": "segment-1-100",
                "candidate_name": "logged_future",
                "candidate_index": 2,
                "rfs_score": 5.0,
                "features": {**base_features, "candidate_name": "logged_future"},
            },
        ]

        metrics = trainer._evaluate(rows, model, candidate_names, candidate_families)

        self.assertEqual(metrics["selected_mean_rfs"], 1.0)
        self.assertEqual(metrics["oracle_mean_rfs"], 10.0)
        self.assertEqual(metrics["top1_oracle_match_rate"], 0.0)

    def test_cross_validate_returns_weighted_metrics(self) -> None:
        if trainer is None:
            self.skipTest("numpy is not installed")
        candidate_names = ["logged_future", "alt"]
        candidate_families: list[str] = []
        base_features = {name: 0.0 for name in trainer.NUMERIC_FEATURES}
        rows = []
        for segment in range(4):
            frame_name = f"segment-{segment}-100"
            rows.append(
                {
                    "frame_name": frame_name,
                    "candidate_name": "logged_future",
                    "candidate_index": 0,
                    "rfs_score": 5.0,
                    "features": {**base_features, "candidate_name": "logged_future"},
                }
            )
            rows.append(
                {
                    "frame_name": frame_name,
                    "candidate_name": "alt",
                    "candidate_index": 1,
                    "rfs_score": 6.0,
                    "features": {**base_features, "candidate_name": "alt"},
                }
            )

        metrics = trainer._cross_validate(
            rows,
            candidate_names,
            candidate_families,
            folds=2,
            seed=7,
            ridge=1.0,
        )

        self.assertEqual(metrics["folds"], 2)
        self.assertEqual(metrics["frames"], 4)
        self.assertGreaterEqual(metrics["selected_mean_rfs"], 5.0)


if __name__ == "__main__":
    unittest.main()
