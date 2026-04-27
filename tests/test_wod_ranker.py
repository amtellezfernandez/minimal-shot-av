from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minimal_shot_av.model.wod_ranker import WodPreferenceRanker


class WodRankerTests(unittest.TestCase):
    def test_load_predict_and_select_uses_model_score_not_true_rfs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "ranker.json"
            path.write_text(
                json.dumps(
                    {
                        "numeric_features": ["x_5s"],
                        "candidate_names": ["fast", "slow"],
                        "feature_mean": [0.0, 0.0, 0.0],
                        "feature_scale": [1.0, 1.0, 1.0],
                        "weights": [1.0, 0.0, 0.0],
                        "bias": 0.0,
                    }
                ),
                encoding="utf-8",
            )
            ranker = WodPreferenceRanker.load(path)

        slow_high_label = {
            "candidate_name": "slow",
            "candidate_index": 0,
            "rfs_score": 10.0,
            "features": {"x_5s": 1.0},
        }
        fast_low_label = {
            "candidate_name": "fast",
            "candidate_index": 1,
            "rfs_score": 1.0,
            "features": {"x_5s": 2.0},
        }

        self.assertEqual(ranker.predict_row(slow_high_label), 1.0)
        self.assertEqual(ranker.predict_row(fast_low_label), 2.0)
        self.assertEqual(ranker.select_row([slow_high_label, fast_low_label]), fast_low_label)

    def test_select_row_rejects_empty_candidates(self) -> None:
        ranker = WodPreferenceRanker(
            numeric_features=[],
            candidate_names=[],
            candidate_families=[],
            feature_mean=[],
            feature_scale=[],
            weights=[],
            bias=0.0,
        )
        with self.assertRaises(ValueError):
            ranker.select_row([])


if __name__ == "__main__":
    unittest.main()
