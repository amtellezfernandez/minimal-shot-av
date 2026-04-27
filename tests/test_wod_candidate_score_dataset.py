from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_wod_candidate_score_dataset.py"
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minimal_shot_av.model.rfs_metric import RfsReference
from minimal_shot_av.model.wod_e2e import WodE2EPreferenceFrame
from minimal_shot_av.model.zero_shot_eval import candidate_record_from_json


def load_builder_module():
    spec = importlib.util.spec_from_file_location("build_wod_candidate_score_dataset", SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


builder = load_builder_module()


class WodCandidateScoreDatasetTests(unittest.TestCase):
    def test_builds_scored_rows_for_existing_candidate_distribution(self) -> None:
        frame = WodE2EPreferenceFrame(
            frame_name="frame-a",
            past_trajectory=[(-1.0, 0.0), (0.0, 0.0)],
            future_trajectory=[(float(index), 0.0) for index in range(1, 21)],
            intent=1,
            init_speed_mps=4.0,
            references=[RfsReference("human", [(float(index), 0.0) for index in range(1, 21)], 9.0)],
        )
        weak = candidate_record_from_json(
            {
                "frame_name": "frame-a",
                "candidate_name": "weak",
                "candidate_index": 0,
                "trajectory_20wp_4hz": [[float(index), 4.0] for index in range(1, 21)],
            }
        )
        strong = candidate_record_from_json(
            {
                "frame_name": "frame-a",
                "candidate_name": "strong",
                "candidate_index": 1,
                "trajectory_20wp_4hz": [[float(index), 0.0] for index in range(1, 21)],
            }
        )

        rows = builder.build_scored_candidate_rows([frame], {"frame-a": [weak, strong]})

        self.assertEqual(2, len(rows))
        self.assertEqual(["weak", "strong"], [row["candidate_name"] for row in rows])
        self.assertGreater(float(rows[1]["rfs_score"]), float(rows[0]["rfs_score"]))
        self.assertEqual(rows[0]["best_candidate_score"], rows[1]["best_candidate_score"])
        self.assertFalse(rows[0]["is_oracle_best"])
        self.assertTrue(rows[1]["is_oracle_best"])
        self.assertIn("candidate_family", rows[0]["features"])

    def test_mean_best_score_is_frame_weighted_when_oracle_ties(self) -> None:
        rows = [
            {"frame_name": "a", "best_candidate_score": 10.0, "is_oracle_best": True},
            {"frame_name": "a", "best_candidate_score": 10.0, "is_oracle_best": True},
            {"frame_name": "b", "best_candidate_score": 4.0, "is_oracle_best": True},
        ]

        self.assertEqual(7.0, builder._mean_best_score_by_frame(rows))


if __name__ == "__main__":
    unittest.main()
