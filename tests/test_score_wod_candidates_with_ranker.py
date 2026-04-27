from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

from minimal_shot_av.model.wod_e2e import WodE2EPreferenceFrame
from minimal_shot_av.model.wod_ranker import WodPreferenceRanker, candidate_ranker_row
from minimal_shot_av.model.wod_ranker import raw_features, selector_numeric_features


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "score_wod_candidates_with_ranker.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("score_wod_candidates_with_ranker", SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ScoreWodCandidatesWithRankerTests(unittest.TestCase):
    def test_score_candidate_rows_writes_ranker_score(self) -> None:
        module = _load_module()
        frame = _frame("frame-a")
        candidate_row = {
            "frame_name": "frame-a",
            "source": "wod_ridge_trajectory_non_text",
            "candidate_name": "ridge_mean",
            "candidate_index": 0,
            "trajectory_20wp_4hz": [[float(index), 0.0] for index in range(1, 21)],
        }
        ranker_row = candidate_ranker_row(
            frame=frame,
            trajectory=[(float(index), 0.0) for index in range(1, 21)],
            candidate_name="ridge_mean",
            candidate_index=0,
            source="wod_ridge_trajectory_non_text",
        )
        numeric_features = selector_numeric_features("contextual")
        feature_len = len(raw_features(ranker_row, numeric_features, ["ridge_mean"], ["ridge_mean"]))
        ranker = WodPreferenceRanker(
            numeric_features=numeric_features,
            candidate_names=["ridge_mean"],
            candidate_families=["ridge_mean"],
            feature_mean=[0.0] * feature_len,
            feature_scale=[1.0] * feature_len,
            weights=[0.0] * feature_len,
            bias=3.25,
        )

        with TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "scored.jsonl"
            count = module.score_candidate_rows([candidate_row], {"frame-a": frame}, ranker, output)
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(1, count)
        self.assertEqual(3.25, rows[0]["ranker_score"])
        self.assertEqual(candidate_row["trajectory_20wp_4hz"], rows[0]["trajectory_20wp_4hz"])


def _frame(name: str) -> WodE2EPreferenceFrame:
    return WodE2EPreferenceFrame(
        frame_name=name,
        past_trajectory=[(-1.0, 0.0), (0.0, 0.0)],
        future_trajectory=[(float(index), 0.0) for index in range(20)],
        intent=1,
        init_speed_mps=2.0,
        references=[],
    )


if __name__ == "__main__":
    unittest.main()
