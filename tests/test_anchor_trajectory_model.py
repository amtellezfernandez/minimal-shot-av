from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minimal_shot_av.model.anchor_trajectory_model import (
    anchor_candidate_payloads,
    fit_anchor_residual_trajectory_model,
)
from minimal_shot_av.model.rfs_metric import RfsReference
from minimal_shot_av.model.wod_e2e import WodE2EPreferenceFrame


class AnchorTrajectoryModelTests(unittest.TestCase):
    def test_fit_generates_ranked_anchor_candidates(self) -> None:
        frames = _frames()

        model = fit_anchor_residual_trajectory_model(
            frames,
            anchor_count=2,
            ridge=1.0,
            anchor_iterations=4,
            seed=3,
        )
        candidates = model.candidate_trajectories(
            frames[0].past_trajectory,
            intent=frames[0].intent,
            init_speed_mps=frames[0].init_speed_mps,
            top_k=2,
        )

        self.assertEqual(2, len(candidates))
        self.assertEqual(20, len(candidates[0][1]))
        self.assertTrue(candidates[0][0].startswith("anchor_residual_"))
        self.assertGreaterEqual(candidates[0][2], candidates[1][2])

    def test_model_round_trips_json(self) -> None:
        model = fit_anchor_residual_trajectory_model(
            _frames(),
            anchor_count=2,
            ridge=1.0,
            anchor_iterations=4,
            seed=3,
        )
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "anchor_model.json"
            model.save(path)
            loaded = type(model).load(path)

        self.assertEqual(model.train_rows, loaded.train_rows)
        self.assertEqual(len(model.anchors), len(loaded.anchors))

    def test_anchor_candidate_payloads_include_confidence(self) -> None:
        frames = _frames()
        model = fit_anchor_residual_trajectory_model(
            frames,
            anchor_count=2,
            ridge=1.0,
            anchor_iterations=4,
            seed=3,
        )

        payloads = anchor_candidate_payloads(frames[:1], model, top_k=2)

        self.assertEqual(2, len(payloads))
        self.assertIn("confidence", payloads[0])
        self.assertEqual(20, len(payloads[0]["trajectory_20wp_4hz"]))


def _frames() -> list[WodE2EPreferenceFrame]:
    base = WodE2EPreferenceFrame(
        frame_name="segment-0",
        past_trajectory=[(float(index), 0.0) for index in range(16)],
        future_trajectory=[(float(index), 0.0) for index in range(1, 21)],
        intent=1,
        init_speed_mps=4.0,
        references=[RfsReference("ref", [(float(index), 0.0) for index in range(1, 21)], 10.0)],
    )
    return [
        base,
        replace(
            base,
            frame_name="segment-1",
            future_trajectory=[(float(index), 2.0) for index in range(1, 21)],
        ),
        replace(
            base,
            frame_name="segment-2",
            past_trajectory=[(float(index), 2.0) for index in range(16)],
            future_trajectory=[(float(index), 4.0) for index in range(1, 21)],
        ),
        replace(
            base,
            frame_name="segment-3",
            past_trajectory=[(float(index), -2.0) for index in range(16)],
            future_trajectory=[(float(index), -4.0) for index in range(1, 21)],
        ),
    ]


if __name__ == "__main__":
    unittest.main()
