from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from minimal_shot_av.model.rfs_metric import RfsReference
from minimal_shot_av.model.v20_planner import (
    build_planner_training_tensors,
    fit_neural_system2_planner,
    load_neural_planner_frame_cache,
    save_neural_planner_frame_cache,
    trajectory_constraint_penalty,
)
from minimal_shot_av.model.wod_e2e import WodE2EPreferenceFrame


def sample_frame(frame_name: str = "segment-a-100", *, lateral: float = 0.0) -> WodE2EPreferenceFrame:
    future = [(float(index), lateral) for index in range(1, 21)]
    preferred = [(float(index), lateral + 0.1) for index in range(1, 21)]
    return WodE2EPreferenceFrame(
        frame_name=frame_name,
        past_trajectory=[(-3.0, 0.0), (-2.0, 0.0), (-1.0, 0.0), (0.0, 0.0)],
        future_trajectory=future,
        intent=1,
        init_speed_mps=4.0,
        references=[RfsReference("human", preferred, 9.2)],
        external_embedding=[0.25, -0.5, 0.75],
    )


class V20PlannerTests(unittest.TestCase):
    def test_training_tensors_keep_reference_critic_bank(self) -> None:
        tensors = build_planner_training_tensors([sample_frame()], max_reference_candidates=3)

        self.assertEqual(["segment-a-100"], tensors.frame_names)
        self.assertEqual((1, 16, 2), tensors.past.shape)
        self.assertEqual((1, 3, 40), tensors.critic_trajectories.shape)
        self.assertEqual((1, 3), tensors.critic_scores.shape)
        self.assertEqual([1.0, 1.0, 0.0], tensors.critic_mask[0].tolist())
        self.assertAlmostEqual(1.0, float(tensors.critic_scores[0, 0]))
        self.assertAlmostEqual(0.92, float(tensors.critic_scores[0, 1]), places=5)
        self.assertEqual(3, tensors.external_embedding_dimension)

    def test_planner_frame_cache_round_trips_references_and_external_embedding(self) -> None:
        frames = [sample_frame("segment-a-100"), sample_frame("segment-b-100", lateral=0.2)]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "planner_frames.jsonl"
            count = save_neural_planner_frame_cache(frames, path)
            loaded = load_neural_planner_frame_cache(path)

        self.assertEqual(2, count)
        self.assertEqual(["segment-a-100", "segment-b-100"], [frame.frame_name for frame in loaded])
        self.assertEqual([0.25, -0.5, 0.75], loaded[0].external_embedding)
        self.assertEqual(9.2, loaded[0].references[0].score)

    def test_constraint_penalty_flags_reverse_and_lateral_outliers(self) -> None:
        stable = [(float(index), 0.0) for index in range(20)]
        unsafe = [(0.0, 0.0), (-2.0, 9.5), (-4.0, 10.0)] + [(-4.0, 10.0)] * 17

        self.assertLess(trajectory_constraint_penalty(stable), trajectory_constraint_penalty(unsafe))

    def test_fit_tiny_planner_produces_scored_candidates(self) -> None:
        try:
            import torch  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("torch is not installed")
        frames = [sample_frame(f"segment-{index}-100", lateral=0.1 * index) for index in range(4)]

        model, report = fit_neural_system2_planner(
            frames,
            hidden_dim=16,
            layers=1,
            heads=4,
            modes=3,
            epochs=1,
            batch_size=2,
            learning_rate=1e-3,
            max_reference_candidates=3,
            device="cpu",
        )
        candidates = model.candidate_trajectories_for_frame(frames[0], top_k=2)

        self.assertEqual(4, report["train_rows"])
        self.assertEqual(2, len(candidates))
        self.assertTrue(all(len(candidate[1]) == 20 for candidate in candidates))
        self.assertTrue(all(isinstance(candidate[2], float) for candidate in candidates))


if __name__ == "__main__":
    unittest.main()
