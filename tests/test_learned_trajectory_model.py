from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minimal_shot_av.model.learned_trajectory_model import (
    FEATURE_SET_EXTERNAL_EMBEDDINGS,
    FEATURE_SET_TEMPORAL,
    RESIDUAL_GROUP_INTENT,
    RidgeTrajectoryModel,
    fit_ridge_trajectory_model,
    learned_candidate_payloads,
    write_learned_candidate_jsonl,
)
from minimal_shot_av.model.wod_e2e import WodE2EPreferenceFrame
from scripts.generate_wod_learned_candidates import write_candidate_jsonl


def sample_frame(frame_name: str, *, step: float = 1.0, intent: int = 1) -> WodE2EPreferenceFrame:
    return WodE2EPreferenceFrame(
        frame_name=frame_name,
        past_trajectory=[(-2.0 * step, 0.0), (-1.0 * step, 0.0), (0.0, 0.0)],
        future_trajectory=[(step * float(index), 0.0) for index in range(1, 21)],
        intent=intent,
        init_speed_mps=step * 4.0,
        references=[],
    )


class LearnedTrajectoryModelTests(unittest.TestCase):
    def test_fit_predicts_wod20_trajectory_without_text_io(self) -> None:
        model = fit_ridge_trajectory_model(
            [
                sample_frame("a", step=0.5),
                sample_frame("b", step=1.0),
                sample_frame("c", step=1.5),
                sample_frame("d", step=2.0),
            ],
            ridge=0.1,
            residual_modes=2,
        )

        trajectory = model.predict(sample_frame("test", step=1.0).past_trajectory, intent=1, init_speed_mps=4.0)

        self.assertEqual(20, len(trajectory))
        self.assertLess(abs(trajectory[-1][0] - 20.0), 1.0)
        self.assertEqual(2, len(model.residual_modes))

    def test_model_round_trips_through_json(self) -> None:
        model = fit_ridge_trajectory_model(
            [sample_frame("a", step=1.0), sample_frame("b", step=2.0)],
            ridge=1.0,
            feature_set=FEATURE_SET_TEMPORAL,
        )

        loaded = RidgeTrajectoryModel.from_dict(model.to_dict())

        self.assertEqual(model.train_rows, loaded.train_rows)
        self.assertEqual(len(model.weights), len(loaded.weights))
        self.assertEqual(FEATURE_SET_TEMPORAL, loaded.feature_set)
        self.assertGreater(len(loaded.feature_mean), 70)

    def test_temporal_summary_features_predict_without_text_io(self) -> None:
        model = fit_ridge_trajectory_model(
            [
                sample_frame("a", step=0.5),
                sample_frame("b", step=1.0),
                sample_frame("c", step=1.5),
                sample_frame("d", step=2.0),
            ],
            ridge=0.1,
            residual_modes=1,
            feature_set=FEATURE_SET_TEMPORAL,
        )

        trajectory = model.predict(sample_frame("test", step=1.0).past_trajectory, intent=1, init_speed_mps=4.0)

        self.assertEqual(20, len(trajectory))
        self.assertEqual(FEATURE_SET_TEMPORAL, model.feature_set)

    def test_external_embedding_features_condition_trajectory_model(self) -> None:
        frames = [
            replace(sample_frame("a", step=1.0), external_embedding=[0.0, 0.0]),
            replace(sample_frame("b", step=1.0), external_embedding=[1.0, 1.0]),
            replace(sample_frame("c", step=2.0), external_embedding=[1.0, 1.0]),
        ]
        model = fit_ridge_trajectory_model(
            frames,
            ridge=0.1,
            residual_modes=1,
            feature_set=FEATURE_SET_EXTERNAL_EMBEDDINGS,
        )

        candidates = model.candidate_trajectories_for_frame(frames[1], max_residual_modes=1)
        loaded = RidgeTrajectoryModel.from_dict(model.to_dict())

        self.assertEqual(FEATURE_SET_EXTERNAL_EMBEDDINGS, loaded.feature_set)
        self.assertEqual(2, loaded.external_embedding_dimension)
        self.assertEqual("ridge_scene_mean", candidates[0][0])

    def test_external_embedding_candidates_include_residual_expansions(self) -> None:
        frames = [
            replace(sample_frame("a", step=0.5, intent=1), external_embedding=[0.0, 0.0]),
            replace(sample_frame("b", step=1.0, intent=1), external_embedding=[1.0, 0.0]),
            replace(sample_frame("c", step=1.5, intent=2), external_embedding=[0.0, 1.0]),
            replace(sample_frame("d", step=2.0, intent=2), external_embedding=[1.0, 1.0]),
        ]
        model = fit_ridge_trajectory_model(
            frames,
            ridge=0.1,
            residual_modes=2,
            feature_set=FEATURE_SET_EXTERNAL_EMBEDDINGS,
            residual_grouping=RESIDUAL_GROUP_INTENT,
        )

        candidates = model.candidate_trajectories_for_frame(
            frames[1],
            max_residual_modes=2,
            include_pairwise_residuals=True,
        )

        names = [name for name, _ in candidates]
        self.assertIn("ridge_scene_residual_pc1_2_plus", names)
        self.assertIn("ridge_scene_residual_pc1_2_minus", names)
        self.assertIn("context_residual_intent_1_mean", names)

    def test_candidate_payloads_include_mean_and_learned_residual_modes(self) -> None:
        model = fit_ridge_trajectory_model(
            [sample_frame("a", step=0.5), sample_frame("b", step=1.0), sample_frame("c", step=1.5)],
            residual_modes=1,
        )

        payloads = learned_candidate_payloads([sample_frame("eval", step=1.0)], model, max_residual_modes=1)

        self.assertEqual(3, len(payloads))
        self.assertEqual(
            ["ridge_mean", "ridge_residual_pc1_plus", "ridge_residual_pc1_minus"],
            [row["candidate_name"] for row in payloads],
        )
        self.assertEqual(20, len(payloads[0]["trajectory_20wp_4hz"]))

    def test_candidate_generation_can_include_pairwise_residual_modes(self) -> None:
        model = fit_ridge_trajectory_model(
            [
                sample_frame("a", step=0.5),
                sample_frame("b", step=1.0),
                sample_frame("c", step=1.5),
                sample_frame("d", step=2.0),
            ],
            residual_modes=2,
        )

        candidates = model.candidate_trajectories(
            sample_frame("eval", step=1.0).past_trajectory,
            intent=1,
            init_speed_mps=4.0,
            max_residual_modes=2,
            include_pairwise_residuals=True,
        )

        names = [name for name, _ in candidates]
        self.assertIn("ridge_residual_pc1_2_plus", names)
        self.assertIn("ridge_residual_pc1_2_minus", names)

    def test_candidate_generation_can_include_context_residual_modes(self) -> None:
        model = fit_ridge_trajectory_model(
            [
                sample_frame("a", step=0.5, intent=1),
                sample_frame("b", step=1.0, intent=1),
                sample_frame("c", step=1.5, intent=2),
                sample_frame("d", step=2.0, intent=2),
            ],
            residual_modes=1,
            residual_grouping=RESIDUAL_GROUP_INTENT,
        )

        candidates = model.candidate_trajectories(
            sample_frame("eval", step=1.0, intent=1).past_trajectory,
            intent=1,
            init_speed_mps=4.0,
            max_residual_modes=1,
        )

        names = [name for name, _ in candidates]
        self.assertIn("context_residual_intent_1_mean", names)
        self.assertIn("context_residual_intent_1_pc1_plus", names)
        self.assertEqual(RESIDUAL_GROUP_INTENT, model.residual_grouping)

    def test_writes_learned_candidate_jsonl(self) -> None:
        model = fit_ridge_trajectory_model([sample_frame("a", step=1.0), sample_frame("b", step=2.0)], residual_modes=0)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "learned.jsonl"
            count = write_learned_candidate_jsonl([sample_frame("eval", step=1.0)], path, model)
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(1, count)
        self.assertEqual("ridge_mean", rows[0]["candidate_name"])

    def test_submission_candidate_writer_can_emit_auxiliary_temporal_candidates(self) -> None:
        frames = [sample_frame("eval", step=1.0)]
        model = fit_ridge_trajectory_model([sample_frame("a", step=1.0), sample_frame("b", step=2.0)], residual_modes=0)
        aux_model = fit_ridge_trajectory_model(
            [sample_frame("a", step=1.0), sample_frame("b", step=2.0)],
            residual_modes=0,
            feature_set=FEATURE_SET_TEMPORAL,
        )
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "combined.jsonl"
            count = write_candidate_jsonl(frames, path, model, aux_model=aux_model, max_residual_modes=0)
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(2, count)
        self.assertEqual(["ridge_mean", "temporal_ridge_mean"], [row["candidate_name"] for row in rows])


if __name__ == "__main__":
    unittest.main()
