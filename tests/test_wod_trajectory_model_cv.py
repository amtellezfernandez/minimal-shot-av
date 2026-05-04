from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "evaluate_wod_trajectory_model_cv.py"


def load_cv_module():
    spec = importlib.util.spec_from_file_location("evaluate_wod_trajectory_model_cv", SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


try:
    cv = load_cv_module()
except ModuleNotFoundError as exc:
    if exc.name != "numpy":
        raise
    cv = None

if cv is not None:
    from minimal_shot_av.model.neural_trajectory_model import (
        fit_neural_anchor_residual_trajectory_model,
        load_neural_training_frame_cache,
        save_neural_training_frame_cache,
    )
    from minimal_shot_av.model.transformer_trajectory_model import (
        TransformerTrajectoryProposalModel,
        fit_transformer_trajectory_proposal_model,
    )
    from minimal_shot_av.model.rfs_metric import RfsReference
    from minimal_shot_av.model.wod_e2e import WodCameraImage
    from minimal_shot_av.model.world_model import (
        attach_external_embedding_cache,
        scene_token_features,
        write_external_embedding_cache,
    )


def sample_frame(
    frame_name: str,
    *,
    step: float,
    camera_payload: bytes | None = None,
    intent: int = 1,
) -> object:
    return cv.WodE2EPreferenceFrame(
        frame_name=frame_name,
        past_trajectory=[(-2.0 * step, 0.0), (-1.0 * step, 0.0), (0.0, 0.0)],
        future_trajectory=[(float(index) * step, 0.0) for index in range(1, 21)],
        intent=intent,
        init_speed_mps=step * 4.0,
        references=[
            RfsReference(
                "human",
                [(float(index) * step, 0.0) for index in range(1, 21)],
                9.0,
            )
        ],
        camera_images=(
            []
            if camera_payload is None
            else [WodCameraImage(name="FRONT", jpeg=camera_payload)]
        ),
    )


class WodTrajectoryModelCvTests(unittest.TestCase):
    def test_segment_split_keeps_segment_frames_together(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")
        frames = [
            sample_frame("segment-a-100", step=1.0),
            sample_frame("segment-a-101", step=1.0),
            sample_frame("segment-b-100", step=1.5),
            sample_frame("segment-c-100", step=2.0),
        ]

        folds = cv._split_frame_names_by_segment(frames, folds=2, seed=3)

        self.assertEqual(2, len(folds))
        self.assertFalse(folds[0] & folds[1])
        segment_a_fold_count = sum("segment-a-100" in fold or "segment-a-101" in fold for fold in folds)
        self.assertEqual(1, segment_a_fold_count)

    def test_cross_validation_reports_candidate_generator_metrics(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")
        frames = [sample_frame(f"segment-{index}-100", step=0.5 + index * 0.25) for index in range(6)]

        report = cv.cross_validate_trajectory_model(
            frames,
            folds=3,
            seed=5,
            ridge=1.0,
            feature_set=cv.FEATURE_SET_TEMPORAL,
            residual_modes=1,
            scorer=cv._local_rfs_score,
        )

        self.assertEqual(6, report["frames"])
        self.assertEqual(3, report["fold_count"])
        self.assertEqual(cv.FEATURE_SET_TEMPORAL, report["feature_set"])
        self.assertEqual("base", report["kinematic_profile"])
        self.assertIn("learned_mean_candidate_mean_rfs", report)
        self.assertIn("combined_oracle_gain_vs_kinematic_oracle", report)
        self.assertIn("selected_learned_rate", report)
        self.assertIn("reference_ceiling_mean_rfs", report)
        self.assertIn("future_trajectory_mean_rfs", report)
        self.assertIn("combined_ranker_mean_normalized_rfs", report)
        self.assertIn("combined_oracle_mean_normalized_rfs", report)
        self.assertFalse(report["target_rfs_reachable_by_references"])
        self.assertFalse(report["target_rfs_reachable_by_candidates"])
        self.assertIn("slices", report)
        self.assertIn("selector_regret_buckets", report)
        self.assertTrue(report["selector_regret_buckets"])
        self.assertTrue(any(key.startswith("speed:") for key in report["slices"]))
        self.assertTrue(any(key.startswith("intent:") for key in report["slices"]))
        first_slice = next(iter(report["slices"].values()))
        self.assertIn("mean_regret", first_slice)
        self.assertIn("selected_source_rates", first_slice)
        self.assertIn("memory", first_slice["selected_source_rates"])
        self.assertNotIn("frame_diagnostics", report["folds_detail"][0])
        self.assertFalse(report["frame_diagnostics_enabled"])
        self.assertEqual(0, report["frame_diagnostics_count"])

    def test_cross_validation_can_include_frame_diagnostics(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")
        frames = [sample_frame(f"segment-{index}-100", step=0.5 + index * 0.25) for index in range(6)]

        report = cv.cross_validate_trajectory_model(
            frames,
            folds=3,
            seed=5,
            ridge=1.0,
            feature_set=cv.FEATURE_SET_TEMPORAL,
            residual_modes=1,
            include_frame_diagnostics=True,
            scorer=cv._local_rfs_score,
        )

        diagnostics = [
            diagnostic
            for fold in report["folds_detail"]
            for diagnostic in fold["frame_diagnostics"]
        ]
        self.assertEqual(6, len(diagnostics))
        self.assertTrue(report["frame_diagnostics_enabled"])
        self.assertEqual(report["frames"], report["frame_diagnostics_count"])
        self.assertEqual(report["frames"], len(diagnostics))
        first = diagnostics[0]
        self.assertIn("frame_name", first)
        self.assertIn("selected_source", first)
        self.assertIn("oracle_source", first)
        self.assertIn("regret", first)
        self.assertIn("selector_opportunity_buckets", report)
        self.assertIsInstance(report["selector_opportunity_buckets"], list)
        for diagnostic in diagnostics:
            expected_regret = float(diagnostic["oracle_score"]) - float(diagnostic["selected_score"])
            self.assertAlmostEqual(expected_regret, float(diagnostic["regret"]))
            self.assertGreaterEqual(float(diagnostic["regret"]), 0.0)

    def test_cross_validation_can_add_ego_world_model_candidates(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")
        frames = [sample_frame(f"segment-{index}-100", step=0.5 + index * 0.25) for index in range(6)]

        report = cv.cross_validate_trajectory_model(
            frames,
            folds=3,
            seed=5,
            ridge=1.0,
            feature_set=cv.FEATURE_SET_TEMPORAL,
            residual_modes=1,
            scorer=cv._local_rfs_score,
            world_model_mode=cv.FEATURE_MODE_EGO_TEMPORAL,
            world_latent_dim=2,
            world_ridge=1.0,
            world_memory_top_k=1,
            world_max_neighbor_distance=100.0,
        )

        self.assertEqual(cv.FEATURE_MODE_EGO_TEMPORAL, report["world_model"])
        self.assertEqual(100.0, report["world_max_neighbor_distance"])
        self.assertIn("world_imagined_mean_rfs", report)
        self.assertIn("world_oracle_mean_rfs", report)
        self.assertIn("selected_world_rate", report)
        self.assertIn("oracle_world_rate", report)
        first_fold = report["folds_detail"][0]
        self.assertIn("world_imagined_mean_rfs", first_fold)
        first_slice = next(iter(report["slices"].values()))
        self.assertIn("world", first_slice["selected_source_rates"])

    def test_cross_validation_can_add_neural_candidates(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")
        try:
            import torch  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("torch is not installed")
        frames = [sample_frame(f"segment-{index}-100", step=0.5 + index * 0.25) for index in range(6)]
        model, metadata = fit_neural_anchor_residual_trajectory_model(
            frames,
            anchor_count=2,
            hidden_dim=8,
            epochs=1,
            batch_size=3,
            learning_rate=1e-3,
            feature_set=cv.FEATURE_SET_BASE,
            seed=11,
            device="cpu",
        )
        self.assertEqual("cpu", metadata["device"])
        with TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "neural.json"
            model.save(model_path)
            report = cv.cross_validate_trajectory_model(
                frames,
                folds=3,
                seed=5,
                ridge=1.0,
                feature_set=cv.FEATURE_SET_TEMPORAL,
                residual_modes=1,
                scorer=cv._local_rfs_score,
                neural_candidate_model_path=model_path,
                neural_top_k=1,
                neural_residual_modes_per_anchor=0,
            )

        self.assertEqual(str(model_path), report["neural_candidate_model"])
        self.assertEqual(1, report["neural_top_k"])
        self.assertGreaterEqual(report["selected_learned_rate"], 0.0)

    def test_neural_training_frame_cache_round_trips_without_references(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")
        frames = [sample_frame(f"segment-{index}-100", step=0.5 + index * 0.25) for index in range(3)]

        with TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "frames.jsonl"
            exported_rows = save_neural_training_frame_cache(frames, cache_path)
            loaded = load_neural_training_frame_cache(cache_path)

        self.assertEqual(3, exported_rows)
        self.assertEqual(3, len(loaded))
        self.assertEqual(frames[0].frame_name, loaded[0].frame_name)
        self.assertEqual(frames[0].past_trajectory, loaded[0].past_trajectory)
        self.assertEqual(frames[0].future_trajectory, loaded[0].future_trajectory)
        self.assertEqual([], loaded[0].references)

    def test_neural_training_frame_cache_can_append_shard_windows(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")
        first_window = [sample_frame("segment-0-100", step=0.5)]
        second_window = [sample_frame("segment-1-100", step=0.75), sample_frame("segment-2-100", step=1.0)]

        with TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "frames.jsonl"
            first_rows = save_neural_training_frame_cache(first_window, cache_path)
            second_rows = save_neural_training_frame_cache(second_window, cache_path, append=True)
            loaded = load_neural_training_frame_cache(cache_path)

        self.assertEqual(1, first_rows)
        self.assertEqual(2, second_rows)
        self.assertEqual(["segment-0-100", "segment-1-100", "segment-2-100"], [frame.frame_name for frame in loaded])

    def test_transformer_model_round_trips_and_generates_candidates(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")
        try:
            import torch  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("torch is not installed")
        frames = [sample_frame(f"segment-{index}-100", step=0.5 + index * 0.25) for index in range(4)]
        model, metadata = fit_transformer_trajectory_proposal_model(
            frames,
            hidden_dim=16,
            layers=1,
            heads=2,
            modes=3,
            epochs=1,
            batch_size=2,
            learning_rate=1e-3,
            seed=13,
            device="cpu",
        )
        self.assertEqual("cpu", metadata["device"])

        with TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "transformer.pt"
            model.save(model_path)
            loaded = TransformerTrajectoryProposalModel.load(model_path)

        candidates = loaded.candidate_trajectories_for_frame(frames[0], top_k=2)

        self.assertEqual(2, len(candidates))
        self.assertTrue(candidates[0][0].startswith("transformer_mode_"))
        self.assertEqual(20, len(candidates[0][1]))

    def test_cross_validation_can_add_transformer_candidates(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")
        try:
            import torch  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("torch is not installed")
        frames = [sample_frame(f"segment-{index}-100", step=0.5 + index * 0.25) for index in range(6)]
        model, _metadata = fit_transformer_trajectory_proposal_model(
            frames,
            hidden_dim=16,
            layers=1,
            heads=2,
            modes=3,
            epochs=1,
            batch_size=3,
            learning_rate=1e-3,
            seed=17,
            device="cpu",
        )
        with TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "transformer.pt"
            model.save(model_path)
            report = cv.cross_validate_trajectory_model(
                frames,
                folds=3,
                seed=5,
                ridge=1.0,
                feature_set=cv.FEATURE_SET_TEMPORAL,
                residual_modes=1,
                scorer=cv._local_rfs_score,
                transformer_candidate_model_path=model_path,
                transformer_top_k=2,
            )

        self.assertEqual(str(model_path), report["transformer_candidate_model"])
        self.assertEqual(2, report["transformer_top_k"])
        self.assertGreaterEqual(report["selected_learned_rate"], 0.0)

    def test_reference_ceiling_can_mark_target_unreachable(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")
        frames = [sample_frame(f"segment-{index}-100", step=0.5 + index * 0.25) for index in range(6)]

        report = cv.cross_validate_trajectory_model(
            frames,
            folds=3,
            seed=5,
            ridge=1.0,
            feature_set=cv.FEATURE_SET_TEMPORAL,
            residual_modes=1,
            target_rfs=9.5,
            scorer=cv._local_rfs_score,
        )

        self.assertAlmostEqual(9.0, report["reference_ceiling_mean_rfs"])
        self.assertFalse(report["target_rfs_reachable_by_references"])
        self.assertFalse(report["target_rfs_reachable_by_candidates"])

    def test_memory_candidates_use_train_fold_frames_only(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")
        train_frames = [
            sample_frame("segment-train-a-100", step=1.0),
            sample_frame("segment-train-b-100", step=2.0),
        ]
        train_frames[0].references.append(
            RfsReference(
                "alternate",
                [(float(index), 1.0) for index in range(1, 21)],
                8.0,
            )
        )
        test_frame = sample_frame("segment-test-100", step=1.1)

        model = cv._fit_memory_candidate_model(
            train_frames,
            mode="nearest_preferences",
            top_k=2,
            feature_set=cv.FEATURE_SET_TEMPORAL,
        )
        candidates = cv._memory_candidate_trajectories(model, test_frame)
        names = [name for name, _trajectory in candidates]

        self.assertTrue(names)
        self.assertTrue(all(name.startswith("memory_neighbor_") for name in names))
        self.assertTrue(any("pref" in name for name in names))
        self.assertFalse(any(test_frame.frame_name in name for name in names))

    def test_cross_validation_can_add_scene_token_world_model_candidates(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")
        frames = [
            sample_frame(f"segment-{index}-100", step=1.0, camera_payload=bytes([index + 1]) * 8)
            for index in range(6)
        ]

        report = cv.cross_validate_trajectory_model(
            frames,
            folds=3,
            seed=5,
            ridge=1.0,
            feature_set=cv.FEATURE_SET_TEMPORAL,
            residual_modes=1,
            scorer=cv._local_rfs_score,
            world_model_mode=cv.FEATURE_MODE_SCENE_TOKENS,
            world_latent_dim=2,
            world_ridge=1.0,
            world_memory_top_k=0,
        )

        self.assertEqual(cv.FEATURE_MODE_SCENE_TOKENS, report["world_model"])
        self.assertIn("combined_with_world_ranker_mean_rfs", report)
        self.assertIn("combined_with_world_oracle_mean_rfs", report)

    def test_scene_token_cache_can_feed_scene_world_model_without_camera_payloads(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")
        camera_frames = [
            sample_frame(f"segment-{index}-100", step=1.0, camera_payload=bytes([index + 1]) * 8)
            for index in range(6)
        ]
        cache = {
            frame.frame_name: scene_token_features(frame)
            for frame in camera_frames
        }
        frames = [
            sample_frame(f"segment-{index}-100", step=1.0)
            for index in range(6)
        ]
        frames = cv.attach_scene_token_cache(frames, cache)

        report = cv.cross_validate_trajectory_model(
            frames,
            folds=3,
            seed=5,
            ridge=1.0,
            feature_set=cv.FEATURE_SET_TEMPORAL,
            residual_modes=1,
            scorer=cv._local_rfs_score,
            world_model_mode=cv.FEATURE_MODE_SCENE_TOKENS,
            world_latent_dim=2,
            world_ridge=1.0,
        )

        self.assertEqual(cv.FEATURE_MODE_SCENE_TOKENS, report["world_model"])
        self.assertTrue(all(frame.scene_tokens is not None for frame in frames))
        self.assertTrue(all(not frame.camera_images for frame in frames))

    def test_external_embedding_cache_can_feed_world_model_without_camera_payloads(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")
        frames = [
            sample_frame(f"segment-{index}-100", step=1.0 + index * 0.1)
            for index in range(6)
        ]
        frames = attach_external_embedding_cache(
            frames,
            {frame.frame_name: [float(index), float(index % 2)] for index, frame in enumerate(frames)},
        )

        report = cv.cross_validate_trajectory_model(
            frames,
            folds=3,
            seed=5,
            ridge=1.0,
            feature_set=cv.FEATURE_SET_TEMPORAL,
            residual_modes=1,
            scorer=cv._local_rfs_score,
            world_model_mode=cv.FEATURE_MODE_EXTERNAL_EMBEDDINGS,
            world_latent_dim=2,
            world_ridge=1.0,
            external_embedding_source="cosmos-test",
        )

        self.assertEqual(cv.FEATURE_MODE_EXTERNAL_EMBEDDINGS, report["world_model"])
        self.assertEqual("all", report["world_latent_source"])
        self.assertEqual("cosmos-test", report["embedding_source"])
        self.assertEqual(2, report["embedding_dimension"])
        self.assertEqual(6, report["embedding_frame_count"])
        self.assertIn("combined_with_world_ranker_mean_rfs", report)
        self.assertTrue(all(frame.external_embedding is not None for frame in frames))

    def test_cli_requires_external_embedding_cache_for_external_world_model(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")
        argv = [
            "evaluate_wod_trajectory_model_cv.py",
            "--world-model",
            cv.FEATURE_MODE_EXTERNAL_EMBEDDINGS,
            "--rfs-backend",
            "local",
        ]
        with patch.object(sys, "argv", argv), patch.object(
            cv,
            "load_preference_frames",
            return_value=[sample_frame("segment-0-100", step=1.0), sample_frame("segment-1-100", step=1.1)],
        ):
            with self.assertRaisesRegex(ValueError, "--external-embedding-cache is required"):
                cv.main()

    def test_cli_external_embeddings_preserve_image_contextual_features(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")
        frames = [
            sample_frame(f"segment-{index}-100", step=1.0 + index * 0.1, camera_payload=bytes([index + 1]) * 8)
            for index in range(6)
        ]
        with TemporaryDirectory() as tmp:
            embedding_path = Path(tmp) / "embeddings.json"
            output_path = Path(tmp) / "report.json"
            write_external_embedding_cache(
                {frame.frame_name: [float(index), float(index % 2)] for index, frame in enumerate(frames)},
                embedding_path,
                source="cosmos-test",
            )
            argv = [
                "evaluate_wod_trajectory_model_cv.py",
                "--world-model",
                cv.FEATURE_MODE_EXTERNAL_EMBEDDINGS,
                "--external-embedding-cache",
                str(embedding_path),
                "--selector-features",
                "image_contextual",
                "--rfs-backend",
                "local",
                "--folds",
                "3",
                "--residual-modes",
                "1",
                "--output",
                str(output_path),
            ]
            with patch.object(sys, "argv", argv), patch.object(cv, "load_preference_frames", return_value=frames):
                cv.main()

            self.assertTrue(all(frame.camera_images for frame in frames))
            report = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual("cosmos-test", report["embedding_source"])
        self.assertEqual("image_contextual", report["selector_features"])
        self.assertEqual(2, report["embedding_dimension"])

    def test_selector_targets_can_use_frame_delta(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")
        rows = [
            {"frame_name": "a", "rfs_score": 4.0},
            {"frame_name": "a", "rfs_score": 6.0},
            {"frame_name": "b", "rfs_score": 10.0},
        ]

        targets = cv._selector_targets(rows, "frame_delta")

        self.assertEqual([-1.0, 1.0, 0.0], targets.tolist())

    def test_selector_targets_can_use_normalized_scores(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")
        rows = [
            {"frame_name": "a", "rfs_score": 4.0, "rfs_score_normalized": 8.0},
            {"frame_name": "a", "rfs_score": 5.0, "rfs_score_normalized": 10.0},
            {"frame_name": "b", "rfs_score": 9.0, "rfs_score_normalized": 9.0},
        ]

        targets = cv._selector_targets(rows, "frame_delta_normalized")

        self.assertEqual([-1.0, 1.0, 0.0], targets.tolist())

    def test_selector_targets_can_weight_frame_delta_by_oracle_gap(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")
        rows = [
            {"frame_name": "a", "rfs_score": 4.0},
            {"frame_name": "a", "rfs_score": 6.0},
            {"frame_name": "b", "rfs_score": 0.0},
            {"frame_name": "b", "rfs_score": 10.0},
        ]

        targets = cv._selector_targets(rows, "frame_delta_oracle_weighted")

        self.assertEqual([-1.0, 1.0, -8.333333333333334, 8.333333333333334], targets.tolist())

    def test_selector_targets_can_weight_frame_delta_by_slice_risk(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")
        rows = [
            {"frame_name": "a", "rfs_score": 4.0, "features": {"intent": 1, "init_speed_mps": 3.0}},
            {"frame_name": "a", "rfs_score": 6.0, "features": {"intent": 1, "init_speed_mps": 3.0}},
            {"frame_name": "b", "rfs_score": 0.0, "features": {"intent": 3, "init_speed_mps": 12.0}},
            {"frame_name": "b", "rfs_score": 10.0, "features": {"intent": 3, "init_speed_mps": 12.0}},
        ]

        targets = cv._selector_targets(rows, "frame_delta_risk_weighted")

        self.assertEqual([-1.0, 1.0], targets[:2].tolist())
        self.assertAlmostEqual(-28.125, targets[2])
        self.assertAlmostEqual(28.125, targets[3])

    def test_selector_targets_can_use_frame_zscore(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")
        rows = [
            {"frame_name": "a", "rfs_score": 4.0},
            {"frame_name": "a", "rfs_score": 6.0},
            {"frame_name": "b", "rfs_score": 10.0},
        ]

        targets = cv._selector_targets(rows, "frame_zscore")

        self.assertEqual([-1.0, 1.0, 0.0], targets.tolist())

    def test_selector_targets_can_use_oracle_binary(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")
        rows = [
            {"frame_name": "a", "rfs_score": 4.0},
            {"frame_name": "a", "rfs_score": 6.0},
            {"frame_name": "b", "rfs_score": 10.0},
        ]

        targets = cv._selector_targets(rows, "oracle_binary")

        self.assertEqual([0.0, 1.0, 1.0], targets.tolist())

    def test_selector_targets_can_use_frame_rank(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")
        rows = [
            {"frame_name": "a", "rfs_score": 4.0},
            {"frame_name": "a", "rfs_score": 8.0},
            {"frame_name": "a", "rfs_score": 6.0},
            {"frame_name": "b", "rfs_score": 10.0},
        ]

        targets = cv._selector_targets(rows, "frame_rank")

        self.assertEqual([0.0, 1.0, 0.5, 0.0], targets.tolist())

    def test_residual_pair_blends_include_learned_residual_representatives(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")

        class FakeModel:
            def candidate_trajectories_for_frame(
                self,
                frame,
                *,
                max_residual_modes,
                include_pairwise_residuals,
            ):
                del frame, max_residual_modes, include_pairwise_residuals
                return [
                    ("ridge_mean", [(float(step), 0.0) for step in range(1, 21)]),
                    ("ridge_residual_0", [(float(step), 1.0) for step in range(1, 21)]),
                ]

        frame = sample_frame("segment-a-100", step=1.0)

        mean_rows = cv._frame_candidate_rows(
            frame,
            FakeModel(),
            cv._local_rfs_score,
            residual_modes=1,
            blend_candidates="mean_pairs",
        )
        residual_rows = cv._frame_candidate_rows(
            frame,
            FakeModel(),
            cv._local_rfs_score,
            residual_modes=1,
            blend_candidates="residual_pairs",
        )

        mean_names = {str(row["candidate_name"]) for row in mean_rows}
        residual_names = {str(row["candidate_name"]) for row in residual_rows}
        self.assertFalse(any("residual" in name and name.startswith("blend_") for name in mean_names))
        self.assertTrue(any("residual" in name and name.startswith("blend_") for name in residual_names))

    def test_pairwise_selector_prefers_better_same_frame_candidate(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")
        frame = sample_frame("segment-a-100", step=1.0)
        rows = []
        for index, (name, source, score) in enumerate(
            [
                ("constant_velocity", "kinematic", 4.0),
                ("ridge_mean", "learned", 8.0),
                ("temporal_ridge_mean", "temporal", 6.0),
            ]
        ):
            row = cv.candidate_ranker_row(
                frame=frame,
                trajectory=[(float(step), 0.0) for step in range(1, 21)],
                candidate_name=name,
                candidate_index=index,
                source=source,
            )
            row["source"] = source
            row["rfs_score"] = score
            rows.append(row)

        selector = cv._fit_selector(
            rows,
            ridge=0.01,
            target_mode="pairwise_delta",
            feature_mode="contextual",
        )

        self.assertEqual("ridge_mean", selector.select_row(rows)["candidate_name"])

    def test_listwise_selector_prefers_better_same_frame_candidate(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")
        frame = sample_frame("segment-a-100", step=1.0)
        rows = []
        for index, (name, source, score) in enumerate(
            [
                ("constant_velocity", "kinematic", 4.0),
                ("ridge_mean", "learned", 8.0),
                ("temporal_ridge_mean", "temporal", 6.0),
            ]
        ):
            row = cv.candidate_ranker_row(
                frame=frame,
                trajectory=[(float(step), 0.0) for step in range(1, 21)],
                candidate_name=name,
                candidate_index=index,
                source=source,
            )
            row["source"] = source
            row["rfs_score"] = score
            rows.append(row)

        selector = cv._fit_selector(
            rows,
            ridge=0.01,
            target_mode="frame_delta",
            feature_mode="contextual",
            model_family="listwise_softmax",
            listwise_iterations=80,
            listwise_lr=0.3,
            listwise_temperature=0.5,
        )

        self.assertEqual("ridge_mean", selector.select_row(rows)["candidate_name"])

    def test_pairwise_logistic_selector_prefers_better_same_frame_candidate(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")
        frame = sample_frame("segment-a-100", step=1.0)
        rows = []
        for index, (name, source, score) in enumerate(
            [
                ("constant_velocity", "kinematic", 4.0),
                ("ridge_mean", "learned", 8.0),
                ("temporal_ridge_mean", "temporal", 6.0),
            ]
        ):
            row = cv.candidate_ranker_row(
                frame=frame,
                trajectory=[(float(step), 0.0) for step in range(1, 21)],
                candidate_name=name,
                candidate_index=index,
                source=source,
            )
            row["source"] = source
            row["rfs_score"] = score
            rows.append(row)

        selector = cv._fit_selector(
            rows,
            ridge=0.01,
            target_mode="frame_delta",
            feature_mode="contextual",
            model_family="pairwise_logistic",
            pairwise_iterations=80,
            pairwise_lr=0.3,
            pairwise_l2=0.001,
            pairwise_max_pairs_per_frame=8,
        )

        self.assertEqual("ridge_mean", selector.select_row(rows)["candidate_name"])

    def test_family_reliability_features_use_train_fold_statistics(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")
        frame = sample_frame("segment-a-100", step=1.0)
        rows = []
        for index, (name, source, score) in enumerate(
            [
                ("constant_velocity", "kinematic", 4.0),
                ("ridge_mean", "learned", 8.0),
                ("temporal_ridge_mean", "temporal", 6.0),
            ]
        ):
            row = cv.candidate_ranker_row(
                frame=frame,
                trajectory=[(float(step), 0.0) for step in range(1, 21)],
                candidate_name=name,
                candidate_index=index,
                source=source,
            )
            row["source"] = source
            row["rfs_score"] = score
            rows.append(row)

        profile = cv._fit_family_reliability_features(rows)
        cv._apply_family_reliability_features(rows, profile)

        learned = next(row for row in rows if row["source"] == "learned")
        self.assertAlmostEqual(8.0, learned["features"]["family_reliability_mean_rfs"])
        self.assertAlmostEqual(1.0, learned["features"]["family_reliability_oracle_rate"])
        self.assertAlmostEqual(0.0, learned["features"]["family_reliability_regret_mean"])

    def test_selector_route_uses_route_specific_target(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")

        primary = object()
        alternate = object()
        frame = cv.WodE2EPreferenceFrame(
            frame_name="segment-a-100",
            past_trajectory=[(-2.0, 0.0), (-1.0, 0.0), (0.0, 0.0)],
            future_trajectory=[(float(step), 0.0) for step in range(1, 21)],
            intent=2,
            init_speed_mps=4.0,
            references=[RfsReference("human", [(float(step), 0.0) for step in range(1, 21)], 9.0)],
        )
        row = cv.candidate_ranker_row(
            frame=frame,
            trajectory=[(float(step), 0.0) for step in range(1, 21)],
            candidate_name="ridge_mean",
            candidate_index=0,
            source="learned",
        )
        selected = cv._selector_for_route(
            primary,  # type: ignore[arg-type]
            {"frame_rank": alternate},  # type: ignore[dict-item]
            {
                "router": "intent",
                "global_target": "frame_delta",
                "routes": {"intent:2": "frame_rank"},
            },
            row,
        )

        self.assertIs(alternate, selected)

    def test_kinematic_fallback_threshold_uses_train_margin(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")
        frame = sample_frame("segment-a-100", step=1.0)
        rows = []
        for index, (name, source, score) in enumerate(
            [
                ("constant_velocity", "kinematic", 8.0),
                ("ridge_mean", "learned", 4.0),
            ]
        ):
            row = cv.candidate_ranker_row(
                frame=frame,
                trajectory=[(float(step), 0.0) for step in range(1, 21)],
                candidate_name=name,
                candidate_index=index,
                source=source,
            )
            row["source"] = source
            row["rfs_score"] = score
            rows.append(row)
        selector = cv._fit_selector(rows, ridge=0.01, target_mode="absolute", feature_mode="contextual")

        threshold = cv._fit_kinematic_fallback_threshold(rows, selector, mode="train_margin")
        selected = cv._select_with_policy(
            selector,
            rows,
            source_guard=None,
            fallback_policy={
                "router": "off",
                "threshold": threshold,
                "sources": ["kinematic"],
            },
        )

        self.assertIsNotNone(threshold)
        self.assertEqual("constant_velocity", selected["candidate_name"])

    def test_speed_routed_fallback_can_learn_always_fallback_policy(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")

        class FakeSelector:
            def select_row(self, rows):
                learned_rows = [row for row in rows if row["source"] == "learned"]
                return learned_rows[0] if learned_rows else rows[0]

            def predict_row(self, row):
                return 1.0 if row["source"] == "learned" else 0.0

        rows = []
        for index, (name, source, score) in enumerate(
            [
                ("constant_velocity", "kinematic", 9.0),
                ("ridge_mean", "learned", 1.0),
            ]
        ):
            row = cv.candidate_ranker_row(
                frame=sample_frame("segment-a-100", step=1.0),
                trajectory=[(float(step), 0.0) for step in range(1, 21)],
                candidate_name=name,
                candidate_index=index,
                source=source,
            )
            row["source"] = source
            row["rfs_score"] = score
            rows.append(row)

        policy = cv._fit_fallback_policy(
            rows,
            FakeSelector(),
            mode="train_margin",
            fallback_sources=("kinematic",),
            router="speed",
            source_options=(("kinematic",),),
        )
        selected = cv._select_with_policy(
            FakeSelector(),
            rows,
            source_guard=None,
            fallback_policy=policy,
        )

        route = policy["routes"]["speed:slow"]
        self.assertEqual("always", route["decision"])
        self.assertEqual("constant_velocity", selected["candidate_name"])

    def test_scene_gate_learns_positive_scene_override(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")

        class FakeSelector:
            def select_row(self, rows):
                return max(rows, key=self.predict_row)

            def predict_row(self, row):
                return float(row["predicted_selector_score"])

        rows = []
        for frame_name, scene_pred, baseline_rfs, scene_rfs in (
            ("segment-a-100", 2.0, 4.0, 9.0),
            ("segment-b-100", -1.0, 9.0, 4.0),
        ):
            frame = sample_frame(frame_name, step=1.0)
            for index, (name, source, predicted, score) in enumerate(
                [
                    ("ridge_mean", "learned", 0.0, baseline_rfs),
                    ("scene_aux_ridge_mean", "scene", scene_pred, scene_rfs),
                ]
            ):
                row = cv.candidate_ranker_row(
                    frame=frame,
                    trajectory=[(float(step), 0.0) for step in range(1, 21)],
                    candidate_name=name,
                    candidate_index=index,
                    source=source,
                )
                row["source"] = source
                row["rfs_score"] = score
                row["predicted_selector_score"] = predicted
                rows.append(row)

        policy = cv._fit_scene_gate_policy(
            rows,
            FakeSelector(),
            mode="train_margin",
            margin=0.0,
            router="off",
            ridge=0.01,
            max_rate=0.5,
            source_calibration=None,
            fallback_policy=None,
            fallback_selectors=None,
        )
        rows_by_frame = {}
        for row in rows:
            rows_by_frame.setdefault(row["frame_name"], []).append(row)

        selected_positive = cv._apply_scene_gate(
            policy,
            FakeSelector(),
            rows_by_frame["segment-a-100"],
            rows_by_frame["segment-a-100"][0],
            source_calibration=None,
        )
        selected_negative = cv._apply_scene_gate(
            policy,
            FakeSelector(),
            rows_by_frame["segment-b-100"],
            rows_by_frame["segment-b-100"][0],
            source_calibration=None,
        )

        self.assertEqual("scene", selected_positive["source"])
        self.assertEqual("learned", selected_negative["source"])

    def test_scene_gate_off_preserves_default_selection(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")
        frames = [sample_frame(f"segment-{index}-100", step=0.5 + index * 0.25) for index in range(6)]

        default_report = cv.cross_validate_trajectory_model(
            frames,
            folds=3,
            seed=5,
            ridge=1.0,
            feature_set=cv.FEATURE_SET_TEMPORAL,
            residual_modes=1,
            scorer=cv._local_rfs_score,
        )
        off_report = cv.cross_validate_trajectory_model(
            frames,
            folds=3,
            seed=5,
            ridge=1.0,
            feature_set=cv.FEATURE_SET_TEMPORAL,
            residual_modes=1,
            scene_gate="off",
            scorer=cv._local_rfs_score,
        )

        self.assertEqual(default_report["combined_ranker_mean_rfs"], off_report["combined_ranker_mean_rfs"])
        self.assertFalse(off_report["scene_gate_enabled"])

    def test_independent_source_gate_fits_source_specific_models(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")

        class FakeSelector:
            def select_row(self, rows):
                return max(rows, key=self.predict_row)

            def predict_row(self, row):
                return float(row["predicted_selector_score"])

        rows = []
        frame = sample_frame("segment-a-100", step=1.0)
        for index, (name, source, predicted, score) in enumerate(
            [
                ("ridge_mean", "learned", 1.0, 4.0),
                ("constant_velocity", "kinematic", 0.5, 9.0),
                ("memory_neighbor_1_future", "memory", 0.5, 1.0),
            ]
        ):
            row = cv.candidate_ranker_row(
                frame=frame,
                trajectory=[(float(step), 0.0) for step in range(1, 21)],
                candidate_name=name,
                candidate_index=index,
                source=source,
            )
            row["source"] = source
            row["rfs_score"] = score
            row["predicted_selector_score"] = predicted
            rows.append(row)

        policy = cv._fit_source_gate_policy(
            rows,
            FakeSelector(),
            mode="independent_train_margin",
            sources=("kinematic", "memory"),
            candidate_prefixes=(),
            route_allowlist=(),
            margin=0.0,
            router="off",
            ridge=0.01,
            max_rate=1.0,
            min_precision=0.0,
            min_route_observations=0,
            min_route_positives=0,
            source_calibration=None,
            fallback_policy=None,
            fallback_selectors=None,
        )
        selected = cv._apply_source_gate(
            policy,
            FakeSelector(),
            rows,
            rows[0],
            source_calibration=None,
        )

        self.assertEqual("independent_train_margin", policy["mode"])
        self.assertEqual({"kinematic", "memory"}, set(policy["source_models"]))
        self.assertEqual("constant_velocity", selected["candidate_name"])

    def test_source_gate_min_precision_can_reject_noisy_thresholds(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")

        observations = [
            {"baseline_score": 5.0, "scene_score": 7.0},
            {"baseline_score": 5.0, "scene_score": 4.0},
        ]
        predicted = cv.np.asarray([1.0, 1.0], dtype=cv.np.float64)

        permissive = cv._fit_scene_gate_threshold(
            observations,
            predicted,
            margin=0.0,
            max_rate=1.0,
            min_precision=0.0,
        )
        strict = cv._fit_scene_gate_threshold(
            observations,
            predicted,
            margin=0.0,
            max_rate=1.0,
            min_precision=0.75,
        )

        self.assertIsNotNone(permissive)
        self.assertIsNone(strict)

    def test_source_gate_threshold_can_require_route_support(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")

        observations = [
            {"baseline_score": 5.0, "scene_score": 7.0},
            {"baseline_score": 5.0, "scene_score": 7.0},
        ]
        predicted = cv.np.asarray([1.0, 1.0], dtype=cv.np.float64)

        supported = cv._fit_scene_gate_threshold(
            observations,
            predicted,
            margin=0.0,
            max_rate=1.0,
            min_observations=2,
            min_positive_overrides=2,
        )
        unsupported = cv._fit_scene_gate_threshold(
            observations,
            predicted,
            margin=0.0,
            max_rate=1.0,
            min_observations=3,
            min_positive_overrides=2,
        )
        too_few_positives = cv._fit_scene_gate_threshold(
            observations,
            predicted,
            margin=0.0,
            max_rate=1.0,
            min_observations=2,
            min_positive_overrides=3,
        )

        self.assertIsNotNone(supported)
        self.assertIsNone(unsupported)
        self.assertIsNone(too_few_positives)

    def test_source_veto_can_replace_bad_selected_source(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")

        class FakeSelector:
            def select_row(self, rows):
                return max(rows, key=self.predict_row)

            def predict_row(self, row):
                return float(row["predicted_selector_score"])

        rows = []
        frame = sample_frame("segment-a-100", step=1.0)
        for index, (name, source, predicted, score) in enumerate(
            [
                ("temporal_ridge_mean", "temporal", 2.0, 3.0),
                ("constant_velocity", "kinematic", 1.0, 9.0),
            ]
        ):
            row = cv.candidate_ranker_row(
                frame=frame,
                trajectory=[(float(step), 0.0) for step in range(1, 21)],
                candidate_name=name,
                candidate_index=index,
                source=source,
            )
            row["source"] = source
            row["rfs_score"] = score
            row["predicted_selector_score"] = predicted
            rows.append(row)

        policy = cv._fit_source_veto_policy(
            rows,
            FakeSelector(),
            mode="train_margin",
            sources=("temporal",),
            fallback_sources=("kinematic",),
            router="off",
            ridge=0.01,
            max_rate=1.0,
            min_precision=0.0,
            min_route_observations=0,
            min_route_positives=0,
            source_calibration=None,
            fallback_policy=None,
            fallback_selectors=None,
            scene_gate_policy=None,
            source_gate_policy=None,
            source_gate_selectors=None,
        )
        selected = cv._apply_source_veto(
            policy,
            FakeSelector(),
            rows,
            rows[0],
            source_calibration=None,
        )

        self.assertEqual("train_margin", policy["mode"])
        self.assertEqual("constant_velocity", selected["candidate_name"])

    def test_source_calibration_can_offset_overconfident_source(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")

        class FakeSelector:
            def select_row(self, rows):
                return max(rows, key=self.predict_row)

            def predict_row(self, row):
                return 1.0 if row["source"] == "temporal" else 0.0

        frame = sample_frame("segment-a-100", step=1.0)
        rows = []
        for index, (name, source, score) in enumerate(
            [
                ("constant_velocity", "kinematic", 9.0),
                ("temporal_ridge_mean", "temporal", 1.0),
            ]
        ):
            row = cv.candidate_ranker_row(
                frame=frame,
                trajectory=[(float(step), 0.0) for step in range(1, 21)],
                candidate_name=name,
                candidate_index=index,
                source=source,
            )
            row["source"] = source
            row["rfs_score"] = score
            rows.append(row)

        calibration = cv._fit_source_calibration(rows, mode="speed")
        selected = cv._select_with_policy(
            FakeSelector(),
            rows,
            source_guard=None,
            fallback_policy=None,
            source_calibration=calibration,
        )

        self.assertEqual("constant_velocity", selected["candidate_name"])

    def test_source_calibration_scale_shrinks_offsets(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")

        frame = sample_frame("segment-source-scale-100", step=1.0)
        rows = []
        for index, (name, source, score) in enumerate(
            [
                ("constant_velocity", "kinematic", 9.0),
                ("temporal_ridge_mean", "temporal", 1.0),
            ]
        ):
            row = cv.candidate_ranker_row(
                frame=frame,
                trajectory=[(float(step), 0.0) for step in range(1, 21)],
                candidate_name=name,
                candidate_index=index,
                source=source,
            )
            row["source"] = source
            row["rfs_score"] = score
            rows.append(row)

        full = cv._fit_source_calibration(rows, mode="speed")
        shrunk = cv._fit_source_calibration(rows, mode="speed", scale=0.25)
        route = cv._fallback_router_key(rows[0], "speed")

        self.assertAlmostEqual(shrunk[route]["kinematic"], full[route]["kinematic"] * 0.25)
        self.assertAlmostEqual(shrunk[route]["temporal"], full[route]["temporal"] * 0.25)

        with self.assertRaisesRegex(ValueError, "non-negative"):
            cv._fit_source_calibration(rows, mode="speed", scale=-0.1)

    def test_source_policy_selects_oracle_source_before_candidate(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")

        class FakeSelector:
            def predict_row(self, row):
                return float(row["selector_score"])

        rows = []
        for frame_index in range(3):
            frame = sample_frame(f"segment-{frame_index}-100", step=1.0)
            for index, (name, source, selector_score, rfs_score) in enumerate(
                [
                    ("temporal_ridge_mean", "temporal", 10.0, 4.0),
                    ("constant_velocity", "kinematic", 1.0, 9.0),
                ]
            ):
                row = cv.candidate_ranker_row(
                    frame=frame,
                    trajectory=[(float(step), 0.0) for step in range(1, 21)],
                    candidate_name=name,
                    candidate_index=index,
                    source=source,
                )
                row["source"] = source
                row["selector_score"] = selector_score
                row["rfs_score"] = rfs_score
                rows.append(row)

        policy = cv._fit_source_policy(rows, mode="oracle_source")
        selected = cv._select_with_policy(
            FakeSelector(),
            rows[:2],
            source_guard=None,
            fallback_policy=None,
            source_policy=policy,
        )

        self.assertEqual("constant_velocity", selected["candidate_name"])

    def test_source_score_policy_routes_by_intent_speed_best_source_score(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")

        class FakeSelector:
            def predict_row(self, row):
                return float(row["selector_score"])

        rows = []
        specs = [
            ("segment-slow-0-100", 1, 2.0, "kinematic", 9.0),
            ("segment-slow-1-100", 1, 2.0, "kinematic", 8.0),
            ("segment-fast-0-100", 1, 12.0, "temporal", 9.0),
            ("segment-fast-1-100", 1, 12.0, "temporal", 8.0),
        ]
        for frame_name, intent, speed, winning_source, winning_score in specs:
            frame = sample_frame(frame_name, step=speed / 4.0, intent=intent)
            for index, source in enumerate(("kinematic", "temporal")):
                name = "constant_velocity" if source == "kinematic" else "temporal_ridge_mean"
                score = winning_score if source == winning_source else 3.0
                row = cv.candidate_ranker_row(
                    frame=frame,
                    trajectory=[(float(step), 0.0) for step in range(1, 21)],
                    candidate_name=name,
                    candidate_index=index,
                    source=source,
                )
                row["source"] = source
                row["selector_score"] = 10.0 if source == "temporal" else 1.0
                row["rfs_score"] = score
                rows.append(row)

        policy = cv._fit_source_policy(rows, mode="source_score_intent_speed")
        slow_selected = cv._select_with_policy(
            FakeSelector(),
            rows[:2],
            source_guard=None,
            fallback_policy=None,
            source_policy=policy,
        )
        fast_selected = cv._select_with_policy(
            FakeSelector(),
            rows[4:6],
            source_guard=None,
            fallback_policy=None,
            source_policy=policy,
        )

        self.assertEqual("kinematic", slow_selected["source"])
        self.assertEqual("temporal", fast_selected["source"])

    def test_zero_shot_geometry_filter_keeps_kinematic_and_drops_implausible_learned(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")

        frame = sample_frame("segment-geometry-0-100", step=1.0)
        kinematic = cv.candidate_ranker_row(
            frame=frame,
            trajectory=[(float(step), 0.0) for step in range(1, 21)],
            candidate_name="constant_velocity",
            candidate_index=0,
            source="kinematic",
        )
        implausible = cv.candidate_ranker_row(
            frame=frame,
            trajectory=[(float(step) * 2.0, float(step) * 1.5) for step in range(1, 21)],
            candidate_name="ridge_residual_bad_geometry",
            candidate_index=1,
            source="learned",
        )

        kept = cv._zero_shot_geometry_filter_rows([kinematic, implausible], mode="reactive")

        self.assertEqual([kinematic], kept)

    def test_affordance_geometry_filter_rejects_world_inconsistent_candidate(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")

        frame = sample_frame("segment-affordance-0-100", step=2.5)
        kinematic = cv.candidate_ranker_row(
            frame=frame,
            trajectory=[(float(step) * 2.5, 0.0) for step in range(1, 21)],
            candidate_name="constant_velocity",
            candidate_index=0,
            source="kinematic",
        )
        wrong_way_turn = cv.candidate_ranker_row(
            frame=frame,
            trajectory=[(float(step) * 0.2, float(step) * 1.2) for step in range(1, 21)],
            candidate_name="scene_bad_affordance",
            candidate_index=1,
            source="scene",
        )

        kept = cv._zero_shot_geometry_filter_rows([kinematic, wrong_way_turn], mode="affordance")

        self.assertEqual([kinematic], kept)

    def test_affordance_geometry_filter_falls_back_if_all_candidates_rejected(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")

        frame = sample_frame("segment-affordance-fallback-0-100", step=2.5)
        implausible = cv.candidate_ranker_row(
            frame=frame,
            trajectory=[(-float(step), float(step) * 2.0) for step in range(1, 21)],
            candidate_name="learned_reverse_arc",
            candidate_index=0,
            source="learned",
        )

        kept = cv._zero_shot_geometry_filter_rows([implausible], mode="affordance")

        self.assertEqual([implausible], kept)

    def test_candidate_features_include_world_geometry_priors(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")

        frame = sample_frame("segment-world-geometry-0-100", step=1.0)
        row = cv.candidate_ranker_row(
            frame=frame,
            trajectory=[(float(step), 0.0) for step in range(1, 21)],
            candidate_name="constant_velocity",
            candidate_index=0,
            source="kinematic",
        )
        features = row["features"]

        self.assertIn("expected_progress_5s", features)
        self.assertIn("progress_ratio_5s", features)
        self.assertIn("monotonic_forward_rate", features)
        self.assertIn("curvature_per_meter", features)
        self.assertAlmostEqual(float(features["monotonic_forward_rate"]), 1.0)
        self.assertAlmostEqual(float(features["curvature_per_meter"]), 0.0)

    def test_family_calibration_falls_back_to_source_offset_for_sparse_bucket(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")

        frame = sample_frame("segment-a-100", step=1.0)
        rows = []
        for index, (name, source, score) in enumerate(
            [
                ("constant_velocity", "kinematic", 9.0),
                ("temporal_ridge_mean", "temporal", 1.0),
            ]
        ):
            row = cv.candidate_ranker_row(
                frame=frame,
                trajectory=[(float(step), 0.0) for step in range(1, 21)],
                candidate_name=name,
                candidate_index=index,
                source=source,
            )
            row["source"] = source
            row["rfs_score"] = score
            rows.append(row)

        calibration = cv._fit_family_calibration(rows, mode="speed_source_family", min_count=3)

        self.assertEqual({}, calibration["offsets"])
        self.assertGreater(calibration["source_offsets"]["kinematic"], 0.0)
        self.assertLess(calibration["source_offsets"]["temporal"], 0.0)

    def test_selector_audit_export_writes_jsonl_diagnostics(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "audit.jsonl"
            report = {
                "folds_detail": [
                    {
                        "fold_index": 2,
                        "frame_diagnostics": [
                            {
                                "frame_name": "segment-a-100",
                                "selected_source": "temporal",
                                "oracle_source": "kinematic",
                            }
                        ],
                    }
                ]
            }

            cv._write_selector_audit_rows(report, path)

            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(1, len(rows))
        self.assertEqual(2, rows[0]["fold_index"])
        self.assertEqual("segment-a-100", rows[0]["frame_name"])

    def test_split_sources_rejects_empty_fallback_source_list(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")

        self.assertEqual(["kinematic", "temporal"], cv._split_sources("kinematic, temporal"))
        self.assertEqual(
            [("kinematic",), ("kinematic", "temporal")],
            cv._split_source_options("kinematic;kinematic, temporal"),
        )
        with self.assertRaisesRegex(ValueError, "at least one fallback source"):
            cv._split_sources(" , ")

    def test_fallback_router_key_can_use_speed_and_intent_speed(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")
        row = cv.candidate_ranker_row(
            frame=sample_frame("segment-a-100", step=1.0),
            trajectory=[(float(step), 0.0) for step in range(1, 21)],
            candidate_name="constant_velocity",
            candidate_index=0,
            source="kinematic",
        )

        self.assertEqual("speed:slow", cv._fallback_router_key(row, "speed"))
        self.assertEqual("intent:1|speed:slow", cv._fallback_router_key(row, "intent_speed"))

    def test_fallback_router_key_can_use_fine_fast_speed(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")
        row = cv.candidate_ranker_row(
            frame=sample_frame("segment-a-100", step=5.0),
            trajectory=[(float(step), 0.0) for step in range(1, 21)],
            candidate_name="constant_velocity",
            candidate_index=0,
            source="kinematic",
        )

        self.assertEqual("speed:fast_high", cv._fallback_router_key(row, "speed_fine"))

    def test_selector_numeric_features_can_include_squares(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")

        features = cv._selector_numeric_features("squared")

        self.assertIn("endpoint_distance", features)
        self.assertIn("sq_endpoint_distance", features)
        self.assertIn("source_learned", features)
        self.assertIn("sq_source_learned", features)

    def test_selector_numeric_features_can_include_context_interactions(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")

        features = cv._selector_numeric_features("contextual")

        self.assertIn("source_temporal", features)
        self.assertIn("speed_bin_slow", features)
        self.assertIn("intent_2", features)
        self.assertIn("source_temporal_x_speed_bin_slow", features)
        self.assertIn("source_kinematic_x_intent_2", features)

    def test_selector_numeric_features_can_include_world_confidence(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")

        features = cv._selector_numeric_features("world_contextual")

        self.assertIn("source_world", features)
        self.assertIn("world_nearest_distance_log", features)
        self.assertIn("source_world_x_world_nearest_distance_log", features)

    def test_selector_numeric_features_can_include_geometry_priors(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")

        features = cv._selector_numeric_features("geometry_contextual")

        self.assertIn("expected_progress_5s", features)
        self.assertIn("progress_ratio_5s", features)
        self.assertIn("monotonic_forward_rate", features)
        self.assertIn("curvature_per_meter", features)

if __name__ == "__main__":
    unittest.main()
