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


def _safety_row(
    candidate_name: str,
    source: str,
    *,
    selector_score: float,
    rfs: float,
    reverse_distance: float = 0.0,
    monotonic_forward_rate: float = 1.0,
    max_abs_accel_mps2: float = 0.0,
    x_5s: float = 40.0,
    lateral_to_progress_ratio: float = 0.0,
    curvature_per_meter: float = 0.0,
) -> dict[str, object]:
    return {
        "frame_name": "segment-safety-100",
        "candidate_name": candidate_name,
        "candidate_index": 0,
        "source": source,
        "rfs_score": float(rfs),
        "features": {
            "selector_score": float(selector_score),
            "candidate_family": candidate_name,
            "intent": 1.0,
            "init_speed_mps": 8.0,
            "max_speed_mps": 8.0,
            "final_speed_mps": 8.0,
            "max_abs_accel_mps2": float(max_abs_accel_mps2),
            "mean_abs_accel_mps2": float(max_abs_accel_mps2) * 0.5,
            "expected_progress_5s": 40.0,
            "x_1s": float(x_5s) * 0.2,
            "x_3s": float(x_5s) * 0.6,
            "x_5s": float(x_5s),
            "progress_ratio_5s": float(x_5s) / 40.0,
            "stop_distance_error": 0.0,
            "reverse_distance": float(reverse_distance),
            "monotonic_forward_rate": float(monotonic_forward_rate),
            "lateral_to_progress_ratio": float(lateral_to_progress_ratio),
            "curvature_per_meter": float(curvature_per_meter),
            "final_speed_ratio": 1.0,
        },
    }


class WodTrajectoryModelCvTests(unittest.TestCase):
    def test_unseen_scenario_uses_fold_calibrated_thresholds(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")
        profile = {
            "thresholds": {
                "family_count_log_unseen": 1.2,
                "source_count_log_unseen": 1.8,
            }
        }
        self.assertTrue(
            cv._is_unseen_scenario(
                {"family_reliability_count_log": 0.7, "source_reliability_count_log": 2.2},
                reliability_profile=profile,
            )
        )
        self.assertFalse(
            cv._is_unseen_scenario(
                {"family_reliability_count_log": 1.4, "source_reliability_count_log": 2.1},
                reliability_profile=profile,
            )
        )

    def test_uncertainty_bin_uses_support_and_ambiguity_not_model_specific_disagreement(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")
        rows = [
            {
                "rfs_score": 8.0,
                "features": {
                    "family_reliability_count_log": 2.5,
                    "source_reliability_count_log": 3.0,
                    "retrieval_latent_disagreement": 99.0,
                },
            },
            {
                "rfs_score": 7.2,
                "features": {
                    "family_reliability_count_log": 2.4,
                    "source_reliability_count_log": 3.1,
                    "retrieval_latent_disagreement": -99.0,
                },
            },
        ]
        profile = {
            "thresholds": {
                "family_count_log_low_support": 1.0,
                "source_count_log_low_support": 1.0,
            }
        }
        self.assertEqual("medium", cv._uncertainty_bin(rows, 0.8, reliability_profile=profile))
        self.assertEqual("high", cv._uncertainty_bin(rows, 0.2, reliability_profile=profile))

    def test_failure_case_observations_require_both_support_channels(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")
        frame = sample_frame("segment-failure-100", step=1.0)
        selected = {
            "rfs_score": 6.0,
            "source": "learned",
            "candidate_name": "selected",
            "features": {
                "retrieval_support_score": 0.9,
                "latent_feasibility_score": 0.1,
                "retrieval_support_present": 1.0,
                "latent_feasibility_present": 0.0,
            },
        }
        oracle = {
            "rfs_score": 8.5,
            "source": "kinematic",
            "candidate_name": "oracle",
            "features": {
                "retrieval_support_score": 0.1,
                "latent_feasibility_score": 0.9,
                "retrieval_support_present": 1.0,
                "latent_feasibility_present": 1.0,
            },
        }
        accumulators: dict[str, dict[str, object]] = {}
        cv._add_failure_case_observations(accumulators, frame, selected, oracle, [selected, oracle])
        self.assertEqual({}, accumulators)

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

    def test_source_gate_baseline_can_hold_out_experimental_sources(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")
        rows = [
            {"source": "kinematic", "candidate_name": "constant_velocity"},
            {"source": "temporal", "candidate_name": "temporal_ridge_mean"},
            {"source": "internnav", "candidate_name": "internnav_s2_waypoint_progress"},
        ]
        policy = {
            "mode": "train_margin",
            "sources": ["kinematic", "internnav"],
            "baseline_deny_sources": ["internnav"],
        }

        baseline_rows = cv._source_gate_baseline_rows(policy, rows)

        self.assertEqual(["kinematic", "temporal"], [row["source"] for row in baseline_rows])

    def test_frame_relative_features_compare_same_frame_candidates(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")
        rows = []
        for index, progress in enumerate((5.0, 10.0, 20.0)):
            rows.append(
                {
                    "candidate_index": index,
                    "features": {
                        "endpoint_distance": progress,
                        "total_distance": progress,
                        "forward_progress": progress,
                        "final_lateral_abs": float(index),
                        "signed_lateral_5s": float(index),
                        "mean_abs_heading_change": 0.1 * index,
                        "max_abs_accel_mps2": 0.5 * index,
                        "progress_error_abs_5s": abs(12.0 - progress),
                        "lateral_to_progress_ratio": 0.1 * index,
                        "curvature_per_meter": 0.01 * index,
                        "x_1s": progress * 0.2,
                        "y_1s": float(index),
                        "x_2s": progress * 0.4,
                        "y_2s": float(index),
                        "x_3s": progress * 0.6,
                        "y_3s": float(index),
                        "x_4s": progress * 0.8,
                        "y_4s": float(index),
                        "x_5s": progress,
                        "y_5s": float(index),
                    },
                }
            )

        cv._add_frame_relative_features(rows)

        self.assertLess(
            rows[0]["features"]["endpoint_distance_frame_rank"],
            rows[2]["features"]["endpoint_distance_frame_rank"],
        )
        self.assertEqual(0.5, rows[1]["features"]["endpoint_distance_frame_rank"])
        self.assertEqual(1.0, rows[2]["features"]["candidate_index_fraction"])
        self.assertGreater(rows[2]["features"]["waypoint_mean_l2_to_frame_median"], 0.0)

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
        self.assertIn("long_tail_slices", report)
        self.assertIn("selector_regret_buckets", report)
        self.assertIn("failure_case_buckets", report)
        self.assertTrue(report["selector_regret_buckets"])
        self.assertTrue(any(key.startswith("speed:") for key in report["slices"]))
        self.assertTrue(any(key.startswith("intent:") for key in report["slices"]))
        self.assertTrue(any(key.startswith("uncertainty:") for key in report["long_tail_slices"]))
        self.assertTrue(any(key.startswith("ambiguity:") for key in report["long_tail_slices"]))
        self.assertTrue(any(key.startswith("unseen:") for key in report["long_tail_slices"]))
        self.assertTrue(any(key.startswith("rare_behavior:") for key in report["long_tail_slices"]))
        first_slice = next(iter(report["slices"].values()))
        first_long_tail_slice = next(iter(report["long_tail_slices"].values()))
        self.assertIn("mean_regret", first_slice)
        self.assertIn("selected_source_rates", first_slice)
        self.assertIn("memory", first_slice["selected_source_rates"])
        self.assertIn("mean_regret", first_long_tail_slice)
        self.assertIsInstance(report["failure_case_buckets"], list)
        self.assertNotIn("frame_diagnostics", report["folds_detail"][0])
        self.assertFalse(report["frame_diagnostics_enabled"])
        self.assertEqual(0, report["frame_diagnostics_count"])

    def test_cross_validation_can_add_external_internvla_candidates(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")
        frames = [sample_frame(f"segment-{index}-100", step=0.5 + index * 0.25) for index in range(6)]
        external_candidates = {
            frame.frame_name: [
                cv.CandidateRecord(
                    frame_name=frame.frame_name,
                    trajectory=frame.future_trajectory,
                    source="internvla",
                    candidate_name="internvla_s2_pixel_goal",
                )
            ]
            for frame in frames
        }

        report = cv.cross_validate_trajectory_model(
            frames,
            folds=3,
            seed=5,
            ridge=1.0,
            feature_set=cv.FEATURE_SET_TEMPORAL,
            residual_modes=1,
            external_candidate_groups=external_candidates,
            scorer=cv._local_rfs_score,
        )

        self.assertEqual(6, report["external_candidate_frame_count"])
        self.assertEqual(6, report["external_candidate_count"])
        self.assertIn("selected_internvla_rate", report)
        self.assertIn("oracle_internvla_rate", report)
        first_slice = next(iter(report["slices"].values()))
        self.assertIn("internvla", first_slice["selected_source_rates"])

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
        self.assertIn("long_tail_slices", report)
        self.assertIn("failure_case_buckets", report)
        self.assertIsInstance(report["selector_opportunity_buckets"], list)
        self.assertIsInstance(report["failure_case_buckets"], list)
        for diagnostic in diagnostics:
            expected_regret = float(diagnostic["oracle_score"]) - float(diagnostic["selected_score"])
            self.assertAlmostEqual(expected_regret, float(diagnostic["regret"]))
            self.assertGreaterEqual(float(diagnostic["regret"]), 0.0)

    def test_load_frame_cache_accepts_legacy_schema_less_payload(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")
        frame = sample_frame("segment-cache-100", step=1.0)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "frames.json"
            path.write_text(
                json.dumps({"frames": [cv._frame_to_payload(frame)]}),
                encoding="utf-8",
            )

            loaded = cv._load_frame_cache(path)

        self.assertEqual([frame.frame_name], [loaded_frame.frame_name for loaded_frame in loaded])
        self.assertEqual(frame.future_trajectory, loaded[0].future_trajectory)

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

    def test_cross_validation_can_add_latent_predictive_world_prior_features(self) -> None:
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
            selector_features="world_prior_contextual",
            world_prior=cv.WORLD_PRIOR_LATENT_PREDICTIVE,
            world_prior_latent_dim=2,
            world_prior_ridge=1.0,
        )

        self.assertEqual(cv.WORLD_PRIOR_LATENT_PREDICTIVE, report["world_prior"])
        self.assertEqual(2, report["world_prior_latent_dim"])
        self.assertEqual(cv.FEATURE_MODE_EGO_TEMPORAL, report["world_prior_feature_mode"])
        self.assertEqual("JEPA/V-JEPA-style context-to-target latent prediction", report["world_prior_inspiration"])
        self.assertEqual("off", report["world_prior_mask_mode"])

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
            cv._target,
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
            with patch.object(sys, "argv", argv), patch.object(
                cv._target, "load_preference_frames", return_value=frames
            ):
                cv.main()

            self.assertTrue(all(frame.camera_images for frame in frames))
            report = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual("cosmos-test", report["embedding_source"])
        self.assertEqual("image_contextual", report["selector_features"])
        self.assertEqual(2, report["embedding_dimension"])

    def test_cli_can_allow_missing_external_embeddings_for_selector_context(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")
        frames = [
            sample_frame(f"segment-{index}-100", step=1.0 + index * 0.1)
            for index in range(6)
        ]
        with TemporaryDirectory() as tmp:
            embedding_path = Path(tmp) / "embeddings.json"
            output_path = Path(tmp) / "report.json"
            write_external_embedding_cache(
                {frame.frame_name: [float(index), float(index % 2)] for index, frame in enumerate(frames[:-1])},
                embedding_path,
                source="cosmos-partial-test",
            )
            argv = [
                "evaluate_wod_trajectory_model_cv.py",
                "--external-embedding-cache",
                str(embedding_path),
                "--allow-missing-external-embeddings",
                "--selector-features",
                "external_contextual",
                "--rfs-backend",
                "local",
                "--folds",
                "3",
                "--residual-modes",
                "1",
                "--output",
                str(output_path),
            ]
            with patch.object(sys, "argv", argv), patch.object(
                cv._target, "load_preference_frames", return_value=frames
            ):
                cv.main()

            report = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual("cosmos-partial-test", report["embedding_source"])
        self.assertEqual("external_contextual", report["selector_features"])
        self.assertEqual(2, report["embedding_dimension"])
        self.assertEqual(5, report["embedding_frame_count"])

    def test_cli_rejects_missing_external_embeddings_for_external_world_model(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")
        with TemporaryDirectory() as tmp:
            embedding_path = Path(tmp) / "embeddings.json"
            write_external_embedding_cache({"segment-0-100": [1.0, 2.0]}, embedding_path, source="partial")
            argv = [
                "evaluate_wod_trajectory_model_cv.py",
                "--world-model",
                cv.FEATURE_MODE_EXTERNAL_EMBEDDINGS,
                "--external-embedding-cache",
                str(embedding_path),
                "--allow-missing-external-embeddings",
                "--rfs-backend",
                "local",
            ]
            with patch.object(sys, "argv", argv), patch.object(
                cv._target,
                "load_preference_frames",
                return_value=[sample_frame("segment-0-100", step=1.0), sample_frame("segment-1-100", step=1.1)],
            ):
                with self.assertRaisesRegex(ValueError, "cannot be used"):
                    cv.main()

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

    def test_raw_feature_matrix_matches_row_features(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")
        frame = sample_frame("segment-a-100", step=1.0)
        rows = []
        for index, (name, source) in enumerate(
            [
                ("constant_velocity", "kinematic"),
                ("ridge_mean", "learned"),
            ]
        ):
            row = cv.candidate_ranker_row(
                frame=frame,
                trajectory=[(float(step), 0.1 * index) for step in range(1, 21)],
                candidate_name=name,
                candidate_index=index,
                source=source,
            )
            row["source"] = source
            rows.append(row)
        numeric_features = cv._selector_numeric_features("contextual")
        candidate_names = ["constant_velocity"]
        candidate_families = sorted({str(row["features"]["candidate_family"]) for row in rows})

        matrix = cv._raw_feature_matrix(rows, numeric_features, candidate_names, candidate_families)
        expected = cv.np.asarray(
            [
                cv.raw_features(row, numeric_features, candidate_names, candidate_families)
                for row in rows
            ],
            dtype=cv.np.float64,
        )

        self.assertTrue(cv.np.allclose(expected, matrix))

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

    def test_pairwise_tournament_selector_prefers_pairwise_winner(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")
        frame = sample_frame("segment-a-100", step=1.0)
        rows = []
        for index, (name, source, score, lateral) in enumerate(
            [
                ("constant_velocity", "kinematic", 4.0, 0.0),
                ("ridge_mean", "learned", 8.0, 0.2),
                ("temporal_ridge_mean", "temporal", 6.0, -0.2),
            ]
        ):
            row = cv.candidate_ranker_row(
                frame=frame,
                trajectory=[(float(step), lateral) for step in range(1, 21)],
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
            model_family="pairwise_tournament",
            pairwise_max_pairs_per_frame=8,
        )

        self.assertEqual("ridge_mean", selector.select_row(rows)["candidate_name"])
        self.assertEqual(
            "ridge_mean",
            cv._select_by_calibrated_score(selector, rows, None)["candidate_name"],
        )

    def test_direct_policy_learns_candidate_gain_against_baseline(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")
        selector = cv.WodPreferenceRanker(
            numeric_features=[],
            candidate_names=["constant_velocity", "ridge_mean"],
            candidate_families=[],
            feature_mean=[0.0, 0.0],
            feature_scale=[1.0, 1.0],
            weights=[1.0, 0.0],
            bias=0.0,
        )
        train_rows = []
        for frame_index in range(4):
            frame = sample_frame(f"segment-{frame_index}-100", step=1.0 + 0.1 * frame_index)
            for index, (name, source, score, lateral) in enumerate(
                [
                    ("constant_velocity", "kinematic", 4.0, 0.0),
                    ("ridge_mean", "learned", 8.0, 0.2),
                ]
            ):
                row = cv.candidate_ranker_row(
                    frame=frame,
                    trajectory=[(float(step), lateral) for step in range(1, 21)],
                    candidate_name=name,
                    candidate_index=index,
                    source=source,
                )
                row["source"] = source
                row["rfs_score"] = score
                train_rows.append(row)

        policy = cv._fit_direct_policy(
            train_rows,
            selector,
            mode="direct_policy",
            ridge=0.01,
            min_margin=0.0,
            max_rate=1.0,
            min_precision=0.0,
            router="off",
            model_family="linear",
            identity_features="full",
            stump_iterations=10,
            stump_lr=0.1,
            stump_thresholds=4,
            rff_dim=16,
            rff_scale=2.0,
            rff_seed=7,
            torch_epochs=3,
            torch_lr=0.01,
            torch_hidden=8,
            torch_batch_size=4,
            device="cpu",
            feature_mode="contextual",
            source_calibration=None,
            family_calibration=None,
            fallback_policy=None,
            fallback_selectors=None,
            scene_gate_policy=None,
            source_gate_policy=None,
            source_gate_selectors=None,
            source_veto_policy=None,
        )
        test_frame = sample_frame("segment-test-100", step=1.2)
        test_rows = []
        for index, (name, source, score, lateral) in enumerate(
            [
                ("constant_velocity", "kinematic", 4.0, 0.0),
                ("ridge_mean", "learned", 8.0, 0.2),
            ]
        ):
            row = cv.candidate_ranker_row(
                frame=test_frame,
                trajectory=[(float(step), lateral) for step in range(1, 21)],
                candidate_name=name,
                candidate_index=index,
                source=source,
            )
            row["source"] = source
            row["rfs_score"] = score
            test_rows.append(row)
        baseline = selector.select_row(test_rows)

        selected = cv._apply_direct_policy(
            policy,
            selector,
            test_rows,
            baseline,
            source_calibration=None,
            family_calibration=None,
        )

        self.assertEqual("constant_velocity", baseline["candidate_name"])
        self.assertEqual("ridge_mean", selected["candidate_name"])

    def test_sklearn_hist_gradient_boosting_gate_model_is_serializable(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")
        try:
            import sklearn  # noqa: F401
        except ImportError:
            self.skipTest("scikit-learn is not installed")
        observations = [
            {"features": [float(index), float(index % 2)], "gain": 2.0 if index % 2 else -1.0}
            for index in range(12)
        ]

        model = cv._fit_gate_model(
            observations,
            ridge=0.1,
            model_family="sklearn_hist_gradient_boosting",
            stump_iterations=8,
            stump_lr=0.1,
            stump_thresholds=4,
        )
        predictions = cv._predict_gate_model_array(
            cv.np.asarray([[1.0, 1.0], [2.0, 0.0]], dtype=cv.np.float64),
            model,
        )

        self.assertEqual("sklearn_hist_gradient_boosting", model["model_family"])
        self.assertEqual((2,), predictions.shape)
        json.dumps(model)

    def test_direct_policy_fit_uses_routed_selector_baseline(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")
        primary_selector = cv.WodPreferenceRanker(
            numeric_features=[],
            candidate_names=["constant_velocity", "ridge_mean"],
            candidate_families=[],
            feature_mean=[0.0, 0.0],
            feature_scale=[1.0, 1.0],
            weights=[1.0, 0.0],
            bias=0.0,
        )
        routed_selector = cv.WodPreferenceRanker(
            numeric_features=[],
            candidate_names=["constant_velocity", "ridge_mean"],
            candidate_families=[],
            feature_mean=[0.0, 0.0],
            feature_scale=[1.0, 1.0],
            weights=[0.0, 1.0],
            bias=0.0,
        )
        train_rows = []
        for frame_index in range(4):
            frame = sample_frame(f"segment-routed-{frame_index}-100", step=1.0)
            for index, (name, source, score, lateral) in enumerate(
                [
                    ("constant_velocity", "kinematic", 4.0, 0.0),
                    ("ridge_mean", "learned", 8.0, 0.2),
                ]
            ):
                row = cv.candidate_ranker_row(
                    frame=frame,
                    trajectory=[(float(step), lateral) for step in range(1, 21)],
                    candidate_name=name,
                    candidate_index=index,
                    source=source,
                )
                row["source"] = source
                row["rfs_score"] = score
                train_rows.append(row)

        policy = cv._fit_direct_policy(
            train_rows,
            primary_selector,
            mode="direct_policy",
            ridge=0.01,
            min_margin=0.0,
            max_rate=1.0,
            min_precision=0.0,
            router="off",
            model_family="linear",
            identity_features="none",
            stump_iterations=10,
            stump_lr=0.1,
            stump_thresholds=4,
            rff_dim=16,
            rff_scale=2.0,
            rff_seed=7,
            torch_epochs=3,
            torch_lr=0.01,
            torch_hidden=8,
            torch_batch_size=4,
            device="cpu",
            feature_mode="contextual",
            source_calibration=None,
            family_calibration=None,
            fallback_policy=None,
            fallback_selectors=None,
            scene_gate_policy=None,
            source_gate_policy=None,
            source_gate_selectors=None,
            source_veto_policy=None,
            routed_selectors={"routed": routed_selector},
            selector_route_policy={
                "router": "speed",
                "routes": {"speed:slow": "routed"},
                "global_target": "routed",
            },
        )

        self.assertIsNotNone(policy)
        self.assertEqual([], policy["candidate_names"])
        self.assertEqual([], policy["candidate_families"])
        self.assertFalse(
            any(
                "candidate_name:" in name or "candidate_family:" in name
                for name in policy["feature_names"]
            )
        )
        self.assertEqual(0.0, policy["train_positive_rate"])
        self.assertEqual("never", policy["decision"])

    def test_direct_policy_filters_baseline_and_candidate_sources(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")
        selector = cv.WodPreferenceRanker(
            numeric_features=[],
            candidate_names=["temporal_ridge_mean", "ridge_mean", "constant_velocity"],
            candidate_families=[],
            feature_mean=[0.0, 0.0, 0.0],
            feature_scale=[1.0, 1.0, 1.0],
            weights=[1.0, 0.0, 0.0],
            bias=0.0,
        )
        train_rows = []
        for frame_index in range(4):
            frame = sample_frame(f"segment-filtered-{frame_index}-100", step=1.0)
            for index, (name, source, score, lateral) in enumerate(
                [
                    ("temporal_ridge_mean", "temporal", 3.0, 0.0),
                    ("ridge_mean", "learned", 9.0, 0.2),
                    ("constant_velocity", "kinematic", 4.0, 0.0),
                ]
            ):
                row = cv.candidate_ranker_row(
                    frame=frame,
                    trajectory=[(float(step), lateral) for step in range(1, 21)],
                    candidate_name=name,
                    candidate_index=index,
                    source=source,
                )
                row["source"] = source
                row["rfs_score"] = score
                train_rows.append(row)

        policy = cv._fit_direct_policy(
            train_rows,
            selector,
            mode="direct_policy",
            ridge=0.01,
            min_margin=0.0,
            max_rate=1.0,
            min_precision=0.0,
            router="off",
            model_family="linear",
            identity_features="full",
            stump_iterations=10,
            stump_lr=0.1,
            stump_thresholds=4,
            rff_dim=16,
            rff_scale=2.0,
            rff_seed=7,
            torch_epochs=3,
            torch_lr=0.01,
            torch_hidden=8,
            torch_batch_size=4,
            device="cpu",
            feature_mode="contextual",
            source_calibration=None,
            family_calibration=None,
            fallback_policy=None,
            fallback_selectors=None,
            scene_gate_policy=None,
            source_gate_policy=None,
            source_gate_selectors=None,
            source_veto_policy=None,
            baseline_sources=("temporal",),
            baseline_prefixes=("temporal_ridge_",),
            candidate_sources=("learned",),
            candidate_prefixes=("ridge_",),
        )

        self.assertEqual(["temporal"], policy["baseline_sources"])
        self.assertEqual(["learned"], policy["candidate_sources"])
        self.assertEqual(
            "ridge_mean",
            cv._apply_direct_policy(
                policy,
                selector,
                train_rows[:3],
                train_rows[0],
                source_calibration=None,
                family_calibration=None,
            )["candidate_name"],
        )
        self.assertIs(
            train_rows[2],
            cv._apply_direct_policy(
                policy,
                selector,
                train_rows[:3],
                train_rows[2],
                source_calibration=None,
                family_calibration=None,
            ),
        )

    def test_direct_policy_threshold_uses_risk_adjusted_utility(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")
        observations = [
            {
                "features": [0.0],
                "gain": -2.0,
                "baseline_score": 7.0,
                "scene_score": 5.0,
                "frame_name": "segment-risk-0",
                "source": "learned",
                "group": "__all__",
                "is_baseline": False,
                "candidate_index": 0,
                "candidate_name": "high_gain_high_risk",
            },
            {
                "features": [1.0],
                "gain": 1.0,
                "baseline_score": 7.0,
                "scene_score": 8.0,
                "frame_name": "segment-risk-0",
                "source": "learned",
                "group": "__all__",
                "is_baseline": False,
                "candidate_index": 1,
                "candidate_name": "lower_gain_low_risk",
            },
        ]

        threshold_observations = cv._direct_policy_threshold_observations(
            observations,
            cv.np.asarray([0.0, 2.0], dtype=cv.np.float64),
            predicted_gains=cv.np.asarray([3.0, 2.0], dtype=cv.np.float64),
            predicted_downsides=cv.np.asarray([3.0, 0.0], dtype=cv.np.float64),
        )

        self.assertEqual(1, len(threshold_observations))
        self.assertEqual("lower_gain_low_risk", threshold_observations[0]["candidate_name"])
        self.assertAlmostEqual(2.0, threshold_observations[0]["predicted_gain"])
        self.assertAlmostEqual(0.0, threshold_observations[0]["predicted_downside"])
        self.assertAlmostEqual(2.0, threshold_observations[0]["predicted_utility"])

    def test_direct_policy_prediction_offset_blocks_unsafe_override(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")

        selector = cv.WodPreferenceRanker(
            numeric_features=[],
            candidate_names=["ridge_mean", "constant_velocity"],
            candidate_families=[],
            feature_mean=[0.0, 0.0],
            feature_scale=[1.0, 1.0],
            weights=[1.0, 0.0],
            bias=0.0,
        )
        frame = sample_frame("segment-direct-offset-100", step=1.0)
        rows = []
        for index, (name, source, score) in enumerate(
            [
                ("ridge_mean", "learned", 5.0),
                ("constant_velocity", "kinematic", 4.0),
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

        baseline = selector.select_row(rows)
        template_policy = {
            "mode": "direct_policy",
            "router": "off",
            "baseline_sources": [],
            "baseline_prefixes": [],
            "candidate_sources": [],
            "candidate_prefixes": [],
            "numeric_features": [],
            "candidate_names": [],
            "candidate_families": [],
        }
        _candidates, x = cv._direct_policy_candidate_matrix(
            template_policy,
            selector,
            rows,
            baseline,
            source_calibration=None,
            family_calibration=None,
        )
        policy = {
            **template_policy,
            "model_family": "linear",
            "feature_mean": [0.0 for _ in range(x.shape[1])],
            "feature_scale": [1.0 for _ in range(x.shape[1])],
            "weights": [0.0 for _ in range(x.shape[1])],
            "bias": 1.0,
            "risk_weight": 0.0,
            "decision": "margin",
            "threshold": 0.0,
            "prediction_offset": -2.0,
        }

        selected = cv._apply_direct_policy(
            policy,
            selector,
            rows,
            baseline,
            source_calibration=None,
            family_calibration=None,
        )

        self.assertEqual("ridge_mean", selected["candidate_name"])

    def test_override_metrics_aggregate_by_override_count(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")
        folds = [
            {
                "frames": 10,
                "direct_policy_override_count": 2,
                "direct_policy_true_positive_count": 2,
                "direct_policy_false_positive_count": 0,
                "direct_policy_gain_sum": 5.0,
                "direct_policy_false_positive_loss_sum": 0.0,
            },
            {
                "frames": 20,
                "direct_policy_override_count": 1,
                "direct_policy_true_positive_count": 0,
                "direct_policy_false_positive_count": 1,
                "direct_policy_gain_sum": -1.0,
                "direct_policy_false_positive_loss_sum": 1.0,
            },
        ]

        metrics = cv._aggregate_override_metric_fields(folds, "direct_policy")

        self.assertEqual(3, metrics["direct_policy_override_count"])
        self.assertEqual(2, metrics["direct_policy_true_positive_count"])
        self.assertAlmostEqual(0.1, metrics["direct_policy_selected_rate"])
        self.assertAlmostEqual(2.0 / 3.0, metrics["direct_policy_precision"])
        self.assertAlmostEqual(4.0 / 3.0, metrics["direct_policy_mean_gain"])
        self.assertAlmostEqual(1.0, metrics["direct_policy_false_positive_loss"])

    def test_direct_policy_oracle_metrics_aggregate_opportunity(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")
        folds = [
            {"frames": 10, "direct_policy_oracle_positive_count": 2, "direct_policy_oracle_gain_sum": 5.0},
            {"frames": 20, "direct_policy_oracle_positive_count": 1, "direct_policy_oracle_gain_sum": 1.0},
        ]

        metrics = cv._aggregate_direct_policy_oracle_metric_fields(folds)

        self.assertEqual(3, metrics["direct_policy_oracle_positive_count"])
        self.assertAlmostEqual(6.0, metrics["direct_policy_oracle_gain_sum"])
        self.assertAlmostEqual(0.1, metrics["direct_policy_oracle_positive_rate"])
        self.assertAlmostEqual(2.0, metrics["direct_policy_oracle_mean_positive_gain"])
        self.assertAlmostEqual(0.2, metrics["direct_policy_oracle_mean_gain_per_frame"])

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

    def test_learned_reliability_features_are_crossfit_by_segment(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")
        rows = []
        for frame_name in ["segment-a-100", "segment-b-100"]:
            frame = sample_frame(frame_name, step=1.0)
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

        profile = cv._fit_learned_reliability_features(
            rows,
            feature_mode="contextual",
            ridge=0.01,
            seed=3,
            folds=2,
        )
        cv._apply_learned_reliability_features(rows, profile, use_crossfit=True)

        learned = next(row for row in rows if row["source"] == "learned")
        kinematic = next(row for row in rows if row["source"] == "kinematic")
        self.assertGreater(
            learned["features"]["learned_reliability_mean_rfs"],
            kinematic["features"]["learned_reliability_mean_rfs"],
        )
        self.assertGreater(
            learned["features"]["learned_reliability_oracle_score"],
            kinematic["features"]["learned_reliability_oracle_score"],
        )
        self.assertIn("learned_reliability_rfs_frame_delta", learned["features"])
        self.assertIn("learned_reliability_rfs_frame_rank", learned["features"])
        self.assertIn("learned_reliability_frame_delta_score", learned["features"])
        self.assertIn("learned_reliability_frame_delta_rank", learned["features"])

    def test_learned_reliability_boosted_stumps_are_crossfit_by_segment(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")
        rows = []
        for frame_name in ["segment-a-100", "segment-b-100"]:
            frame = sample_frame(frame_name, step=1.0)
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

        profile = cv._fit_learned_reliability_features(
            rows,
            feature_mode="contextual",
            model_family="boosted_stumps",
            ridge=0.01,
            stump_iterations=20,
            stump_lr=0.2,
            stump_thresholds=4,
            seed=3,
            folds=2,
        )
        cv._apply_learned_reliability_features(rows, profile, use_crossfit=True)

        learned = next(row for row in rows if row["source"] == "learned")
        kinematic = next(row for row in rows if row["source"] == "kinematic")
        self.assertEqual("boosted_stumps", profile["model_family"])
        self.assertEqual(5, len(profile["target_models"]))
        self.assertGreater(
            learned["features"]["learned_reliability_mean_rfs"],
            kinematic["features"]["learned_reliability_mean_rfs"],
        )
        self.assertGreater(
            learned["features"]["learned_reliability_oracle_score"],
            kinematic["features"]["learned_reliability_oracle_score"],
        )

    def test_learned_reliability_sklearn_hist_gradient_boosting(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")
        try:
            import sklearn  # noqa: F401
        except ImportError:
            self.skipTest("scikit-learn is not installed")
        rows = []
        for frame_name in ["segment-a-100", "segment-b-100"]:
            frame = sample_frame(frame_name, step=1.0)
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

        profile = cv._fit_learned_reliability_features(
            rows,
            feature_mode="contextual",
            model_family="sklearn_hist_gradient_boosting",
            ridge=0.01,
            stump_iterations=5,
            stump_lr=0.2,
            stump_thresholds=4,
            seed=3,
            folds=2,
        )
        cv._apply_learned_reliability_features(rows, profile, use_crossfit=True)

        learned = next(row for row in rows if row["source"] == "learned")
        self.assertEqual("sklearn_hist_gradient_boosting", profile["model_family"])
        self.assertEqual(5, len(profile["estimators"]))
        self.assertTrue(cv.np.isfinite(learned["features"]["learned_reliability_mean_rfs"]))
        self.assertIn("learned_reliability_rfs_frame_delta", learned["features"])

    def test_learned_reliability_torch_mlp(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch is not installed")
        rows = []
        for frame_name in ["segment-a-100", "segment-b-100"]:
            frame = sample_frame(frame_name, step=1.0)
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

        profile = cv._fit_learned_reliability_features(
            rows,
            feature_mode="contextual",
            model_family="torch_mlp",
            ridge=0.01,
            torch_epochs=3,
            torch_lr=0.01,
            torch_hidden=8,
            torch_batch_size=4,
            device="cpu",
            seed=3,
            folds=2,
        )
        cv._apply_learned_reliability_features(rows, profile, use_crossfit=True)

        learned = next(row for row in rows if row["source"] == "learned")
        self.assertEqual("torch_mlp", profile["model_family"])
        self.assertEqual(3, len(profile["torch_layers"]))
        self.assertTrue(cv.np.isfinite(learned["features"]["learned_reliability_mean_rfs"]))
        self.assertIn("learned_reliability_rfs_frame_delta", learned["features"])

    def test_source_gate_features_include_learned_reliability_signal(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")
        frame = sample_frame("segment-a-100", step=1.0)
        baseline = cv.candidate_ranker_row(
            frame=frame,
            trajectory=[(float(step), 0.0) for step in range(1, 21)],
            candidate_name="constant_velocity",
            candidate_index=0,
            source="kinematic",
        )
        candidate = cv.candidate_ranker_row(
            frame=frame,
            trajectory=[(float(step), 0.1) for step in range(1, 21)],
            candidate_name="ridge_mean",
            candidate_index=1,
            source="learned",
        )
        baseline["rfs_score"] = 4.0
        candidate["rfs_score"] = 8.0
        baseline["features"]["learned_reliability_mean_rfs"] = 0.125
        candidate["features"]["learned_reliability_mean_rfs"] = 0.875
        selector = cv.WodPreferenceRanker([], [], [], [], [], [], 0.0)

        features = cv._source_gate_features(selector, baseline, candidate, None)

        latent_feature_count = 2 * (
            len(cv.WORLD_PRIOR_FEATURES) + len(cv.RETRIEVAL_LATENT_DISAGREEMENT_FEATURES)
        )
        learned_start = len(features) - 8 - latent_feature_count - (2 * len(cv.LEARNED_RELIABILITY_FEATURES))
        self.assertAlmostEqual(0.875, features[learned_start])
        self.assertAlmostEqual(0.75, features[learned_start + 1])

    def test_source_gate_features_include_latent_predictive_disagreement(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")

        frame = sample_frame("segment-a-100", step=1.0)
        baseline = cv.candidate_ranker_row(
            frame=frame,
            trajectory=[(float(step), 0.0) for step in range(1, 21)],
            candidate_name="constant_velocity",
            candidate_index=0,
            source="kinematic",
        )
        candidate = cv.candidate_ranker_row(
            frame=frame,
            trajectory=[(float(step), 0.1) for step in range(1, 21)],
            candidate_name="internnav_s2_waypoint_cautious",
            candidate_index=1,
            source="internnav",
        )
        candidate["features"]["world_prior_latent_error"] = 0.25
        baseline["features"]["world_prior_latent_error"] = 0.75
        candidate["features"]["retrieval_latent_disagreement"] = -0.4
        baseline["features"]["retrieval_latent_disagreement"] = 0.2
        selector = cv.WodPreferenceRanker([], [], [], [], [], [], 0.0)

        features = cv._source_gate_features(selector, baseline, candidate, None)
        latent_start = len(features) - 8 - 2 * (
            len(cv.WORLD_PRIOR_FEATURES) + len(cv.RETRIEVAL_LATENT_DISAGREEMENT_FEATURES)
        )
        world_error_index = latent_start
        disagreement_index = latent_start + 2 * len(cv.WORLD_PRIOR_FEATURES) + 2 * (
            cv.RETRIEVAL_LATENT_DISAGREEMENT_FEATURES.index("retrieval_latent_disagreement")
        )

        self.assertAlmostEqual(0.25, features[world_error_index])
        self.assertAlmostEqual(-0.5, features[world_error_index + 1])
        self.assertAlmostEqual(-0.4, features[disagreement_index])
        self.assertAlmostEqual(-0.6, features[disagreement_index + 1])

    def test_source_gate_can_append_local_selector_confidence_features(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")

        class FakeSelector:
            def __init__(self, scores: dict[str, float]) -> None:
                self._scores = scores

            def predict_row(self, row: dict[str, object]) -> float:
                return self._scores[str(row["candidate_name"])]

        frame = sample_frame("segment-a-100", step=1.0)
        baseline = cv.candidate_ranker_row(
            frame=frame,
            trajectory=[(float(step), 0.0) for step in range(1, 21)],
            candidate_name="constant_velocity",
            candidate_index=0,
            source="kinematic",
        )
        candidate = cv.candidate_ranker_row(
            frame=frame,
            trajectory=[(float(step), 0.1) for step in range(1, 21)],
            candidate_name="ensemble_a",
            candidate_index=1,
            source="learned",
        )
        peer = cv.candidate_ranker_row(
            frame=frame,
            trajectory=[(float(step), 0.2) for step in range(1, 21)],
            candidate_name="ensemble_b",
            candidate_index=2,
            source="learned",
        )
        global_selector = FakeSelector(
            {
                "constant_velocity": 1.0,
                "ensemble_a": 2.0,
                "ensemble_b": 3.0,
            }
        )
        local_selector = FakeSelector(
            {
                "constant_velocity": 1.5,
                "ensemble_a": 5.0,
                "ensemble_b": 4.0,
            }
        )

        base_features = cv._source_gate_features(global_selector, baseline, candidate, None)
        features = cv._source_gate_features(
            global_selector,
            baseline,
            candidate,
            None,
            local_selector=local_selector,
            source_rows=[candidate, peer],
            include_local_confidence=True,
        )

        self.assertEqual(len(base_features) + 8, len(features))
        self.assertAlmostEqual(5.0, features[-8])
        self.assertAlmostEqual(3.0, features[-7])
        self.assertAlmostEqual(3.5, features[-6])
        self.assertAlmostEqual(2.5, features[-5])
        self.assertAlmostEqual(1.0, features[-4])
        self.assertAlmostEqual(-1.0, features[-3])
        self.assertAlmostEqual(2.0, features[-2])
        self.assertAlmostEqual(cv.np.log1p(2), features[-1])

    def test_boosted_source_gate_model_learns_nonlinear_gain(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")

        observations = [
            {"features": [-2.0], "gain": -1.0},
            {"features": [-1.0], "gain": -0.5},
            {"features": [1.0], "gain": 0.5},
            {"features": [2.0], "gain": 1.0},
        ]
        model = cv._fit_gate_model(
            observations,
            ridge=0.01,
            model_family="boosted_stumps",
            stump_iterations=20,
            stump_lr=0.2,
            stump_thresholds=4,
        )
        predictions = cv._predict_gate_model_array(cv.np.asarray([[-2.0], [2.0]]), model)

        self.assertEqual("boosted_stumps", model["model_family"])
        self.assertGreater(predictions[1], predictions[0])

    def test_gate_ensemble_min_is_conservative_member_consensus(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")

        observations = [
            {"features": [-2.0, 0.0], "gain": -1.0},
            {"features": [-1.0, 1.0], "gain": -0.5},
            {"features": [1.0, 0.0], "gain": 0.5},
            {"features": [2.0, 1.0], "gain": 1.0},
        ]
        model = cv._fit_gate_model(
            observations,
            ridge=0.01,
            model_family="ensemble_min",
            stump_iterations=20,
            stump_lr=0.1,
            stump_thresholds=4,
        )
        x = cv.np.asarray([observation["features"] for observation in observations], dtype=cv.np.float64)
        ensemble_predictions = cv._predict_gate_model_array(x, model)
        member_predictions = [
            cv._predict_gate_model_array(x, member)
            for member in model["members"]
        ]

        self.assertEqual("ensemble_min", model["model_family"])
        self.assertEqual(2, len(model["members"]))
        self.assertTrue(cv.np.allclose(ensemble_predictions, cv.np.min(cv.np.vstack(member_predictions), axis=0)))

    def test_logistic_gate_model_learns_positive_override(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")

        observations = [
            {"features": [-2.0], "gain": -1.0},
            {"features": [-1.0], "gain": -0.5},
            {"features": [1.0], "gain": 0.5},
            {"features": [2.0], "gain": 1.0},
        ]
        model = cv._fit_gate_model(
            observations,
            ridge=0.01,
            model_family="logistic",
            stump_iterations=120,
            stump_lr=0.1,
        )
        predictions = cv._predict_gate_model_array(cv.np.asarray([[-2.0], [2.0]]), model)

        self.assertEqual("logistic", model["model_family"])
        self.assertGreater(predictions[1], predictions[0])
        self.assertGreaterEqual(float(predictions[1]), 0.0)
        self.assertLessEqual(float(predictions[1]), 1.0)

    def test_random_fourier_gate_model_learns_nonlinear_gain(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")

        observations = [
            {"features": [-3.0], "gain": -1.0},
            {"features": [-2.0], "gain": -1.0},
            {"features": [-1.0], "gain": -0.5},
            {"features": [0.0], "gain": 1.0},
            {"features": [1.0], "gain": -0.5},
            {"features": [2.0], "gain": -1.0},
            {"features": [3.0], "gain": -1.0},
        ]
        model = cv._fit_gate_model(
            observations,
            ridge=0.001,
            model_family="random_fourier",
            rff_dim=64,
            rff_scale=1.0,
            rff_seed=11,
        )
        predictions = cv._predict_gate_model_array(cv.np.asarray([[0.0], [3.0]]), model)

        self.assertEqual("random_fourier", model["model_family"])
        self.assertGreater(predictions[0], predictions[1])

    def test_torch_mlp_gate_model_learns_gain(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch is not installed")

        observations = [
            {"features": [-2.0, 1.0], "gain": -1.0},
            {"features": [-1.0, 0.5], "gain": -0.5},
            {"features": [1.0, -0.5], "gain": 0.5},
            {"features": [2.0, -1.0], "gain": 1.0},
        ]
        model = cv._fit_gate_model(
            observations,
            ridge=0.01,
            model_family="torch_mlp",
            torch_epochs=80,
            torch_lr=0.01,
            torch_hidden=16,
            torch_batch_size=4,
            device="cpu",
            rff_seed=13,
        )
        predictions = cv._predict_gate_model_array(cv.np.asarray([[-2.0, 1.0], [2.0, -1.0]]), model)

        self.assertEqual("torch_mlp", model["model_family"])
        self.assertEqual("cpu", model["torch_device_used"])
        self.assertGreater(predictions[1], predictions[0])

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
            risk_weight=0.0,
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
            risk_weight=0.0,
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

    def test_logistic_source_gate_model_can_classify_positive_override(self) -> None:
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
            sources=("kinematic",),
            candidate_prefixes=(),
            route_allowlist=(),
            margin=0.0,
            risk_weight=0.0,
            router="off",
            ridge=0.01,
            max_rate=1.0,
            min_precision=0.0,
            min_route_observations=0,
            min_route_positives=0,
            source_calibration=None,
            fallback_policy=None,
            fallback_selectors=None,
            model_family="logistic",
            stump_iterations=30,
            stump_lr=0.2,
        )
        selected = cv._apply_source_gate(
            policy,
            FakeSelector(),
            rows,
            rows[0],
            source_calibration=None,
        )

        self.assertEqual("logistic", policy["gate_model"])
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

    def test_source_gate_prediction_offset_blocks_unsafe_override(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")

        class FakeSelector:
            def select_row(self, rows):
                return max(rows, key=self.predict_row)

            def predict_row(self, row):
                return float(row["predicted_selector_score"])

        frame = sample_frame("segment-a-100", step=1.0)
        rows = []
        for index, (name, source, predicted, score) in enumerate(
            [
                ("ridge_mean", "learned", 1.0, 5.0),
                ("constant_velocity", "kinematic", 2.0, 4.0),
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

        features = cv._source_gate_features(FakeSelector(), rows[0], rows[1], None)
        source_model = {
            "model_family": "linear",
            "feature_mean": [0.0 for _ in features],
            "feature_scale": [1.0 for _ in features],
            "weights": [0.0 for _ in features],
            "bias": 1.0,
        }
        policy = {
            "mode": "independent_train_margin",
            "router": "off",
            "sources": ["kinematic"],
            "candidate_prefixes": [],
            "deny_prefixes": [],
            "route_allowlist": [],
            "source_models": {"kinematic": source_model},
            "source_decisions": {
                "kinematic": {
                    "decision": "margin",
                    "threshold": 0.0,
                    "prediction_offset": -2.0,
                }
            },
        }

        selected = cv._apply_source_gate(
            policy,
            FakeSelector(),
            rows,
            rows[0],
            source_calibration=None,
        )

        self.assertEqual("learned", selected["source"])

    def test_gate_threshold_can_penalize_false_positive_loss(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")

        observations = [
            {"baseline_score": 5.0, "scene_score": 8.0},
            {"baseline_score": 5.0, "scene_score": 4.0},
        ]
        predicted = cv.np.asarray([1.0, 1.0], dtype=cv.np.float64)

        score_only = cv._fit_scene_gate_threshold(
            observations,
            predicted,
            margin=0.0,
            max_rate=1.0,
            min_precision=0.0,
            risk_weight=0.0,
        )
        risk_adjusted = cv._fit_scene_gate_threshold(
            observations,
            predicted,
            margin=0.0,
            max_rate=1.0,
            min_precision=0.0,
            risk_weight=3.0,
        )

        self.assertIsNotNone(score_only)
        self.assertIsNone(risk_adjusted)

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
            candidate_prefixes=(),
            deny_prefixes=(),
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

    def test_source_veto_training_respects_selector_deny_sources(self) -> None:
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
                ("neural_system2_mode_0_rank0", "system2", 3.0, 1.0),
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
            candidate_prefixes=(),
            deny_prefixes=(),
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
            selector_deny_sources=("system2",),
        )

        self.assertEqual(["system2"], policy["selector_deny_sources"])
        self.assertEqual(1, policy["train_observations"])
        self.assertEqual(
            "constant_velocity",
            cv._apply_source_veto(
                policy,
                FakeSelector(),
                rows,
                rows[1],
                source_calibration=None,
            )["candidate_name"],
        )

    def test_boosted_source_veto_model_can_replace_bad_selected_source(self) -> None:
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
            candidate_prefixes=(),
            deny_prefixes=(),
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
            model_family="boosted_stumps",
            stump_iterations=20,
            stump_lr=0.2,
            stump_thresholds=4,
        )
        selected = cv._apply_source_veto(
            policy,
            FakeSelector(),
            rows,
            rows[0],
            source_calibration=None,
        )

        self.assertEqual("boosted_stumps", policy["gate_model"])
        self.assertEqual("boosted_stumps", policy["model_family"])
        self.assertEqual("constant_velocity", selected["candidate_name"])

    def test_source_veto_can_target_selected_candidate_prefix(self) -> None:
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

        matching_policy = cv._fit_source_veto_policy(
            rows,
            FakeSelector(),
            mode="train_margin",
            sources=("temporal",),
            candidate_prefixes=("temporal_ridge_",),
            deny_prefixes=(),
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
        filtered_policy = cv._fit_source_veto_policy(
            rows,
            FakeSelector(),
            mode="train_margin",
            sources=("temporal",),
            candidate_prefixes=("temporal_ridge_residual_",),
            deny_prefixes=(),
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

        self.assertEqual(["temporal_ridge_"], matching_policy["candidate_prefixes"])
        self.assertEqual(
            "constant_velocity",
            cv._apply_source_veto(
                matching_policy,
                FakeSelector(),
                rows,
                rows[0],
                source_calibration=None,
            )["candidate_name"],
        )
        self.assertEqual(0, filtered_policy["train_observations"])
        self.assertIs(
            rows[0],
            cv._apply_source_veto(
                filtered_policy,
                FakeSelector(),
                rows,
                rows[0],
                source_calibration=None,
            ),
        )

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

    def test_safety_utility_rejects_unsafe_high_utility_candidate(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")

        class FakeSelector:
            def predict_row(self, row):
                return float(row["features"].get("selector_score", 0.0))

        policy = {
            "mode": "safety_utility",
            "feature_mean": [0.0 for _ in cv.SAFETY_UTILITY_FEATURES],
            "feature_scale": [1.0 for _ in cv.SAFETY_UTILITY_FEATURES],
            "risk_bias": 0.0,
            "risk_weights": [0.0 for _ in cv.SAFETY_UTILITY_FEATURES],
            "utility_bias": 0.0,
            "utility_weights": [
                1.0 if name == "source_is_learned" else 0.0
                for name in cv.SAFETY_UTILITY_FEATURES
            ],
        }
        kinematic = _safety_row("constant_velocity", "kinematic", selector_score=0.2, rfs=7.0)
        unsafe = _safety_row(
            "learned_reverse",
            "learned",
            selector_score=1.0,
            rfs=10.0,
            reverse_distance=30.0,
        )

        selected = cv._apply_safety_utility_policy(
            policy,
            FakeSelector(),
            [kinematic, unsafe],
            unsafe,
            source_calibration=None,
            family_calibration=None,
        )

        self.assertEqual("constant_velocity", selected["candidate_name"])

    def test_safety_filter_only_replaces_hard_unsafe_selection(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")

        class FakeSelector:
            def predict_row(self, row):
                return float(row["features"].get("selector_score", 0.0))

        kinematic = _safety_row("constant_velocity", "kinematic", selector_score=0.2, rfs=7.0)
        safe_learned = _safety_row("learned_safe", "learned", selector_score=0.4, rfs=8.0)
        unsafe = _safety_row(
            "learned_reverse",
            "learned",
            selector_score=1.0,
            rfs=10.0,
            reverse_distance=30.0,
        )

        kept = cv._apply_safety_utility_policy(
            {"mode": "safety_filter"},
            FakeSelector(),
            [kinematic, safe_learned, unsafe],
            safe_learned,
            source_calibration=None,
            family_calibration=None,
        )
        replaced = cv._apply_safety_utility_policy(
            {"mode": "safety_filter"},
            FakeSelector(),
            [kinematic, safe_learned, unsafe],
            unsafe,
            source_calibration=None,
            family_calibration=None,
        )

        self.assertEqual("learned_safe", kept["candidate_name"])
        self.assertEqual("learned_safe", replaced["candidate_name"])

    def test_safety_filter_ignores_affordance_only_rejection(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")

        class FakeSelector:
            def predict_row(self, row):
                return float(row["features"].get("selector_score", 0.0))

        kinematic = _safety_row("constant_velocity", "kinematic", selector_score=0.2, rfs=7.0)
        affordance_only = _safety_row(
            "learned_affordance_only",
            "learned",
            selector_score=1.0,
            rfs=9.0,
            reverse_distance=4.0,
        )

        selected = cv._apply_safety_utility_policy(
            {"mode": "safety_filter"},
            FakeSelector(),
            [kinematic, affordance_only],
            affordance_only,
            source_calibration=None,
            family_calibration=None,
        )

        self.assertEqual("learned_affordance_only", selected["candidate_name"])

    def test_safety_utility_keeps_safe_learned_candidate(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")

        class FakeSelector:
            def predict_row(self, row):
                return float(row["features"].get("selector_score", 0.0))

        policy = {
            "mode": "safety_utility",
            "feature_mean": [0.0 for _ in cv.SAFETY_UTILITY_FEATURES],
            "feature_scale": [1.0 for _ in cv.SAFETY_UTILITY_FEATURES],
            "risk_bias": 0.0,
            "risk_weights": [0.0 for _ in cv.SAFETY_UTILITY_FEATURES],
            "utility_bias": 0.0,
            "utility_weights": [
                1.0 if name == "source_is_learned" else 0.0
                for name in cv.SAFETY_UTILITY_FEATURES
            ],
        }
        kinematic = _safety_row("constant_velocity", "kinematic", selector_score=0.2, rfs=7.0)
        learned = _safety_row("learned_safe", "learned", selector_score=1.0, rfs=9.0)

        selected = cv._apply_safety_utility_policy(
            policy,
            FakeSelector(),
            [kinematic, learned],
            kinematic,
            source_calibration=None,
            family_calibration=None,
        )

        self.assertEqual("learned_safe", selected["candidate_name"])

    def test_safety_utility_postprocess_cv_smoke(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")
        frames = [
            sample_frame(f"segment-{index}-100", step=1.0 + index * 0.1)
            for index in range(6)
        ]

        report = cv.cross_validate_trajectory_model(
            frames,
            folds=3,
            seed=5,
            ridge=1.0,
            feature_set=cv.FEATURE_SET_BASE,
            residual_modes=1,
            scorer=cv._local_rfs_score,
            selector_postprocess="safety_utility",
            safety_utility_ridge=1.0,
            safety_utility_floor=7.0,
        )

        self.assertEqual("safety_utility", report["selector_postprocess"])
        self.assertIn("safety_utility_selected_rate", report)
        self.assertIn("safety_utility_mean_gain", report)

    def test_safety_direct_policy_cv_smoke(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")
        frames = [
            sample_frame(f"segment-{index}-100", step=1.0 + index * 0.1)
            for index in range(6)
        ]

        report = cv.cross_validate_trajectory_model(
            frames,
            folds=3,
            seed=5,
            ridge=1.0,
            feature_set=cv.FEATURE_SET_BASE,
            residual_modes=1,
            scorer=cv._local_rfs_score,
            selector_postprocess="safety_direct_policy",
            direct_policy_model="linear",
            direct_policy_router="speed",
            direct_policy_max_rate=0.5,
            direct_policy_min_precision=0.0,
            direct_policy_min_route_observations=2,
            direct_policy_min_route_positives=1,
            learned_reliability_gate_features=True,
            learned_reliability_gate_feature_mode="learned_reliability_confidence_contextual",
            safety_utility_ridge=1.0,
            safety_utility_floor=7.0,
        )

        self.assertEqual("safety_direct_policy", report["selector_postprocess"])
        self.assertTrue(report["direct_policy_enabled"])
        self.assertTrue(report["safety_utility_enabled"])
        self.assertIn("direct_policy_selected_rate", report)
        self.assertIn("safety_utility_selected_rate", report)
        self.assertEqual(2, report["direct_policy_min_route_observations"])
        self.assertEqual(1, report["direct_policy_min_route_positives"])
        self.assertTrue(report["learned_reliability_gate_features"])
        self.assertEqual("relative_confidence_contextual", report["learned_reliability_resolved_gate_feature_mode"])

    def test_safety_then_direct_policy_cv_smoke(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")
        frames = [
            sample_frame(f"segment-{index}-100", step=1.0 + index * 0.1)
            for index in range(6)
        ]

        report = cv.cross_validate_trajectory_model(
            frames,
            folds=3,
            seed=5,
            ridge=1.0,
            feature_set=cv.FEATURE_SET_BASE,
            residual_modes=1,
            scorer=cv._local_rfs_score,
            selector_postprocess="safety_then_direct_policy",
            safety_utility_train_scope="selector",
            safety_utility_deny_sources=("system2",),
            direct_policy_model="linear",
            direct_policy_max_rate=0.5,
            direct_policy_min_precision=0.0,
            safety_utility_ridge=1.0,
            safety_utility_floor=7.0,
        )

        self.assertEqual("safety_then_direct_policy", report["selector_postprocess"])
        self.assertTrue(report["direct_policy_enabled"])
        self.assertTrue(report["safety_utility_enabled"])
        self.assertEqual("selector", report["safety_utility_train_scope"])
        self.assertEqual(["system2"], report["safety_utility_deny_sources"])

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

    def test_meta_rescore_learns_to_override_overconfident_selector(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")

        class FakeSelector:
            def predict_row(self, row):
                return float(row["predicted_selector_score"])

        rows = []
        for frame_index in range(3):
            frame = sample_frame(f"segment-meta-{frame_index}-100", step=1.0)
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
                row["predicted_selector_score"] = predicted
                row["rfs_score"] = score
                rows.append(row)

        policy = cv._fit_meta_rescore_policy(
            rows,
            FakeSelector(),
            mode="meta_rescore",
            ridge=0.01,
            min_margin=0.0,
            max_rate=1.0,
            source_calibration=None,
            family_calibration=None,
        )
        selected = cv._apply_meta_rescore_policy(
            policy,
            FakeSelector(),
            rows[:2],
            rows[0],
            source_calibration=None,
            family_calibration=None,
        )

        self.assertEqual("meta_rescore", policy["mode"])
        self.assertEqual("constant_velocity", selected["candidate_name"])

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

    def test_selector_numeric_features_can_include_world_prior(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")

        features = cv._selector_numeric_features("world_prior_contextual")

        self.assertIn("world_prior_latent_error", features)
        self.assertIn("world_prior_constraint_cost", features)
        self.assertIn("world_prior_predicted_final_x", features)

    def test_selector_numeric_features_can_include_candidate_confidence(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")

        features = cv._selector_numeric_features("relative_confidence_contextual")

        self.assertIn("candidate_model_confidence", features)
        self.assertIn("candidate_model_confidence_signed_log", features)
        self.assertIn("candidate_model_confidence_present", features)

    def test_learned_reliability_confidence_mode_fits_required_features(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")

        features = cv._selector_numeric_features("learned_reliability_confidence_contextual")

        self.assertTrue(cv._selector_uses_learned_reliability_features("learned_reliability_confidence_contextual"))
        self.assertEqual(
            "relative_confidence_contextual",
            cv._learned_reliability_base_feature_mode("learned_reliability_confidence_contextual"),
        )
        self.assertIn("learned_reliability_margin", features)
        self.assertIn("candidate_model_confidence_present", features)

    def test_selector_numeric_features_can_include_retrieval_latent_disagreement(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")

        features = cv._selector_numeric_features("relative_retrieval_latent_contextual")

        self.assertIn("candidate_model_confidence", features)
        self.assertIn("world_prior_latent_error", features)
        self.assertIn("retrieval_support_score", features)
        self.assertIn("latent_feasibility_score", features)
        self.assertIn("retrieval_latent_disagreement", features)
        self.assertIn("retrieval_latent_low_support", features)

    def test_selector_numeric_features_can_include_precedent_latent_disagreement(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")

        features = cv._selector_numeric_features("relative_precedent_latent_contextual")

        self.assertIn("family_reliability_mean_rfs", features)
        self.assertIn("source_reliability_oracle_rate", features)
        self.assertIn("world_prior_latent_error_z", features)
        self.assertIn("retrieval_support_score", features)
        self.assertIn("latent_feasibility_score", features)

    def test_candidate_metadata_features_include_model_confidence(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")
        row = cv.candidate_ranker_row(
            frame=sample_frame("segment-conf-100", step=1.0),
            trajectory=[(float(index), 0.0) for index in range(1, 21)],
            candidate_name="neural_system2_mode_0_rank0",
            candidate_index=0,
            source="wod_v20_neural_system2_planner",
        )

        cv._add_candidate_metadata_features(row, {"candidate_model_confidence": 8.5})

        self.assertEqual(8.5, row["features"]["candidate_model_confidence"])
        self.assertGreater(row["features"]["candidate_model_confidence_signed_log"], 0.0)
        self.assertEqual(1.0, row["features"]["candidate_model_confidence_present"])

    def test_retrieval_latent_disagreement_features_reflect_support_mismatch(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")
        row = cv.candidate_ranker_row(
            frame=sample_frame("segment-disagree-100", step=1.0),
            trajectory=[(float(index), 0.0) for index in range(1, 21)],
            candidate_name="neural_system2_mode_0_rank0",
            candidate_index=0,
            source="wod_v20_neural_system2_planner",
        )

        cv._add_candidate_metadata_features(row, {"candidate_model_confidence": 2.0})
        row["features"]["world_prior_latent_error_log"] = 0.9
        row["features"]["world_prior_constraint_cost"] = 0.4

        cv._add_retrieval_latent_disagreement_features(row)

        self.assertGreater(row["features"]["retrieval_support_score"], 0.0)
        self.assertLess(row["features"]["latent_feasibility_score"], 0.0)
        self.assertGreater(row["features"]["retrieval_latent_disagreement"], 0.0)
        self.assertGreater(row["features"]["retrieval_without_latent"], 0.0)

    def test_precedent_latent_disagreement_uses_reliability_features(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")
        features = {
            "family_reliability_mean_rfs": 8.0,
            "source_reliability_mean_rfs": 7.0,
            "family_reliability_oracle_rate": 0.6,
            "source_reliability_oracle_rate": 0.5,
            "family_reliability_regret_mean": 0.8,
            "source_reliability_regret_mean": 1.2,
            "world_prior_latent_error_z": 0.3,
            "world_prior_latent_error_log": 0.2,
            "world_prior_constraint_cost": 0.1,
            "retrieval_support_score": 0.0,
            "latent_feasibility_score": 0.0,
            "retrieval_latent_disagreement": 0.0,
            "retrieval_latent_abs_disagreement": 0.0,
            "retrieval_latent_agreement": 0.0,
            "retrieval_without_latent": 0.0,
            "latent_without_retrieval": 0.0,
            "retrieval_latent_low_support": 0.0,
        }

        cv._add_precedent_latent_disagreement_features(features)

        self.assertGreater(features["retrieval_support_score"], 0.0)
        self.assertGreater(features["latent_feasibility_score"], 0.0)
        self.assertGreaterEqual(features["retrieval_latent_agreement"], 0.0)

    def test_candidate_record_from_json_preserves_confidence(self) -> None:
        if cv is None:
            self.skipTest("numpy is not installed")
        payload = {
            "frame_name": "segment-conf-100",
            "source": "wod_v20_neural_system2_planner",
            "candidate_name": "neural_system2_mode_0_rank0",
            "candidate_index": 0,
            "confidence": 8.5,
            "trajectory_20wp_4hz": [[float(index), 0.0] for index in range(1, 21)],
        }

        record = cv.candidate_record_from_json(payload)

        self.assertEqual(8.5, record.confidence)

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
