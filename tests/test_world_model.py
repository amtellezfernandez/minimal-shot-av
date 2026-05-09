from __future__ import annotations

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minimal_shot_av.model.wod_e2e import WodCameraImage, WodE2EPreferenceFrame
from minimal_shot_av.model.world_model import (
    FEATURE_MODE_EXTERNAL_EMBEDDINGS,
    FEATURE_MODE_SCENE_TOKENS,
    LATENT_SOURCE_SCENE,
    LatentPredictiveWorldPrior,
    LearnedWorldModel,
    attach_external_embedding_cache,
    attach_scene_token_cache,
    external_embedding_features,
    fit_latent_predictive_world_prior,
    fit_world_model,
    load_external_embedding_cache,
    load_scene_token_cache,
    scene_token_features,
    scene_token_names,
    write_external_embedding_cache,
    write_scene_token_cache,
)


def sample_frame(
    frame_name: str,
    *,
    step: float = 1.0,
    lateral: float = 0.0,
    intent: int = 1,
) -> WodE2EPreferenceFrame:
    past = [
        (-3.0 * step, -0.25 * lateral),
        (-2.0 * step, -0.10 * lateral),
        (-1.0 * step, 0.0),
        (0.0, 0.0),
    ]
    future = [
        (
            step * float(index),
            lateral * (float(index) / 20.0) ** 2,
        )
        for index in range(1, 21)
    ]
    return WodE2EPreferenceFrame(
        frame_name=frame_name,
        past_trajectory=past,
        future_trajectory=future,
        intent=intent,
        init_speed_mps=step * 4.0,
        references=[],
    )


def camera_frame(frame_name: str, payload: bytes) -> WodE2EPreferenceFrame:
    base = sample_frame(frame_name)
    return WodE2EPreferenceFrame(
        frame_name=base.frame_name,
        past_trajectory=base.past_trajectory,
        future_trajectory=base.future_trajectory,
        intent=base.intent,
        init_speed_mps=base.init_speed_mps,
        references=[],
        camera_images=[WodCameraImage(name="FRONT", jpeg=payload)],
    )


class LearnedWorldModelTests(unittest.TestCase):
    def test_fit_predicts_embedding_summary_and_future_without_text_io(self) -> None:
        model = fit_world_model(
            [
                sample_frame("straight_slow", step=0.5),
                sample_frame("straight", step=1.0),
                sample_frame("left", step=1.0, lateral=4.0, intent=2),
                sample_frame("right", step=1.0, lateral=-4.0, intent=3),
            ],
            latent_dim=3,
            ridge=0.1,
        )

        prediction = model.predict(
            sample_frame("query", step=1.0, lateral=4.0, intent=2).past_trajectory,
            intent=2,
            init_speed_mps=4.0,
        )

        self.assertEqual(3, len(prediction.embedding))
        self.assertEqual(20, len(prediction.trajectory))
        self.assertIn("future_final_x", prediction.future_summary)
        self.assertIn("future_mean_abs_heading_change", prediction.future_summary)
        self.assertEqual(3, len(prediction.nearest_experiences))
        self.assertEqual("left", prediction.nearest_experiences[0][0])

    def test_retrieval_uses_learned_experience_embeddings(self) -> None:
        model = fit_world_model(
            [
                sample_frame("slow", step=0.5),
                sample_frame("fast", step=2.0),
                sample_frame("left", step=1.0, lateral=3.0, intent=2),
            ],
            latent_dim=4,
        )

        embedding = model.encode(sample_frame("query", step=2.0).past_trajectory, intent=1, init_speed_mps=8.0)
        neighbors = model.retrieve(embedding, limit=2)

        self.assertEqual(2, len(neighbors))
        self.assertEqual("fast", neighbors[0][0])

    def test_scene_token_mode_conditions_embedding_on_camera_context(self) -> None:
        dark = camera_frame("dark_scene", bytes([0, 1, 2, 3]) * 8)
        bright = camera_frame("bright_scene", bytes([250, 251, 252, 253]) * 8)
        model = fit_world_model(
            [dark, bright, sample_frame("motion_reference", step=2.0)],
            latent_dim=4,
            feature_mode=FEATURE_MODE_SCENE_TOKENS,
        )

        prediction = model.predict_frame(camera_frame("query", bytes([250, 251, 252, 253]) * 8))

        self.assertEqual(FEATURE_MODE_SCENE_TOKENS, model.feature_mode)
        self.assertEqual(len(scene_token_names()), len(model.scene_token_names))
        self.assertEqual("bright_scene", prediction.nearest_experiences[0][0])

    def test_scene_token_features_have_stable_named_shape(self) -> None:
        frame = camera_frame("front_only", b"abcdef")

        tokens = scene_token_features(frame)

        self.assertEqual(len(scene_token_names()), len(tokens))
        self.assertEqual(1.0, tokens[0])
        self.assertGreater(tokens[1], 0.0)

    def test_scene_token_cache_round_trips_and_attaches_to_frames(self) -> None:
        frame = camera_frame("front_only", b"abcdef")
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "scene_tokens.json"
            count = write_scene_token_cache([frame], path)
            cache = load_scene_token_cache(path)
            attached = attach_scene_token_cache([sample_frame("front_only")], cache)

        self.assertEqual(1, count)
        self.assertEqual(len(scene_token_names()), len(attached[0].scene_tokens or []))
        self.assertEqual([], attached[0].camera_images)
        self.assertEqual(scene_token_features(frame), scene_token_features(attached[0]))

    def test_scene_token_cache_requires_requested_frames_by_default(self) -> None:
        with self.assertRaises(KeyError):
            attach_scene_token_cache([sample_frame("missing")], {})

    def test_external_embedding_cache_round_trips_and_attaches_to_frames(self) -> None:
        cache = {"a": [0.1, 0.2, 0.3], "b": [0.4, 0.5, 0.6]}
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "cosmos_embeddings.json"
            count = write_external_embedding_cache(cache, path, source="cosmos-test")
            loaded = load_external_embedding_cache(path)
            attached = attach_external_embedding_cache([sample_frame("a")], loaded)

        self.assertEqual(2, count)
        self.assertEqual([0.1, 0.2, 0.3], external_embedding_features(attached[0]))

    def test_external_embedding_cache_rejects_bad_rows(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad_embeddings.json"
            path.write_text(
                json.dumps(
                    {
                        "schema": "external_frame_embeddings_v1",
                        "source": "unit-test",
                        "dimension": 2,
                        "frames": {"a": [1.0, 2.0], "b": [3.0]},
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                load_external_embedding_cache(path)

    def test_external_embedding_cache_requires_source_and_dimension(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad_embeddings.json"
            path.write_text(
                json.dumps(
                    {
                        "schema": "external_frame_embeddings_v1",
                        "frames": {"a": [1.0, 2.0]},
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "source"):
                load_external_embedding_cache(path)

            path.write_text(
                json.dumps(
                    {
                        "schema": "external_frame_embeddings_v1",
                        "source": "unit-test",
                        "frames": {"a": [1.0, 2.0]},
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "dimension"):
                load_external_embedding_cache(path)

    def test_external_embedding_attach_preserves_camera_payloads(self) -> None:
        frame = camera_frame("front_only", b"abcdef")

        attached = attach_external_embedding_cache([frame], {"front_only": [1.0, 2.0]})

        self.assertEqual([1.0, 2.0], attached[0].external_embedding)
        self.assertEqual(frame.camera_images, attached[0].camera_images)

    def test_external_embedding_cache_requires_requested_frames_by_default(self) -> None:
        with self.assertRaises(KeyError):
            attach_external_embedding_cache([sample_frame("missing")], {"present": [1.0, 2.0]})

    def test_external_embedding_mode_conditions_embedding_on_cached_vector(self) -> None:
        frames = attach_external_embedding_cache(
            [
                sample_frame("slow_visual", step=1.0),
                sample_frame("fast_visual", step=1.0),
                sample_frame("motion_reference", step=2.0),
            ],
            {
                "slow_visual": [0.0, 0.0, 0.0],
                "fast_visual": [1.0, 1.0, 1.0],
                "motion_reference": [0.5, 0.5, 0.5],
            },
        )
        model = fit_world_model(frames, latent_dim=2, feature_mode=FEATURE_MODE_EXTERNAL_EMBEDDINGS)
        query = attach_external_embedding_cache([sample_frame("query", step=1.0)], {"query": [1.0, 1.0, 1.0]})[0]

        prediction = model.predict_frame(query)

        self.assertEqual(FEATURE_MODE_EXTERNAL_EMBEDDINGS, model.feature_mode)
        self.assertEqual(3, model.external_embedding_dimension)
        self.assertEqual("fast_visual", prediction.nearest_experiences[0][0])

    def test_external_embedding_scene_latent_uses_cached_vector_directly(self) -> None:
        frames = attach_external_embedding_cache(
            [
                sample_frame("slow_visual", step=1.0),
                sample_frame("fast_visual", step=1.0),
                sample_frame("motion_reference", step=2.0),
            ],
            {
                "slow_visual": [0.0, 0.0, 0.0],
                "fast_visual": [1.0, 1.0, 1.0],
                "motion_reference": [0.45, 0.45, 0.45],
            },
        )
        model = fit_world_model(
            frames,
            latent_dim=2,
            feature_mode=FEATURE_MODE_EXTERNAL_EMBEDDINGS,
            latent_source=LATENT_SOURCE_SCENE,
        )
        query = attach_external_embedding_cache([sample_frame("query", step=1.0)], {"query": [1.0, 1.0, 1.0]})[0]

        prediction = model.predict_frame(query)
        loaded = LearnedWorldModel.from_dict(model.to_dict())

        self.assertEqual(LATENT_SOURCE_SCENE, model.latent_source)
        self.assertEqual("fast_visual", prediction.nearest_experiences[0][0])
        self.assertEqual(LATENT_SOURCE_SCENE, loaded.latent_source)

    def test_model_round_trips_through_json(self) -> None:
        model = fit_world_model(
            [sample_frame("a", step=1.0), sample_frame("b", step=2.0)],
            latent_dim=2,
        )

        loaded = LearnedWorldModel.from_dict(model.to_dict())

        self.assertEqual(model.train_rows, loaded.train_rows)
        self.assertEqual(model.latent_dim, loaded.latent_dim)
        self.assertEqual(
            [experience.frame_name for experience in model.experiences],
            [experience.frame_name for experience in loaded.experiences],
        )

    def test_external_embedding_model_requires_dimension_when_loaded(self) -> None:
        model = fit_world_model(
            attach_external_embedding_cache(
                [sample_frame("a", step=1.0), sample_frame("b", step=2.0)],
                {"a": [1.0, 0.0], "b": [0.0, 1.0]},
            ),
            latent_dim=2,
            feature_mode=FEATURE_MODE_EXTERNAL_EMBEDDINGS,
        )
        payload = model.to_dict()
        del payload["external_embedding_dimension"]

        with self.assertRaisesRegex(ValueError, "external_embedding_dimension"):
            LearnedWorldModel.from_dict(payload)

    def test_save_load_and_candidate_payloads_are_submission_compatible(self) -> None:
        model = fit_world_model(
            [sample_frame("a", step=1.0), sample_frame("b", step=2.0), sample_frame("c", lateral=2.0, intent=2)],
            latent_dim=2,
        )
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "world_model.json"
            model.save(path)
            loaded = LearnedWorldModel.load(path)
            payloads = loaded.candidate_payloads([sample_frame("eval", step=1.0)])

        self.assertEqual(1, len(payloads))
        self.assertEqual("world_ego_imagined_future", payloads[0]["candidate_name"])
        self.assertEqual(20, len(payloads[0]["trajectory_20wp_4hz"]))
        self.assertEqual(2, len(payloads[0]["world_embedding"]))
        json.dumps(payloads[0])

    def test_candidate_payloads_can_emit_memory_neighbors(self) -> None:
        model = fit_world_model(
            [sample_frame("a", step=1.0), sample_frame("b", step=2.0), sample_frame("c", lateral=2.0, intent=2)],
            latent_dim=2,
        )

        payloads = model.candidate_payloads([sample_frame("eval", step=1.0)], memory_top_k=2)

        self.assertEqual(3, len(payloads))
        self.assertEqual(
            ["world_ego_imagined_future", "world_ego_memory_neighbor_1", "world_ego_memory_neighbor_2"],
            [row["candidate_name"] for row in payloads],
        )
        self.assertEqual(20, len(payloads[1]["trajectory_20wp_4hz"]))

    def test_candidate_payloads_can_gate_by_nearest_experience_distance(self) -> None:
        model = fit_world_model(
            [sample_frame("a", step=1.0), sample_frame("b", step=2.0), sample_frame("c", lateral=2.0, intent=2)],
            latent_dim=2,
        )

        payloads = model.candidate_payloads(
            [sample_frame("far", step=10.0)],
            max_neighbor_distance=0.01,
        )

        self.assertEqual([], payloads)

    def test_rejects_empty_or_invalid_latent_dimension(self) -> None:
        with self.assertRaises(ValueError):
            fit_world_model([])
        with self.assertRaises(ValueError):
            fit_world_model([sample_frame("a")], latent_dim=0)

    def test_latent_predictive_world_prior_scores_plausible_future_lower(self) -> None:
        prior = fit_latent_predictive_world_prior(
            [
                sample_frame("straight_slow", step=0.5),
                sample_frame("straight", step=1.0),
                sample_frame("left", step=1.0, lateral=4.0, intent=2),
                sample_frame("right", step=1.0, lateral=-4.0, intent=3),
            ],
            latent_dim=3,
            ridge=0.1,
        )
        frame = sample_frame("query_left", step=1.0, lateral=4.0, intent=2)
        plausible = frame.future_trajectory
        implausible = sample_frame("query_wrong", step=1.0, lateral=-4.0, intent=3).future_trajectory

        plausible_features = prior.candidate_features(frame, plausible)
        implausible_features = prior.candidate_features(frame, implausible)

        self.assertLess(
            plausible_features["world_prior_constraint_cost"],
            implausible_features["world_prior_constraint_cost"],
        )
        self.assertLess(
            plausible_features["world_prior_latent_error"],
            implausible_features["world_prior_latent_error"],
        )

    def test_latent_predictive_world_prior_round_trips(self) -> None:
        prior = fit_latent_predictive_world_prior(
            [sample_frame("a", step=1.0), sample_frame("b", step=2.0), sample_frame("c", lateral=2.0, intent=2)],
            latent_dim=2,
        )

        loaded = LatentPredictiveWorldPrior.from_dict(prior.to_dict())

        self.assertEqual(prior.latent_dim, loaded.latent_dim)
        self.assertEqual(prior.feature_mode, loaded.feature_mode)
        self.assertEqual(prior.latent_source, loaded.latent_source)
        self.assertEqual(prior.train_rows, loaded.train_rows)

    def test_latent_predictive_world_prior_can_train_with_mixed_context_masks(self) -> None:
        prior = fit_latent_predictive_world_prior(
            [sample_frame("a", step=1.0), sample_frame("b", step=2.0), sample_frame("c", lateral=2.0, intent=2)],
            latent_dim=2,
            context_mask_mode="mixed",
            seed=7,
        )

        self.assertEqual("mixed", prior.context_mask_mode)
        self.assertGreater(prior.train_rows, 3)


if __name__ == "__main__":
    unittest.main()
