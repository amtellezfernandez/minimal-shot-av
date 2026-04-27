from __future__ import annotations

import unittest

from minimal_shot_av.model.wod_e2e import WodCameraImage, WodE2EPreferenceFrame
from minimal_shot_av.model.wod_ranker import candidate_ranker_row, candidate_source_family, selector_numeric_features


class WodRankerFeatureTests(unittest.TestCase):
    def test_contextual_row_marks_source_speed_and_intent(self) -> None:
        frame = WodE2EPreferenceFrame(
            frame_name="segment-100",
            past_trajectory=[(-1.0, 0.0), (0.0, 0.0)],
            future_trajectory=[(float(index), 0.0) for index in range(20)],
            intent=2,
            init_speed_mps=3.0,
            references=[],
        )

        row = candidate_ranker_row(
            frame=frame,
            trajectory=[(float(index), 0.0) for index in range(1, 21)],
            candidate_name="temporal_ridge_mean",
            candidate_index=4,
            source="wod_temporal_summary_trajectory_non_text",
        )

        features = row["features"]
        self.assertEqual("temporal", row["source"])
        self.assertEqual(1.0, features["source_temporal"])
        self.assertEqual(1.0, features["speed_bin_slow"])
        self.assertEqual(1.0, features["intent_2"])
        self.assertEqual(1.0, features["source_temporal_x_speed_bin_slow"])
        self.assertEqual(1.0, features["source_temporal_x_intent_2"])

    def test_selector_feature_modes_are_shared(self) -> None:
        contextual = selector_numeric_features("contextual")
        intent_contextual = selector_numeric_features("intent_contextual")
        world_contextual = selector_numeric_features("world_contextual")
        external_contextual = selector_numeric_features("external_contextual")
        camera_contextual = selector_numeric_features("camera_contextual")
        image_contextual = selector_numeric_features("image_contextual")
        squared = selector_numeric_features("squared")

        self.assertIn("source_temporal_x_speed_bin_slow", contextual)
        self.assertIn("source_temporal_x_intent_2", intent_contextual)
        self.assertNotIn("source_temporal_x_speed_bin_slow", intent_contextual)
        self.assertIn("camera_front_present", camera_contextual)
        self.assertIn("camera_payload_bytes_total_log", camera_contextual)
        self.assertIn("source_temporal_x_camera_front_present", camera_contextual)
        self.assertIn("camera_front_luma_mean", image_contextual)
        self.assertIn("source_temporal_x_camera_front_luma_mean", image_contextual)
        self.assertIn("source_world", contextual)
        self.assertIn("source_scene", contextual)
        self.assertIn("world_nearest_distance_log", world_contextual)
        self.assertIn("source_world_x_world_nearest_distance_log", world_contextual)
        self.assertIn("external_embedding_00", external_contextual)
        self.assertIn("source_scene_x_external_embedding_00", external_contextual)
        self.assertIn("source_temporal_x_external_embedding_00", external_contextual)
        self.assertIn("sq_source_temporal", squared)

    def test_external_contextual_row_marks_embedding_features(self) -> None:
        frame = WodE2EPreferenceFrame(
            frame_name="segment-100",
            past_trajectory=[(-1.0, 0.0), (0.0, 0.0)],
            future_trajectory=[(float(index), 0.0) for index in range(20)],
            intent=1,
            init_speed_mps=8.0,
            references=[],
            external_embedding=[0.25, -0.5],
        )

        row = candidate_ranker_row(
            frame=frame,
            trajectory=[(float(index), 0.0) for index in range(1, 21)],
            candidate_name="temporal_ridge_mean",
            candidate_index=0,
            source="wod_temporal_summary_trajectory_non_text",
        )
        features = row["features"]

        self.assertEqual(0.25, features["external_embedding_00"])
        self.assertEqual(-0.5, features["external_embedding_01"])
        self.assertEqual(0.0, features["external_embedding_02"])
        self.assertEqual(0.25, features["source_temporal_x_external_embedding_00"])
        self.assertEqual(0.0, features["source_kinematic_x_external_embedding_00"])

    def test_candidate_source_family_maps_submission_sources(self) -> None:
        self.assertEqual(
            "temporal",
            candidate_source_family(source="wod_temporal_summary_trajectory_non_text", candidate_name="ridge_mean"),
        )
        self.assertEqual(
            "learned",
            candidate_source_family(source="wod_ridge_trajectory_non_text", candidate_name="ridge_mean"),
        )
        self.assertEqual(
            "world",
            candidate_source_family(source="wod_learned_world_model", candidate_name="world_scene_imagined_future"),
        )
        self.assertEqual(
            "scene",
            candidate_source_family(source="scene", candidate_name="scene_aux_ridge_scene_mean"),
        )

    def test_camera_contextual_row_marks_camera_payload_features(self) -> None:
        frame = WodE2EPreferenceFrame(
            frame_name="segment-100",
            past_trajectory=[(-1.0, 0.0), (0.0, 0.0)],
            future_trajectory=[(float(index), 0.0) for index in range(20)],
            intent=1,
            init_speed_mps=8.0,
            references=[],
            camera_images=[
                WodCameraImage(name="FRONT", jpeg=b"front-bytes"),
                WodCameraImage(name="REAR", jpeg=b"rear"),
            ],
        )

        row = candidate_ranker_row(
            frame=frame,
            trajectory=[(float(index), 0.0) for index in range(1, 21)],
            candidate_name="ridge_mean",
            candidate_index=0,
            source="wod_ridge_trajectory_non_text",
        )
        features = row["features"]

        self.assertEqual(2.0, features["camera_count"])
        self.assertEqual(1.0, features["camera_front_present"])
        self.assertEqual(1.0, features["camera_rear_present"])
        self.assertEqual(0.0, features["camera_side_left_present"])
        self.assertEqual(1.0, features["source_learned_x_camera_front_present"])
        self.assertGreater(features["camera_front_bytes_log"], features["camera_rear_bytes_log"])

    def test_image_contextual_row_marks_decoded_image_features(self) -> None:
        try:
            jpeg = _tiny_gray_jpeg()
        except ModuleNotFoundError:
            self.skipTest("Pillow is not installed")
        frame = WodE2EPreferenceFrame(
            frame_name="segment-100",
            past_trajectory=[(-1.0, 0.0), (0.0, 0.0)],
            future_trajectory=[(float(index), 0.0) for index in range(20)],
            intent=1,
            init_speed_mps=8.0,
            references=[],
            camera_images=[WodCameraImage(name="FRONT", jpeg=jpeg)],
        )

        row = candidate_ranker_row(
            frame=frame,
            trajectory=[(float(index), 0.0) for index in range(1, 21)],
            candidate_name="ridge_mean",
            candidate_index=0,
            source="wod_ridge_trajectory_non_text",
        )
        features = row["features"]

        self.assertGreater(features["camera_front_luma_mean"], 0.0)
        self.assertGreaterEqual(features["camera_front_luma_std"], 0.0)
        self.assertGreaterEqual(features["camera_front_edge_mean"], 0.0)
        self.assertGreater(features["source_learned_x_camera_front_luma_mean"], 0.0)


def _tiny_gray_jpeg() -> bytes:
    from io import BytesIO

    from PIL import Image

    buffer = BytesIO()
    image = Image.new("L", (2, 2))
    image.putdata([0, 64, 128, 255])
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


if __name__ == "__main__":
    unittest.main()
