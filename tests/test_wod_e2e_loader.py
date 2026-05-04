from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minimal_shot_av.model.wod_e2e import (
    _shards_from_glob,
    _validation_shards,
    align_preference_trajectory,
    camera_images_from_frame,
    init_speed_from_states,
    load_preference_frames,
    preference_frame_from_proto,
    trajectory_from_states,
)


@dataclass
class FakeStates:
    pos_x: list[float]
    pos_y: list[float]
    vel_x: list[float] = field(default_factory=list)
    vel_y: list[float] = field(default_factory=list)
    preference_score: float = -1.0
    has_score: bool = False

    def HasField(self, name: str) -> bool:
        if name != "preference_score":
            raise ValueError(name)
        return self.has_score


@dataclass
class FakeContext:
    name: str


@dataclass
class FakeFramePayload:
    context: FakeContext
    images: list[object] = field(default_factory=list)


@dataclass
class FakeImage:
    name: int
    image: bytes


@dataclass
class FakeE2EDFrame:
    frame: FakeFramePayload
    past_states: FakeStates
    future_states: FakeStates
    intent: int
    preference_trajectories: list[FakeStates]


class WodE2ELoaderTests(unittest.TestCase):
    def test_align_preference_trajectory_truncates_like_official_metric(self) -> None:
        trajectory = [(float(i), float(-i)) for i in range(21)]
        aligned = align_preference_trajectory(trajectory)
        self.assertEqual(len(aligned), 20)
        self.assertEqual(aligned[0], (0.0, 0.0))
        self.assertEqual(aligned[-1], (19.0, -19.0))

    def test_align_preference_trajectory_keeps_twenty_point_input(self) -> None:
        trajectory = [(float(i), 0.0) for i in range(20)]
        self.assertEqual(align_preference_trajectory(trajectory), trajectory)

    def test_align_preference_trajectory_pads_short_input_like_official_metric(self) -> None:
        aligned = align_preference_trajectory([(1.0, 2.0)] * 19)
        self.assertEqual(len(aligned), 20)
        self.assertEqual(aligned[-1], (1.0, 2.0))

    def test_align_preference_trajectory_rejects_empty_input(self) -> None:
        with self.assertRaises(ValueError):
            align_preference_trajectory([])

    def test_init_speed_uses_last_velocity_when_available(self) -> None:
        states = FakeStates([0.0], [0.0], vel_x=[3.0], vel_y=[4.0])
        self.assertEqual(init_speed_from_states(states), 5.0)

    def test_init_speed_falls_back_to_position_delta_at_4hz(self) -> None:
        states = FakeStates([0.0, 0.75], [0.0, 1.0])
        self.assertEqual(init_speed_from_states(states), 5.0)

    def test_preference_frame_from_proto_skips_invalid_scores(self) -> None:
        future = FakeStates([float(i) for i in range(1, 21)], [0.0] * 20)
        past = FakeStates([0.0, 0.75], [0.0, 1.0])
        valid_preference = FakeStates(
            [float(i) for i in range(21)],
            [float(i * 0.1) for i in range(21)],
            preference_score=8.5,
            has_score=True,
        )
        invalid_preference = FakeStates(
            [float(i) for i in range(21)],
            [0.0] * 21,
            preference_score=-1.0,
            has_score=True,
        )
        missing_score_preference = FakeStates(
            [float(i) for i in range(21)],
            [0.0] * 21,
            preference_score=9.0,
            has_score=False,
        )
        frame = FakeE2EDFrame(
            frame=FakeFramePayload(FakeContext("segment-150")),
            past_states=past,
            future_states=future,
            intent=2,
            preference_trajectories=[
                invalid_preference,
                valid_preference,
                missing_score_preference,
            ],
        )

        parsed = preference_frame_from_proto(frame)

        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed.frame_name, "segment-150")
        self.assertEqual(parsed.intent, 2)
        self.assertEqual(parsed.init_speed_mps, 5.0)
        self.assertEqual(len(parsed.future_trajectory), 20)
        self.assertEqual(len(parsed.references), 1)
        self.assertEqual(parsed.references[0].label, "wod_preference_1")
        self.assertEqual(parsed.references[0].score, 8.5)
        self.assertEqual(len(parsed.references[0].trajectory), 20)
        self.assertEqual(parsed.references[0].trajectory[0], (0.0, 0.0))

    def test_preference_frame_from_proto_can_skip_camera_images(self) -> None:
        future = FakeStates([float(i) for i in range(1, 21)], [0.0] * 20)
        preference = FakeStates([float(i) for i in range(20)], [0.0] * 20, preference_score=8.0, has_score=True)
        frame = FakeE2EDFrame(
            frame=FakeFramePayload(FakeContext("segment"), images=[FakeImage(1, b"front")]),
            past_states=FakeStates([0.0, 1.0], [0.0, 0.0]),
            future_states=future,
            intent=1,
            preference_trajectories=[preference],
        )

        parsed = preference_frame_from_proto(frame, include_camera_images=False)

        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed.camera_images, [])

    def test_preference_frame_from_proto_accepts_unlabeled_test_frame(self) -> None:
        frame = FakeE2EDFrame(
            frame=FakeFramePayload(FakeContext("test-segment")),
            past_states=FakeStates([0.0, 1.0], [0.0, 0.0], vel_x=[4.0], vel_y=[0.0]),
            future_states=FakeStates([], []),
            intent=3,
            preference_trajectories=[],
        )

        parsed = preference_frame_from_proto(frame, require_preferences=False, include_camera_images=False)

        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual("test-segment", parsed.frame_name)
        self.assertEqual([], parsed.future_trajectory)
        self.assertEqual([], parsed.references)
        self.assertEqual(4.0, parsed.init_speed_mps)

    def test_preference_frame_from_proto_skips_empty_valid_scored_preference(self) -> None:
        future = FakeStates([float(i) for i in range(1, 21)], [0.0] * 20)
        past = FakeStates([0.0, 0.75], [0.0, 1.0])
        empty_preference = FakeStates(
            [],
            [],
            preference_score=8.0,
            has_score=True,
        )
        valid_preference = FakeStates(
            [float(i) for i in range(20)],
            [0.0] * 20,
            preference_score=7.0,
            has_score=True,
        )
        frame = FakeE2EDFrame(
            frame=FakeFramePayload(FakeContext("segment-151")),
            past_states=past,
            future_states=future,
            intent=1,
            preference_trajectories=[empty_preference, valid_preference],
        )

        parsed = preference_frame_from_proto(frame)

        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(len(parsed.references), 1)
        self.assertEqual(parsed.references[0].label, "wod_preference_1")

    def test_trajectory_from_states_pairs_xy_values(self) -> None:
        states = FakeStates([1, 2, 3], [4, 5])
        self.assertEqual(trajectory_from_states(states), [(1.0, 4.0), (2.0, 5.0)])

    def test_camera_images_from_frame_exports_named_jpegs(self) -> None:
        frame = FakeE2EDFrame(
            frame=FakeFramePayload(FakeContext("segment"), images=[FakeImage(1, b"front"), FakeImage(7, b"rear")]),
            past_states=FakeStates([], []),
            future_states=FakeStates([], []),
            intent=0,
            preference_trajectories=[],
        )

        images = camera_images_from_frame(frame)

        self.assertEqual([image.name for image in images], ["FRONT", "REAR"])
        self.assertEqual(images[0].jpeg, b"front")

    def test_validation_shards_supports_windowed_processing(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for index in range(5):
                (root / f"train_000.tfrecord-{index:05d}-of-00005").write_bytes(b"")

            shards = _validation_shards(root, shard_start=2, max_shards=2)

            self.assertEqual(
                [f"train_000.tfrecord-{index:05d}-of-00005" for index in (2, 3)],
                [shard.name for shard in shards],
            )

    def test_validation_shards_rejects_negative_start(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "val_000.tfrecord-00000-of-00001").write_bytes(b"")

            with self.assertRaises(ValueError):
                _validation_shards(root, shard_start=-1, max_shards=None)

    def test_shards_from_glob_supports_gcs_style_paths(self) -> None:
        paths = [
            "gs://bucket/training.tfrecord-00002-of-00003",
            "gs://bucket/training.tfrecord-00000-of-00003",
            "gs://bucket/training.tfrecord-00001-of-00003",
        ]

        shards = _shards_from_glob(
            "gs://bucket/training*.tfrecord-*",
            glob_fn=lambda pattern: paths,
            shard_start=1,
            max_shards=1,
        )

        self.assertEqual(["gs://bucket/training.tfrecord-00001-of-00003"], shards)

    def test_record_start_is_validated_before_importing_tensorflow(self) -> None:
        with self.assertRaises(ValueError):
            list(load_preference_frames("unused", record_start=-1))


if __name__ == "__main__":
    unittest.main()
