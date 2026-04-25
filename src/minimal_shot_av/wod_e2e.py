from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterator, Sequence

from .spotlight_reflex import RfsReference, Trajectory


@dataclass(frozen=True)
class WodE2EPreferenceFrame:
    frame_name: str
    past_trajectory: Trajectory
    future_trajectory: Trajectory
    intent: int
    init_speed_mps: float
    references: list[RfsReference]


def load_preference_frames(
    val_dir: str | Path,
    *,
    max_shards: int | None = None,
    max_records: int | None = None,
) -> Iterator[WodE2EPreferenceFrame]:
    """Yield validation frames with valid WOD-E2E rater preference labels."""

    tf, wod_e2ed_pb2 = _import_official_parser()
    shards = _validation_shards(Path(val_dir), max_shards=max_shards)
    records_seen = 0

    for shard in shards:
        dataset = tf.data.TFRecordDataset([str(shard)], compression_type="")
        for record in dataset:
            records_seen += 1
            if max_records is not None and records_seen > max_records:
                return

            frame = wod_e2ed_pb2.E2EDFrame()
            frame.ParseFromString(record.numpy())
            parsed = preference_frame_from_proto(frame)
            if parsed is not None:
                yield parsed


def preference_frame_from_proto(frame: object) -> WodE2EPreferenceFrame | None:
    """Convert an official `E2EDFrame` proto into verifier-ready references."""

    future_trajectory = trajectory_from_states(frame.future_states)
    if len(future_trajectory) != 20:
        return None

    references: list[RfsReference] = []
    for index, preference in enumerate(frame.preference_trajectories):
        if not _has_valid_preference_score(preference):
            continue
        try:
            trajectory = align_preference_trajectory(
                trajectory_from_states(preference),
                target_len=len(future_trajectory),
            )
        except ValueError:
            continue
        references.append(
            RfsReference(
                label=f"wod_preference_{index}",
                trajectory=trajectory,
                score=float(preference.preference_score),
            )
        )

    if not references:
        return None

    return WodE2EPreferenceFrame(
        frame_name=str(frame.frame.context.name),
        past_trajectory=trajectory_from_states(frame.past_states),
        future_trajectory=future_trajectory,
        intent=int(frame.intent),
        init_speed_mps=init_speed_from_states(frame.past_states),
        references=references,
    )


def trajectory_from_states(states: object) -> Trajectory:
    return [(float(x), float(y)) for x, y in zip(states.pos_x, states.pos_y)]


def align_preference_trajectory(
    trajectory: Sequence[tuple[float, float]],
    *,
    target_len: int = 20,
) -> Trajectory:
    """Align a rater trajectory exactly like the official RFS utility.

    The official `process_rater_specified_trajectories` truncates trajectories
    longer than the target waypoint count and pads shorter trajectories by
    repeating their final waypoint.
    """

    points = [(float(x), float(y)) for x, y in trajectory]
    if not points:
        raise ValueError("preference trajectory is empty")
    if len(points) >= target_len:
        return points[:target_len]
    return points + [points[-1]] * (target_len - len(points))


def init_speed_from_states(states: object) -> float:
    if len(states.vel_x) and len(states.vel_y):
        return math.hypot(float(states.vel_x[-1]), float(states.vel_y[-1]))
    if len(states.pos_x) >= 2 and len(states.pos_y) >= 2:
        dx = float(states.pos_x[-1]) - float(states.pos_x[-2])
        dy = float(states.pos_y[-1]) - float(states.pos_y[-2])
        return math.hypot(dx, dy) * 4.0
    return 0.0


def _validation_shards(val_dir: Path, *, max_shards: int | None) -> list[Path]:
    shards = sorted(val_dir.glob("val_*.tfrecord-*"))
    if not shards:
        raise FileNotFoundError(f"no WOD-E2E validation shards found under {val_dir}")
    if max_shards is not None:
        return shards[:max_shards]
    return shards


def _has_valid_preference_score(preference: object) -> bool:
    has_field = getattr(preference, "HasField", None)
    if callable(has_field) and not has_field("preference_score"):
        return False
    return 0.0 <= float(preference.preference_score) <= 10.0


def _import_official_parser():
    try:
        import tensorflow as tf
        from waymo_open_dataset.protos import end_to_end_driving_data_pb2 as wod_e2ed_pb2
    except ImportError as exc:
        raise ImportError(
            "WOD-E2E parsing requires the local parser environment. Run with "
            "`PYTHONPATH=.wod-protos .venv-wod/bin/python`, after following "
            "`docs/wod-e2e-parser-setup.md`."
        ) from exc
    return tf, wod_e2ed_pb2
