from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from minimal_shot_av.audit import load_audit_log
from minimal_shot_av.audit.review import frame_bookmarks, paired_bookmarks


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two normalized audit logs.")
    parser.add_argument("left", type=Path, help="First audit log directory")
    parser.add_argument("right", type=Path, help="Second audit log directory")
    args = parser.parse_args()

    left_manifest, left_frames = load_audit_log(args.left)
    right_manifest, right_frames = load_audit_log(args.right)
    report = compare_audit_logs(left_manifest, left_frames, right_manifest, right_frames)
    print(json.dumps(report, indent=2))


def compare_audit_logs(
    left_manifest: dict[str, Any],
    left_frames: list[dict[str, Any]],
    right_manifest: dict[str, Any],
    right_frames: list[dict[str, Any]],
) -> dict[str, Any]:
    left_summary = _frame_summary(left_frames)
    right_summary = _frame_summary(right_frames)
    alignment_mode = _alignment_mode(left_frames, right_frames)
    left_bookmarks = frame_bookmarks(left_frames)
    right_bookmarks = frame_bookmarks(right_frames)
    return {
        "left": {
            "source": left_manifest.get("source"),
            "frame_count": len(left_frames),
            "summary": left_summary,
            "bookmarks": left_bookmarks,
        },
        "right": {
            "source": right_manifest.get("source"),
            "frame_count": len(right_frames),
            "summary": right_summary,
            "bookmarks": right_bookmarks,
        },
        "delta": _summary_delta(left_summary, right_summary),
        "alignment_mode": alignment_mode,
        "aligned_samples": _aligned_samples(left_frames, right_frames, mode=alignment_mode),
        "paired_bookmarks": paired_bookmarks(left_bookmarks, right_bookmarks),
    }


def _frame_summary(frames: list[dict[str, Any]]) -> dict[str, Any]:
    if not frames:
        return {
            "min_clearance": math.inf,
            "max_collision_risk": 0.0,
            "avg_speed": 0.0,
            "action_modes": {},
        }
    min_clearance = math.inf
    max_collision_risk = 0.0
    speed_total = 0.0
    action_modes: dict[str, int] = {}
    for frame in frames:
        step = frame.get("step", {})
        min_clearance = min(min_clearance, float(step.get("min_obstacle_distance", math.inf)))
        max_collision_risk = max(max_collision_risk, float(step.get("collision_risk", 0.0) or 0.0))
        speed_total += float(frame.get("ego", {}).get("speed", 0.0) or 0.0)
        mode = str(step.get("action_mode", ""))
        if mode:
            action_modes[mode] = action_modes.get(mode, 0) + 1
    return {
        "min_clearance": min_clearance,
        "max_collision_risk": max_collision_risk,
        "avg_speed": speed_total / len(frames),
        "action_modes": action_modes,
    }


def _summary_delta(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    return {
        "min_clearance_delta": _finite(left["min_clearance"]) - _finite(right["min_clearance"]),
        "max_collision_risk_delta": float(left["max_collision_risk"]) - float(right["max_collision_risk"]),
        "avg_speed_delta": float(left["avg_speed"]) - float(right["avg_speed"]),
    }


def _aligned_samples(
    left_frames: list[dict[str, Any]],
    right_frames: list[dict[str, Any]],
    *,
    mode: str,
    sample_count: int = 8,
) -> list[dict[str, Any]]:
    if not left_frames or not right_frames:
        return []
    if mode == "timestamp":
        return _timestamp_aligned_samples(left_frames, right_frames, sample_count=sample_count)
    target_count = min(sample_count, max(len(left_frames), len(right_frames)))
    samples: list[dict[str, Any]] = []
    for sample_idx in range(target_count):
        fraction = sample_idx / max(target_count - 1, 1)
        left_frame = left_frames[_fractional_index(len(left_frames), fraction)]
        right_frame = right_frames[_fractional_index(len(right_frames), fraction)]
        samples.append(
            {
                "sample_idx": sample_idx,
                "progress": round(fraction, 3),
                "left": _sample_frame(left_frame),
                "right": _sample_frame(right_frame),
            }
        )
    return samples


def _timestamp_aligned_samples(
    left_frames: list[dict[str, Any]],
    right_frames: list[dict[str, Any]],
    *,
    sample_count: int,
) -> list[dict[str, Any]]:
    left_times = [float(frame.get("timestamp_s", 0.0) or 0.0) for frame in left_frames]
    right_times = [float(frame.get("timestamp_s", 0.0) or 0.0) for frame in right_frames]
    min_time = max(left_times[0], right_times[0])
    max_time = min(left_times[-1], right_times[-1])
    if max_time <= min_time:
        return _aligned_samples(left_frames, right_frames, mode="progress", sample_count=sample_count)
    target_count = min(sample_count, max(len(left_frames), len(right_frames)))
    samples: list[dict[str, Any]] = []
    for sample_idx in range(target_count):
        fraction = sample_idx / max(target_count - 1, 1)
        target_time = min_time + fraction * (max_time - min_time)
        left_frame = _nearest_timestamp_frame(left_frames, target_time)
        right_frame = _nearest_timestamp_frame(right_frames, target_time)
        samples.append(
            {
                "sample_idx": sample_idx,
                "progress": round(fraction, 3),
                "target_timestamp_s": round(target_time, 3),
                "timestamp_delta_s": round(abs(float(left_frame.get("timestamp_s", 0.0)) - float(right_frame.get("timestamp_s", 0.0))), 3),
                "left": _sample_frame(left_frame),
                "right": _sample_frame(right_frame),
            }
        )
    return samples


def _fractional_index(length: int, fraction: float) -> int:
    if length <= 1:
        return 0
    return min(length - 1, max(0, int(round(fraction * (length - 1)))))


def _nearest_timestamp_frame(frames: list[dict[str, Any]], timestamp_s: float) -> dict[str, Any]:
    return min(frames, key=lambda frame: abs(float(frame.get("timestamp_s", 0.0) or 0.0) - timestamp_s))


def _alignment_mode(left_frames: list[dict[str, Any]], right_frames: list[dict[str, Any]]) -> str:
    if _has_dense_timestamps(left_frames) and _has_dense_timestamps(right_frames):
        return "timestamp"
    return "progress"


def _has_dense_timestamps(frames: list[dict[str, Any]]) -> bool:
    if len(frames) < 2:
        return False
    timestamps = [float(frame.get("timestamp_s", 0.0) or 0.0) for frame in frames]
    return all(later > earlier for earlier, later in zip(timestamps, timestamps[1:]))


def _sample_frame(frame: dict[str, Any]) -> dict[str, Any]:
    step = frame.get("step", {})
    ego = frame.get("ego", {})
    return {
        "frame_idx": int(frame.get("frame_idx", 0)),
        "timestamp_s": float(frame.get("timestamp_s", 0.0) or 0.0),
        "speed": float(ego.get("speed", 0.0) or 0.0),
        "action_mode": step.get("action_mode"),
        "min_clearance": _finite(float(step.get("min_obstacle_distance", math.inf) or math.inf)),
        "collision_risk": float(step.get("collision_risk", 0.0) or 0.0),
        "lane_error": float(step.get("lane_error", 0.0) or 0.0),
        "media_count": len(frame.get("media", [])) if isinstance(frame.get("media"), list) else 0,
    }


def _finite(value: float) -> float:
    return value if math.isfinite(value) else 0.0


if __name__ == "__main__":
    main()
