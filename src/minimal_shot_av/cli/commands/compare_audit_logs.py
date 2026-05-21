from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from minimal_shot_av.audit import load_audit_log


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
    return {
        "left": {
            "source": left_manifest.get("source"),
            "frame_count": len(left_frames),
            "summary": _frame_summary(left_frames),
        },
        "right": {
            "source": right_manifest.get("source"),
            "frame_count": len(right_frames),
            "summary": _frame_summary(right_frames),
        },
        "delta": _summary_delta(_frame_summary(left_frames), _frame_summary(right_frames)),
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


def _finite(value: float) -> float:
    return value if math.isfinite(value) else 0.0


if __name__ == "__main__":
    main()
