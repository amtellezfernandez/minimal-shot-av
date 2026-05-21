from __future__ import annotations

import math
from typing import Any


def frame_bookmarks(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    bookmarks: list[dict[str, Any]] = []
    previous_trigger_keys: set[str] = set()
    low_motion_streak = 0
    for frame in frames:
        step = frame.get("step", {})
        trigger_state = frame.get("trigger_state", {})
        active_trigger_keys = {
            str(key)
            for key, window in trigger_state.items()
            if isinstance(window, dict) and window.get("active_from") is not None
        }
        new_trigger_keys = sorted(active_trigger_keys - previous_trigger_keys)
        if new_trigger_keys:
            bookmarks.append(bookmark("trigger_activation", frame, {"trigger_regions": new_trigger_keys}))
        previous_trigger_keys = active_trigger_keys

        min_clearance = as_float(step.get("min_obstacle_distance"), default=math.inf)
        if min_clearance <= 1.0:
            bookmarks.append(bookmark("near_miss", frame, {"min_clearance": min_clearance}))

        collision_risk = as_float(step.get("collision_risk"), default=0.0)
        if collision_risk >= 0.7:
            bookmarks.append(bookmark("collision_risk_spike", frame, {"collision_risk": collision_risk}))

        lane_error = as_float(step.get("lane_error"), default=0.0)
        if lane_error >= 1.0:
            bookmarks.append(bookmark("lane_violation", frame, {"lane_error": lane_error}))

        if has_intervention(frame):
            bookmarks.append(bookmark("intervention", frame, {"action_mode": step.get("action_mode")}))

        if is_low_motion(frame):
            low_motion_streak += 1
        else:
            low_motion_streak = 0
        if bool(step.get("stall")) or (low_motion_streak >= 4 and as_float(frame.get("ego", {}).get("goal_distance"), default=0.0) > 5.0):
            bookmarks.append(
                bookmark(
                    "stall_or_deadlock",
                    frame,
                    {
                        "speed": as_float(frame.get("ego", {}).get("speed"), default=0.0),
                        "goal_distance": as_float(frame.get("ego", {}).get("goal_distance"), default=0.0),
                        "low_motion_streak": low_motion_streak,
                    },
                )
            )
    return bookmarks


def bookmarks_for_frame(frame: dict[str, Any]) -> list[dict[str, Any]]:
    return frame_bookmarks([frame])


def paired_bookmarks(
    left_bookmarks: list[dict[str, Any]],
    right_bookmarks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    right_by_kind: dict[str, list[dict[str, Any]]] = {}
    for item in right_bookmarks:
        right_by_kind.setdefault(str(item.get("kind", "")), []).append(item)
    for items in right_by_kind.values():
        items.sort(key=lambda item: float(item.get("timestamp_s", 0.0) or 0.0))
    for left in left_bookmarks:
        kind = str(left.get("kind", ""))
        candidates = right_by_kind.get(kind, [])
        if not candidates:
            continue
        left_time = float(left.get("timestamp_s", 0.0) or 0.0)
        right = min(candidates, key=lambda item: abs(float(item.get("timestamp_s", 0.0) or 0.0) - left_time))
        pairs.append(
            {
                "kind": kind,
                "left_frame_idx": int(left.get("frame_idx", 0)),
                "right_frame_idx": int(right.get("frame_idx", 0)),
                "left_timestamp_s": left_time,
                "right_timestamp_s": float(right.get("timestamp_s", 0.0) or 0.0),
                "timestamp_delta_s": round(abs(left_time - float(right.get("timestamp_s", 0.0) or 0.0)), 3),
            }
        )
    return pairs


def critical_event_bundle(
    manifest: dict[str, Any],
    frames: list[dict[str, Any]],
    *,
    context_radius: int = 2,
) -> dict[str, Any]:
    bookmarks = frame_bookmarks(frames)
    selected_indices: set[int] = set()
    for bookmark_item in bookmarks:
        frame_idx = int(bookmark_item.get("frame_idx", 0))
        for offset in range(-context_radius, context_radius + 1):
            selected_indices.add(frame_idx + offset)
    selected_frames = [frame for frame in frames if int(frame.get("frame_idx", -1)) in selected_indices]
    return {
        "manifest": manifest,
        "bookmark_count": len(bookmarks),
        "bookmarks": bookmarks,
        "critical_frame_count": len(selected_frames),
        "critical_frames": selected_frames,
    }


def bookmark(kind: str, frame: dict[str, Any], detail: dict[str, Any]) -> dict[str, Any]:
    step = frame.get("step", {})
    return {
        "kind": kind,
        "frame_idx": int(frame.get("frame_idx", 0)),
        "timestamp_s": float(frame.get("timestamp_s", 0.0) or 0.0),
        "action_mode": step.get("action_mode"),
        "detail": detail,
    }


def has_intervention(frame: dict[str, Any]) -> bool:
    step = frame.get("step", {})
    planner = frame.get("planner", {})
    if bool(step.get("intervention")):
        return True
    action_mode = str(step.get("action_mode", "") or "")
    if action_mode and action_mode not in {"maintain", "direct_actor_planner"}:
        return True
    decision_type = str(step.get("decision_type", "") or "")
    if decision_type and decision_type not in {"spotlight_wins", "maintain"}:
        return True
    selection_mode = str(planner.get("selection_mode", "") or "")
    if selection_mode and selection_mode != "hybrid_veto":
        return True
    return False


def is_low_motion(frame: dict[str, Any]) -> bool:
    ego = frame.get("ego", {})
    return as_float(ego.get("speed"), default=0.0) <= 0.1


def as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
