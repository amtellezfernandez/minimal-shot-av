from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_audit_log(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads((root / "manifest.json").read_text())
    frames: list[dict[str, Any]] = []
    with (root / "frames.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                frames.append(json.loads(line))
    return manifest, frames


def summarize_audit_log(root: Path) -> dict[str, Any]:
    manifest, frames = load_audit_log(root)
    bookmarks = _frame_bookmarks(frames)
    return {
        "manifest": manifest,
        "frame_count": len(frames),
        "media_frame_count": sum(1 for frame in frames if frame.get("media")),
        "bookmark_count": len(bookmarks),
        "bookmark_index": bookmarks,
    }


def view_audit_log_with_rerun(root: Path, *, spawn: bool = False) -> dict[str, Any]:
    manifest, frames = load_audit_log(root)
    try:
        import rerun as rr
    except ImportError as exc:  # pragma: no cover - depends on optional package.
        raise RuntimeError("rerun is not installed; install with `pip install rerun-sdk`") from exc

    rr.init(f"minimal-shot-av:{manifest['scenario_cluster']}", spawn=spawn)
    if frames:
        first = frames[0]
        lane_center = first["route"]["lane_center"]
        rr.log("world/route/center", rr.LineStrips2D([lane_center]))
        rr.log("world/route/start", rr.Points2D([first["route"]["start"]], colors=[[0, 150, 150]], radii=[6.0]))
        rr.log("world/route/goal", rr.Points2D([first["route"]["goal"]], colors=[[238, 155, 0]], radii=[6.0]))

    for frame in frames:
        rr.set_time_sequence("frame", int(frame["frame_idx"]))
        rr.set_time_seconds("sim_time", float(frame["timestamp_s"]))
        ego = frame["ego"]
        rr.log("world/ego", rr.Points2D([[ego["x"], ego["y"]]], colors=[[0, 95, 115]], radii=[5.0]))
        if frame["active_obstacles"]:
            rr.log(
                "world/obstacles",
                rr.Points2D(
                    [[item["x"], item["y"]] for item in frame["active_obstacles"]],
                    colors=[[163, 61, 43] for _ in frame["active_obstacles"]],
                    radii=[max(2.0, float(item["radius"]) * 3.0) for item in frame["active_obstacles"]],
                ),
            )
        step = frame["step"]
        rr.log("metrics/min_clearance", rr.Scalars([float(step.get("min_obstacle_distance", 0.0))]))
        rr.log("metrics/collision_risk", rr.Scalars([float(step.get("collision_risk", 0.0))]))
        rr.log("metrics/lane_error", rr.Scalars([float(step.get("lane_error", 0.0))]))
        rr.log("planner/action_mode", rr.TextLog(str(step.get("action_mode", ""))))
        for bookmark in _bookmarks_for_frame(frame):
            rr.log("events/bookmarks", rr.TextLog(f"{bookmark['kind']}: {json.dumps(bookmark['detail'], sort_keys=True)}"))
        _log_frame_media(rr, root, frame)
    return manifest


def _frame_bookmarks(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
            bookmarks.append(_bookmark("trigger_activation", frame, {"trigger_regions": new_trigger_keys}))
        previous_trigger_keys = active_trigger_keys

        min_clearance = _as_float(step.get("min_obstacle_distance"), default=float("inf"))
        if min_clearance <= 1.0:
            bookmarks.append(_bookmark("near_miss", frame, {"min_clearance": min_clearance}))

        collision_risk = _as_float(step.get("collision_risk"), default=0.0)
        if collision_risk >= 0.7:
            bookmarks.append(_bookmark("collision_risk_spike", frame, {"collision_risk": collision_risk}))

        lane_error = _as_float(step.get("lane_error"), default=0.0)
        if lane_error >= 1.0:
            bookmarks.append(_bookmark("lane_violation", frame, {"lane_error": lane_error}))

        if _has_intervention(frame):
            bookmarks.append(_bookmark("intervention", frame, {"action_mode": step.get("action_mode")}))

        if _is_low_motion(frame):
            low_motion_streak += 1
        else:
            low_motion_streak = 0
        if bool(step.get("stall")) or (low_motion_streak >= 4 and _as_float(frame.get("ego", {}).get("goal_distance"), default=0.0) > 5.0):
            bookmarks.append(
                _bookmark(
                    "stall_or_deadlock",
                    frame,
                    {
                        "speed": _as_float(frame.get("ego", {}).get("speed"), default=0.0),
                        "goal_distance": _as_float(frame.get("ego", {}).get("goal_distance"), default=0.0),
                        "low_motion_streak": low_motion_streak,
                    },
                )
            )
    return bookmarks


def _bookmarks_for_frame(frame: dict[str, Any]) -> list[dict[str, Any]]:
    return _frame_bookmarks([frame])


def _bookmark(kind: str, frame: dict[str, Any], detail: dict[str, Any]) -> dict[str, Any]:
    step = frame.get("step", {})
    return {
        "kind": kind,
        "frame_idx": int(frame.get("frame_idx", 0)),
        "timestamp_s": float(frame.get("timestamp_s", 0.0) or 0.0),
        "action_mode": step.get("action_mode"),
        "detail": detail,
    }


def _has_intervention(frame: dict[str, Any]) -> bool:
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


def _is_low_motion(frame: dict[str, Any]) -> bool:
    ego = frame.get("ego", {})
    return _as_float(ego.get("speed"), default=0.0) <= 0.1


def _log_frame_media(rr, audit_root: Path, frame: dict[str, Any]) -> None:  # pragma: no cover - depends on optional packages.
    media_refs = frame.get("media", [])
    if not isinstance(media_refs, list):
        return
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        return
    for item in media_refs:
        if not isinstance(item, dict):
            continue
        path_value = item.get("path")
        if not isinstance(path_value, str) or not path_value:
            continue
        media_path = Path(path_value)
        if not media_path.is_absolute():
            media_path = audit_root / media_path
        if not media_path.is_file():
            continue
        try:
            with Image.open(media_path) as image:
                rgb = image.convert("RGB")
                rr.log(f"media/{item.get('label', 'frame')}", rr.Image(np.asarray(rgb)))
        except Exception:
            continue


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
