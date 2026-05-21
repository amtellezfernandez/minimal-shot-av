from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from minimal_shot_av.neutral.alpasim_metrics import build_alpasim_evidence
from minimal_shot_av.simulator.alpasim_signal import scenario_from_command
from minimal_shot_av.simulator.environment import route_centerline


def export_alpasim_audit_log(run_dir: Path, output_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    launch_metadata = _load_json_if_exists(run_dir / "launch-metadata.json")
    selection_rows = _load_jsonl_if_exists(run_dir / "driver" / "selection-log.jsonl")
    direct_rows = _load_jsonl_if_exists(run_dir / "driver" / "direct-planner-log.jsonl")
    frames = _build_alpasim_frames(selection_rows, direct_rows)
    metrics_evidence = _safe_metrics_evidence(run_dir)

    manifest = {
        "format_version": 1,
        "source": "alpasim",
        "model": launch_metadata.get("model"),
        "scene_preset": launch_metadata.get("scene_preset"),
        "scene_ids": launch_metadata.get("scene_ids", []),
        "frame_count": len(frames),
        "files": {
            "manifest": "manifest.json",
            "frames": "frames.jsonl",
            "launch_metadata": "launch_metadata.json",
            "metrics_evidence": "metrics_evidence.json",
        },
    }

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (output_dir / "launch_metadata.json").write_text(json.dumps(launch_metadata, indent=2))
    (output_dir / "metrics_evidence.json").write_text(json.dumps(metrics_evidence, indent=2))
    with (output_dir / "frames.jsonl").open("w", encoding="utf-8") as handle:
        for frame in frames:
            handle.write(json.dumps(frame) + "\n")
    return manifest


def _build_alpasim_frames(selection_rows: list[dict[str, Any]], direct_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if direct_rows:
        return [_frame_from_direct_row(row) for row in direct_rows]
    return [_frame_from_selection_row(row) for row in selection_rows]


def _frame_from_selection_row(row: dict[str, Any]) -> dict[str, Any]:
    command = str(row.get("command", "straight"))
    signal = row.get("alpasim_signal", {}) if isinstance(row.get("alpasim_signal"), dict) else {}
    scenario = scenario_from_command(command, signal)
    ego_pose = signal.get("oracle_actor_proxy_current_ego_pose") if isinstance(signal.get("oracle_actor_proxy_current_ego_pose"), dict) else {}
    ego_x = float(ego_pose.get("x", scenario.start[0]))
    ego_y = float(ego_pose.get("y", scenario.start[1]))
    return {
        "frame_idx": int(row.get("frame_index", 0)),
        "timestamp_s": round((int(row.get("frame_index", 0)) - 1) * 0.25, 3),
        "ego": {
            "x": ego_x,
            "y": ego_y,
            "speed": float(row.get("speed_mps", 0.0)),
        },
        "route": {
            "start": list(scenario.start),
            "goal": list(scenario.goal),
            "lane_center": [list(point) for point in scenario.lane_center],
            "route_center": [list(point) for point in route_centerline(scenario)],
            "lane_half_width": float(scenario.lane_half_width),
        },
        "actors": [row_actor for row_actor in signal.get("structured_hazards", []) if _is_moving_hazard(row_actor)],
        "active_obstacles": [row_actor for row_actor in signal.get("structured_hazards", []) if not _is_moving_hazard(row_actor)],
        "step": {
            "action_mode": str(row.get("hybrid_token", row.get("spotlight_token", ""))),
            "decision_type": row.get("decision_type"),
            "collision_risk": float(signal.get("dynamics_risk", 0.0) or 0.0),
            "lane_error": 0.0,
            "min_obstacle_distance": float(row.get("dagger_argmax_geo_gap", 0.0) or 0.0),
            "selected_maneuver": row.get("hybrid_token"),
        },
        "planner": {
            "selection_mode": row.get("selection_mode"),
            "trajectory_mode": row.get("trajectory_mode"),
            "top_logits": row.get("top_logits", []),
            "selection_trace": {k: v for k, v in row.items() if k not in {"alpasim_signal", "top_logits", "spotlight_top_candidates"}},
            "spotlight_top_candidates": row.get("spotlight_top_candidates", []),
        },
        "trigger_state": {},
        "signal": signal,
    }


def _frame_from_direct_row(row: dict[str, Any]) -> dict[str, Any]:
    command = str(row.get("command", "straight"))
    signal = row.get("alpasim_signal", {}) if isinstance(row.get("alpasim_signal"), dict) else {}
    scenario = scenario_from_command(command, signal)
    plan = row.get("plan", {}) if isinstance(row.get("plan"), dict) else {}
    return {
        "frame_idx": int(row.get("frame_index", 0)),
        "timestamp_s": round((int(row.get("frame_index", 0)) - 1) * 0.25, 3),
        "ego": {
            "x": float(scenario.start[0]),
            "y": float(scenario.start[1]),
            "speed": float(row.get("speed_mps", 0.0)),
        },
        "route": {
            "start": list(scenario.start),
            "goal": list(scenario.goal),
            "lane_center": [list(point) for point in scenario.lane_center],
            "route_center": [list(point) for point in route_centerline(scenario)],
            "lane_half_width": float(scenario.lane_half_width),
        },
        "actors": [row_actor for row_actor in signal.get("structured_hazards", []) if _is_moving_hazard(row_actor)],
        "active_obstacles": [row_actor for row_actor in signal.get("structured_hazards", []) if not _is_moving_hazard(row_actor)],
        "step": {
            "action_mode": "direct_actor_planner",
            "collision_risk": float(signal.get("dynamics_risk", 0.0) or 0.0),
            "lane_error": 0.0,
            "min_obstacle_distance": float(plan.get("min_clearance_m", 0.0) or 0.0),
            "selected_maneuver": "direct_plan",
        },
        "planner": {
            "planner": row.get("planner"),
            "latency_ms": row.get("planner_latency_ms"),
            "plan": plan,
        },
        "trigger_state": {},
        "signal": signal,
    }


def _safe_metrics_evidence(run_dir: Path) -> dict[str, Any]:
    try:
        return build_alpasim_evidence(run_dir).to_dict()
    except Exception as exc:
        return {"source": "alpasim", "error": str(exc)}


def _load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _load_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _is_moving_hazard(hazard: dict[str, Any]) -> bool:
    vx = float(hazard.get("vx", 0.0) or 0.0)
    vy = float(hazard.get("vy", 0.0) or 0.0)
    return abs(vx) > 1e-6 or abs(vy) > 1e-6
