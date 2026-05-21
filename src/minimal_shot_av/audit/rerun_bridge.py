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
    return manifest
