#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Mapping

from minimal_shot_av.cli.commands.run_nuplan_maneuvertoken_rollout import FAILURE_RUNGS
from minimal_shot_av.cli.commands.run_nuplan_maneuvertoken_rollout import RESOLVED_SAFE
from minimal_shot_av.cli.commands.run_nuplan_maneuvertoken_rollout import classify_failure_rung


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_INPUT_JSON = ROOT / "artifacts" / "corl2027" / "nuplan_mini_rollout50.json"
DEFAULT_OUTPUT_JSON = ROOT / "artifacts" / "corl2027" / "nuplan_mini_rollout50_realized_clearance.json"
DEFAULT_OUTPUT_MARKDOWN = ROOT / "artifacts" / "corl2027" / "nuplan_mini_rollout50_realized_clearance.md"
DEFAULT_NEAR_MISS_THRESHOLD_M = 1.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay selected ManeuverToken rollouts against logged future actor trajectories and "
            "write realized-clearance diagnostics."
        )
    )
    parser.add_argument("--input-json", type=Path, default=DEFAULT_INPUT_JSON)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_OUTPUT_MARKDOWN)
    parser.add_argument("--near-miss-threshold-m", type=float, default=DEFAULT_NEAR_MISS_THRESHOLD_M)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = json.loads(args.input_json.read_text(encoding="utf-8"))
    enriched = enrich_report_with_realized_clearance(
        report,
        clearance_provider=compute_scene_log_replay_realized_fields,
        near_miss_threshold_m=float(args.near_miss_threshold_m),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(enriched, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = markdown_report(enriched)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text(markdown + "\n", encoding="utf-8")
    print(markdown)
    return 0


def enrich_report_with_realized_clearance(
    report: Mapping[str, Any],
    *,
    clearance_provider: Callable[[dict[str, Any]], dict[str, Any] | None],
    near_miss_threshold_m: float,
) -> dict[str, Any]:
    enriched = json.loads(json.dumps(report))
    scenes = list(enriched.get("scenes", []))
    for scene in scenes:
        fields = clearance_provider(scene) or {}
        scene["selected_token_realized_min_clearance_m"] = _optional_round(
            fields.get("selected_token_realized_min_clearance_m")
        )
        scene["selected_token_realized_min_clearance_t"] = _optional_round(
            fields.get("selected_token_realized_min_clearance_t")
        )
        scene["selected_token_realized_collision"] = _optional_bool(fields.get("selected_token_realized_collision"))
        scene["selected_token_realized_near_miss"] = _optional_bool(fields.get("selected_token_realized_near_miss"))
        scene["realized_clearance_source"] = fields.get("realized_clearance_source")
        scene["realized_min_clearance"] = scene["selected_token_realized_min_clearance_m"]
        scene["collision_or_near_miss"] = scene["selected_token_realized_near_miss"]
        scene["failure_rung"] = classify_failure_rung(
            actor_visible_count=int(scene["actor_summary"]["visible_actor_count"]),
            actor_count=int(scene["actor_summary"]["actor_count"]),
            safe_token_existed=bool(scene["safe_token_existed"]),
            selector_chose_safe_token=bool(scene["selector_chose_safe_token"]),
            selected_token_realized_min_clearance_m=scene["selected_token_realized_min_clearance_m"],
            near_miss_threshold_m=float(near_miss_threshold_m),
        )
    _recompute_report_summary(enriched)
    enriched["schema"] = "nuplan_maneuvertoken_realized_clearance_v1"
    enriched["near_miss_threshold_m"] = float(near_miss_threshold_m)
    return enriched


def compute_scene_log_replay_realized_fields(scene: dict[str, Any]) -> dict[str, Any] | None:
    source_db_file = scene.get("source_db_file")
    scenario_token = scene.get("scenario_token")
    if not source_db_file or not scenario_token:
        return None
    query = _nuplan_replay_imports()
    sensor_source = query["get_lidarpc_sensor_data"]()
    timestamp_us = int(
        scene.get("sensor_timestamp_us")
        or query["get_sensor_data_token_timestamp_from_db"](source_db_file, sensor_source, scenario_token)
        or 0
    )
    if timestamp_us <= 0:
        return None
    ego = query["get_ego_state_for_lidarpc_token_from_db"](source_db_file, scenario_token)
    tracked_objects = list(query["get_tracked_objects_for_lidarpc_token_from_db"](source_db_file, scenario_token))
    dt_s = float(scene.get("selected_token_rollout_dt_s", 0.5))
    poses = list(scene["selected_token_rollout"]["poses"])
    horizon_us = int(round(len(poses) * dt_s * 1_000_000))
    actor_tracks = _build_actor_tracks(
        query=query,
        source_db_file=str(source_db_file),
        tracked_objects=tracked_objects,
        start_timestamp_us=timestamp_us,
        end_timestamp_us=timestamp_us + horizon_us,
    )
    min_clearance_m, min_time_s = _compute_log_replay_clearance(
        ego=ego,
        poses=poses,
        dt_s=dt_s,
        actor_tracks=actor_tracks,
    )
    if min_clearance_m is None:
        return None
    return {
        "selected_token_realized_min_clearance_m": min_clearance_m,
        "selected_token_realized_min_clearance_t": min_time_s,
        "selected_token_realized_collision": min_clearance_m < 0.0,
        "selected_token_realized_near_miss": min_clearance_m < DEFAULT_NEAR_MISS_THRESHOLD_M,
        "realized_clearance_source": "log_replay",
    }


def _recompute_report_summary(report: dict[str, Any]) -> None:
    scenes = list(report.get("scenes", []))
    rung_counts = {rung: 0 for rung in FAILURE_RUNGS}
    proxy_safe_selected_count = 0
    proxy_safe_realized_safe_count = 0
    proxy_safe_realized_near_or_collision_count = 0
    proxy_safe_realized_missing_count = 0
    for scene in scenes:
        rung = str(scene["failure_rung"])
        if rung in rung_counts:
            rung_counts[rung] += 1
        if bool(scene.get("proxy_safe_selected")):
            proxy_safe_selected_count += 1
            if scene.get("selected_token_realized_min_clearance_m") is None:
                proxy_safe_realized_missing_count += 1
            elif rung == RESOLVED_SAFE:
                proxy_safe_realized_safe_count += 1
            elif rung == "proxy predicted safe but realized failed":
                proxy_safe_realized_near_or_collision_count += 1
    report["proxy_safe_selected_count"] = proxy_safe_selected_count
    report["proxy_safe_realized_safe_count"] = proxy_safe_realized_safe_count
    report["proxy_safe_realized_near_or_collision_count"] = proxy_safe_realized_near_or_collision_count
    report["proxy_safe_realized_missing_count"] = proxy_safe_realized_missing_count
    report["failure_rung_table"] = [
        {
            "failure_rung": rung,
            "count": rung_counts[rung],
            "rate": round(rung_counts[rung] / len(scenes), 6) if scenes else 0.0,
        }
        for rung in FAILURE_RUNGS
    ]


def _build_actor_tracks(
    *,
    query: dict[str, Any],
    source_db_file: str,
    tracked_objects: list[Any],
    start_timestamp_us: int,
    end_timestamp_us: int,
) -> list[dict[str, Any]]:
    actor_tracks: list[dict[str, Any]] = []
    agent_tokens: list[str] = []
    current_rows: dict[str, dict[str, Any]] = {}
    for obj in tracked_objects:
        token = str(getattr(obj.metadata, "track_token", None) or getattr(obj.metadata, "token", ""))
        if not token:
            continue
        current_rows[token] = {
            "token": token,
            "x_m": float(obj.center.x),
            "y_m": float(obj.center.y),
            "radius_m": _object_radius_m(obj),
            "future": [],
        }
        if getattr(obj, "tracked_object_type", None) is not None and token not in agent_tokens:
            agent_tokens.append(token)
    for token, waypoint in query["get_future_waypoints_for_agents_from_db"](
        source_db_file,
        agent_tokens,
        start_timestamp_us,
        end_timestamp_us,
    ):
        key = str(token)
        if key not in current_rows:
            continue
        current_rows[key]["future"].append(
            {
                "time_us": int(waypoint.time_point.time_us),
                "x_m": float(waypoint.oriented_box.center.x),
                "y_m": float(waypoint.oriented_box.center.y),
                "radius_m": _oriented_box_radius_m(waypoint.oriented_box),
            }
        )
    actor_tracks.extend(current_rows.values())
    return actor_tracks


def _compute_log_replay_clearance(
    *,
    ego: Any,
    poses: list[list[float] | tuple[float, ...]],
    dt_s: float,
    actor_tracks: list[dict[str, Any]],
) -> tuple[float | None, float | None]:
    if not poses or not actor_tracks:
        return None, None
    ego_x = float(ego.rear_axle.x)
    ego_y = float(ego.rear_axle.y)
    ego_heading = float(ego.rear_axle.heading)
    min_clearance_m = math.inf
    min_time_s = None
    for index, pose in enumerate(poses):
        t_s = (index + 1) * dt_s
        global_x, global_y = _local_to_global(
            origin_x=ego_x,
            origin_y=ego_y,
            origin_heading=ego_heading,
            local_x=float(pose[0]),
            local_y=float(pose[1]),
        )
        for actor in actor_tracks:
            actor_state = _actor_state_at_time(actor, t_s=t_s)
            clearance = math.hypot(global_x - actor_state["x_m"], global_y - actor_state["y_m"]) - 1.0 - float(
                actor_state["radius_m"]
            )
            if clearance < min_clearance_m:
                min_clearance_m = clearance
                min_time_s = t_s
    if not math.isfinite(min_clearance_m):
        return None, None
    return float(min_clearance_m), float(min_time_s or 0.0)


def _actor_state_at_time(actor: dict[str, Any], *, t_s: float) -> dict[str, float]:
    future = list(actor.get("future", []))
    if not future:
        return {"x_m": float(actor["x_m"]), "y_m": float(actor["y_m"]), "radius_m": float(actor["radius_m"])}
    target_us = int(round(t_s * 1_000_000))
    nearest = min(future, key=lambda row: abs(int(row["time_us"]) - target_us))
    return {
        "x_m": float(nearest["x_m"]),
        "y_m": float(nearest["y_m"]),
        "radius_m": float(nearest["radius_m"]),
    }


def _local_to_global(
    *,
    origin_x: float,
    origin_y: float,
    origin_heading: float,
    local_x: float,
    local_y: float,
) -> tuple[float, float]:
    cos_h = math.cos(origin_heading)
    sin_h = math.sin(origin_heading)
    return (
        origin_x + cos_h * local_x - sin_h * local_y,
        origin_y + sin_h * local_x + cos_h * local_y,
    )


def _object_radius_m(obj: Any) -> float:
    box = getattr(obj, "box", None)
    if box is not None:
        return 0.5 * math.hypot(float(box.width), float(box.length))
    return 1.0


def _oriented_box_radius_m(box: Any) -> float:
    return 0.5 * math.hypot(float(box.width), float(box.length))


def _optional_round(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    return None if not math.isfinite(value) else round(value, 6)


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _nuplan_replay_imports() -> dict[str, Any]:
    try:
        from nuplan.database.nuplan_db.nuplan_db_utils import get_lidarpc_sensor_data
        from nuplan.database.nuplan_db.nuplan_scenario_queries import get_ego_state_for_lidarpc_token_from_db
        from nuplan.database.nuplan_db.nuplan_scenario_queries import get_future_waypoints_for_agents_from_db
        from nuplan.database.nuplan_db.nuplan_scenario_queries import get_sensor_data_token_timestamp_from_db
        from nuplan.database.nuplan_db.nuplan_scenario_queries import get_tracked_objects_for_lidarpc_token_from_db
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ImportError(
            "nuPlan devkit is not installed. Run `./scripts/bootstrap_nuplan_env.sh` first."
        ) from exc
    return {
        "get_ego_state_for_lidarpc_token_from_db": get_ego_state_for_lidarpc_token_from_db,
        "get_future_waypoints_for_agents_from_db": get_future_waypoints_for_agents_from_db,
        "get_lidarpc_sensor_data": get_lidarpc_sensor_data,
        "get_sensor_data_token_timestamp_from_db": get_sensor_data_token_timestamp_from_db,
        "get_tracked_objects_for_lidarpc_token_from_db": get_tracked_objects_for_lidarpc_token_from_db,
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# nuPlan Selected-Token Realized Clearance Audit",
        "",
        f"- Scene count: `{report['scene_count']}`",
        f"- Proxy-safe selected count: `{report['proxy_safe_selected_count']}`",
        f"- Proxy-safe + realized safe count: `{report['proxy_safe_realized_safe_count']}`",
        f"- Proxy-safe + realized near/collision count: `{report['proxy_safe_realized_near_or_collision_count']}`",
        f"- Proxy-safe + realized missing count: `{report['proxy_safe_realized_missing_count']}`",
        f"- Realized clearance source: `log_replay`",
        "",
        "## Failure Table",
        "",
        "| Failure rung | Count | Rate |",
        "|---|---:|---:|",
    ]
    for row in report["failure_rung_table"]:
        lines.append(f"| {row['failure_rung']} | {row['count']} | {row['rate']:.3f} |")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
