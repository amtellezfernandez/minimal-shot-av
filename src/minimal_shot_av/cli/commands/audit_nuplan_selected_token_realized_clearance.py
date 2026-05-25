#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
THRESHOLD_SENSITIVITY_METERS = (0.5, 1.0, 1.5, 2.0)
HORIZON_SENSITIVITY_SECONDS = (1.0, 2.0, 3.0, 4.0, 5.0)
REPLAY_INFEASIBLE_RUNG = "proxy-safe but replay-infeasible"
REPLAY_FAILURE_RUNGS = (
    "no actor/state visibility",
    "no safe token existed",
    "safe token existed but selector missed it",
    REPLAY_INFEASIBLE_RUNG,
    "metric/spec ambiguity",
)


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
        scene["selected_token_realized_clearance_trace"] = list(
            fields.get("selected_token_realized_clearance_trace", [])
        )
        scene["candidate_replay_evaluations"] = list(fields.get("candidate_replay_evaluations", []))
        scene["oracle_log_replay_safe_token"] = fields.get("oracle_log_replay_safe_token")
        scene["selected_failed_stop_cause"] = fields.get("selected_failed_stop_cause")
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
        scene["failure_rung"] = _replay_specific_failure_rung(scene["failure_rung"], scene["realized_clearance_source"])
        scene["hypothesis_supported"] = (
            "controller_proxy_mismatch_candidate"
            if scene["failure_rung"] == REPLAY_INFEASIBLE_RUNG
            else None
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
    proxy_trace = list(scene["selected_token_rollout"]["per_frame_diagnostics"])
    horizon_us = int(round(len(poses) * dt_s * 1_000_000))
    actor_tracks = _build_actor_tracks(
        query=query,
        source_db_file=str(source_db_file),
        tracked_objects=tracked_objects,
        start_timestamp_us=timestamp_us,
        end_timestamp_us=timestamp_us + horizon_us,
    )
    min_clearance_m, min_time_s, trace = _compute_log_replay_clearance(
        ego=ego,
        poses=poses,
        dt_s=dt_s,
        proxy_trace=proxy_trace,
        actor_tracks=actor_tracks,
    )
    if min_clearance_m is None:
        return None
    candidate_replay_evaluations = _candidate_replay_evaluations(
        ego=ego,
        scene=scene,
        dt_s=dt_s,
        actor_tracks=actor_tracks,
    )
    oracle_log_replay_safe_token = _oracle_log_replay_safe_token(candidate_replay_evaluations)
    selected_failed_stop_cause = _selected_failed_stop_cause(
        scene=scene,
        candidate_replay_evaluations=candidate_replay_evaluations,
        selected_token_realized_min_clearance_m=min_clearance_m,
    )
    return {
        "selected_token_realized_min_clearance_m": min_clearance_m,
        "selected_token_realized_min_clearance_t": min_time_s,
        "selected_token_realized_collision": min_clearance_m < 0.0,
        "selected_token_realized_near_miss": min_clearance_m < DEFAULT_NEAR_MISS_THRESHOLD_M,
        "selected_token_realized_clearance_trace": trace,
        "candidate_replay_evaluations": candidate_replay_evaluations,
        "oracle_log_replay_safe_token": oracle_log_replay_safe_token,
        "selected_failed_stop_cause": selected_failed_stop_cause,
        "realized_clearance_source": "log_replay",
    }


def _recompute_report_summary(report: dict[str, Any]) -> None:
    scenes = list(report.get("scenes", []))
    rung_counts = {rung: 0 for rung in REPLAY_FAILURE_RUNGS}
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
            elif rung == REPLAY_INFEASIBLE_RUNG:
                proxy_safe_realized_near_or_collision_count += 1
    report["proxy_safe_selected_count"] = proxy_safe_selected_count
    report["proxy_safe_realized_safe_count"] = proxy_safe_realized_safe_count
    report["proxy_safe_realized_near_or_collision_count"] = proxy_safe_realized_near_or_collision_count
    report["proxy_safe_realized_missing_count"] = proxy_safe_realized_missing_count
    report["proxy_safe_realized_near_or_collision_rate"] = (
        round(proxy_safe_realized_near_or_collision_count / proxy_safe_selected_count, 6)
        if proxy_safe_selected_count
        else 0.0
    )
    report["proxy_safe_realized_near_or_collision_ci95"] = _binomial_ci95(
        successes=proxy_safe_realized_near_or_collision_count,
        trials=proxy_safe_selected_count,
    )
    report["failure_rung_table"] = [
        {
            "failure_rung": rung,
            "count": rung_counts[rung],
            "rate": round(rung_counts[rung] / len(scenes), 6) if scenes else 0.0,
        }
        for rung in REPLAY_FAILURE_RUNGS
    ]
    report["token_failure_table"] = _token_failure_table(scenes)
    report["threshold_sensitivity_table"] = _threshold_sensitivity_table(
        scenes=scenes,
        thresholds_m=THRESHOLD_SENSITIVITY_METERS,
    )
    report["horizon_sensitivity_table"] = _horizon_sensitivity_table(
        scenes=scenes,
        horizons_s=HORIZON_SENSITIVITY_SECONDS,
    )
    report["no_safe_token_cause_table"] = _no_safe_token_cause_table(scenes)
    report["replay_oracle_case_table"] = _replay_oracle_case_table(scenes)
    report["failed_stop_cause_table"] = _failed_stop_cause_table(scenes)
    report["stop_vs_evasive_right_table"] = _stop_vs_evasive_right_table(scenes)
    report["stop_vs_evasive_right_summary"] = _stop_vs_evasive_right_summary(report["stop_vs_evasive_right_table"])
    report["clearance_trace_examples"] = _clearance_trace_examples(scenes)


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
    proxy_trace: list[dict[str, Any]],
    actor_tracks: list[dict[str, Any]],
) -> tuple[float | None, float | None, list[dict[str, float]]]:
    if not poses or not actor_tracks:
        return None, None, []
    ego_x = float(ego.rear_axle.x)
    ego_y = float(ego.rear_axle.y)
    ego_heading = float(ego.rear_axle.heading)
    min_clearance_m = math.inf
    min_time_s = None
    trace: list[dict[str, float]] = []
    for index, pose in enumerate(poses):
        t_s = (index + 1) * dt_s
        global_x, global_y = _local_to_global(
            origin_x=ego_x,
            origin_y=ego_y,
            origin_heading=ego_heading,
            local_x=float(pose[0]),
            local_y=float(pose[1]),
        )
        realized_clearance = math.inf
        for actor in actor_tracks:
            actor_state = _actor_state_at_time(actor, t_s=t_s)
            clearance = math.hypot(global_x - actor_state["x_m"], global_y - actor_state["y_m"]) - 1.0 - float(
                actor_state["radius_m"]
            )
            realized_clearance = min(realized_clearance, clearance)
        if realized_clearance < min_clearance_m:
            min_clearance_m = realized_clearance
            min_time_s = t_s
        proxy_clearance = float(proxy_trace[index]["proxy_clearance_m"]) if index < len(proxy_trace) else math.nan
        trace.append(
            {
                "step": float(index),
                "time_s": round(float(t_s), 6),
                "proxy_clearance_m": round(float(proxy_clearance), 6) if math.isfinite(proxy_clearance) else 50.0,
                "realized_clearance_m": (
                    round(float(realized_clearance), 6) if math.isfinite(realized_clearance) else 50.0
                ),
            }
        )
    if not math.isfinite(min_clearance_m):
        return None, None, trace
    return float(min_clearance_m), float(min_time_s or 0.0), trace


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


def _candidate_replay_evaluations(
    *,
    ego: Any,
    scene: dict[str, Any],
    dt_s: float,
    actor_tracks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for candidate in scene.get("candidates", []):
        min_clearance_m, min_time_s, trace = _compute_log_replay_clearance(
            ego=ego,
            poses=list(candidate["poses"]),
            dt_s=dt_s,
            proxy_trace=list(candidate.get("per_frame_diagnostics", [])),
            actor_tracks=actor_tracks,
        )
        rows.append(
            {
                "token": str(candidate["token"]),
                "proxy_safe": bool(candidate["proxy_safe"]),
                "proxy_min_clearance_m": round(float(candidate["min_proxy_clearance_m"]), 6),
                "realized_min_clearance_m": _optional_round(min_clearance_m),
                "realized_min_clearance_t": _optional_round(min_time_s),
                "realized_near_miss": (
                    None if min_clearance_m is None else bool(min_clearance_m < DEFAULT_NEAR_MISS_THRESHOLD_M)
                ),
                "realized_collision": None if min_clearance_m is None else bool(min_clearance_m < 0.0),
                "realized_clearance_trace": trace,
            }
        )
    return rows


def _oracle_log_replay_safe_token(candidate_replay_evaluations: list[dict[str, Any]]) -> str | None:
    safe = [
        row
        for row in candidate_replay_evaluations
        if row["realized_min_clearance_m"] is not None
        and float(row["realized_min_clearance_m"]) >= DEFAULT_NEAR_MISS_THRESHOLD_M
    ]
    if not safe:
        return None
    best = max(
        safe,
        key=lambda row: (float(row["realized_min_clearance_m"]), float(row["proxy_min_clearance_m"])),
    )
    return str(best["token"])


def _selected_failed_stop_cause(
    *,
    scene: dict[str, Any],
    candidate_replay_evaluations: list[dict[str, Any]],
    selected_token_realized_min_clearance_m: float | None,
) -> str | None:
    if str(scene.get("selected_token")) != "stop":
        return None
    selected_clearance = selected_token_realized_min_clearance_m
    if selected_clearance is None or float(selected_clearance) >= DEFAULT_NEAR_MISS_THRESHOLD_M:
        return None
    actor_summary = scene.get("actor_summary", {})
    if int(actor_summary.get("rear_closing_actor_count", 0)) > 0:
        return "rear_closing"
    if int(actor_summary.get("crossing_actor_count", 0)) > 0:
        return "crossing_conflict"
    oracle = _oracle_log_replay_safe_token(candidate_replay_evaluations)
    if oracle in {"evasive_left", "evasive_right", "nudge_left", "nudge_right", "lane_recover"}:
        return "stopped_in_conflict_zone"
    return "horizon_artifact"


def _token_failure_table(scenes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for scene in scenes:
        token = str(scene["selected_token"])
        row = rows.setdefault(
            token,
            {"token": token, "proxy_safe_selected": 0, "realized_near_or_collision": 0, "failure_rate": 0.0},
        )
        if bool(scene.get("proxy_safe_selected")):
            row["proxy_safe_selected"] += 1
            if scene.get("selected_token_realized_near_miss"):
                row["realized_near_or_collision"] += 1
    for row in rows.values():
        denom = int(row["proxy_safe_selected"])
        row["failure_rate"] = round(row["realized_near_or_collision"] / denom, 6) if denom else 0.0
    return [rows[key] for key in sorted(rows)]


def _threshold_sensitivity_table(
    *,
    scenes: list[dict[str, Any]],
    thresholds_m: tuple[float, ...],
) -> list[dict[str, Any]]:
    proxy_safe = [scene for scene in scenes if bool(scene.get("proxy_safe_selected"))]
    rows = []
    for threshold in thresholds_m:
        count = sum(
            1
            for scene in proxy_safe
            if scene.get("selected_token_realized_min_clearance_m") is not None
            and float(scene["selected_token_realized_min_clearance_m"]) < threshold
        )
        rows.append(
            {
                "threshold_m": float(threshold),
                "proxy_safe_realized_near_or_collision_count": count,
                "rate": round(count / len(proxy_safe), 6) if proxy_safe else 0.0,
            }
        )
    return rows


def _no_safe_token_cause_table(scenes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for scene in scenes:
        if bool(scene.get("safe_token_existed")):
            continue
        cause = _classify_no_safe_token_cause(scene)
        counts[cause] = counts.get(cause, 0) + 1
    return [{"cause": cause, "count": counts[cause]} for cause in sorted(counts)]


def _classify_no_safe_token_cause(scene: dict[str, Any]) -> str:
    candidates = list(scene.get("candidates", []))
    stop = next((candidate for candidate in candidates if candidate["token"] == "stop"), None)
    lateral = [
        candidate
        for candidate in candidates
        if candidate["token"] in {"nudge_left", "nudge_right", "evasive_left", "evasive_right", "lane_recover"}
    ]
    if candidates and all(float(candidate["min_proxy_clearance_m"]) < 0.0 for candidate in candidates):
        if stop is not None and float(stop["min_proxy_clearance_m"]) < 0.0:
            return "stop still unsafe"
        if lateral and all(float(candidate["min_proxy_clearance_m"]) < 0.0 for candidate in lateral):
            return "insufficient lateral options"
        return "actor clearance"
    return "unknown/spec issue"


def _clearance_trace_examples(scenes: list[dict[str, Any]], *, limit: int = 5) -> list[dict[str, Any]]:
    failed = [
        scene
        for scene in scenes
        if scene.get("failure_rung") == REPLAY_INFEASIBLE_RUNG
        and scene.get("selected_token_realized_clearance_trace")
    ]
    failed.sort(key=lambda scene: float(scene["selected_token_realized_min_clearance_m"]))
    examples = []
    for scene in failed[:limit]:
        examples.append(
            {
                "scene_id": scene["scene_id"],
                "selected_token": scene["selected_token"],
                "selected_token_realized_min_clearance_m": scene["selected_token_realized_min_clearance_m"],
                "selected_token_realized_min_clearance_t": scene["selected_token_realized_min_clearance_t"],
                "trace": list(scene["selected_token_realized_clearance_trace"]),
            }
        )
    return examples


def _replay_oracle_case_table(scenes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = {
        "proxy-safe token exists and replay-safe token exists": 0,
        "proxy-safe token exists but no replay-safe token exists": 0,
        "proxy-safe selected token fails but another token would replay-safe pass": 0,
        "no proxy-safe token but replay-safe token exists": 0,
    }
    for scene in scenes:
        proxy_safe_exists = bool(scene.get("safe_token_existed"))
        replay_safe_exists = scene.get("oracle_log_replay_safe_token") is not None
        selected_fails = bool(scene.get("selected_token_realized_near_miss"))
        selected_token = str(scene.get("selected_token"))
        oracle = scene.get("oracle_log_replay_safe_token")
        if proxy_safe_exists and replay_safe_exists:
            counts["proxy-safe token exists and replay-safe token exists"] += 1
        if proxy_safe_exists and not replay_safe_exists:
            counts["proxy-safe token exists but no replay-safe token exists"] += 1
        if proxy_safe_exists and selected_fails and oracle is not None and str(oracle) != selected_token:
            counts["proxy-safe selected token fails but another token would replay-safe pass"] += 1
        if not proxy_safe_exists and replay_safe_exists:
            counts["no proxy-safe token but replay-safe token exists"] += 1
    return [{"case": key, "count": value} for key, value in counts.items()]


def _failed_stop_cause_table(scenes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for scene in scenes:
        cause = scene.get("selected_failed_stop_cause")
        if cause is None:
            continue
        counts[str(cause)] = counts.get(str(cause), 0) + 1
    return [{"cause": cause, "count": counts[cause]} for cause in sorted(counts)]


def _stop_vs_evasive_right_table(scenes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for scene in scenes:
        if str(scene.get("selected_token")) != "stop":
            continue
        if not bool(scene.get("selected_token_realized_near_miss")):
            continue
        evasive = next(
            (
                row
                for row in scene.get("candidate_replay_evaluations", [])
                if str(row.get("token")) == "evasive_right"
            ),
            None,
        )
        rows.append(
            {
                "scene_id": str(scene["scene_id"]),
                "evasive_right_available": evasive is not None,
                "evasive_right_realized_min_clearance_m": (
                    None if evasive is None else evasive["realized_min_clearance_m"]
                ),
                "would_evasive_right_avoid_failure": (
                    False
                    if evasive is None or evasive["realized_min_clearance_m"] is None
                    else float(evasive["realized_min_clearance_m"]) >= DEFAULT_NEAR_MISS_THRESHOLD_M
                ),
            }
        )
    return rows


def _stop_vs_evasive_right_summary(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "failed_stop_scene_count": len(rows),
        "evasive_right_available_count": sum(1 for row in rows if bool(row["evasive_right_available"])),
        "evasive_right_would_avoid_failure_count": sum(
            1 for row in rows if bool(row["would_evasive_right_avoid_failure"])
        ),
    }


def _binomial_ci95(*, successes: int, trials: int) -> dict[str, float]:
    if trials <= 0:
        return {"lower": 0.0, "upper": 0.0}
    p = successes / trials
    margin = 1.96 * math.sqrt(max(p * (1.0 - p), 0.0) / trials)
    return {"lower": round(max(0.0, p - margin), 6), "upper": round(min(1.0, p + margin), 6)}


def _replay_specific_failure_rung(failure_rung: str, realized_clearance_source: Any) -> str:
    if str(realized_clearance_source) == "log_replay" and failure_rung == "proxy predicted safe but realized failed":
        return REPLAY_INFEASIBLE_RUNG
    return str(failure_rung)


def _horizon_sensitivity_table(
    *,
    scenes: list[dict[str, Any]],
    horizons_s: tuple[float, ...],
) -> list[dict[str, Any]]:
    rows = []
    for horizon_s in horizons_s:
        selected_failures = 0
        stop_failures = 0
        no_replay_safe_token = 0
        available = False
        for scene in scenes:
            selected_trace = list(scene.get("selected_token_realized_clearance_trace", []))
            if selected_trace:
                available = available or _horizon_available(selected_trace, horizon_s)
            if bool(scene.get("proxy_safe_selected")) and _trace_near_miss(selected_trace, horizon_s):
                selected_failures += 1
                if str(scene.get("selected_token")) == "stop":
                    stop_failures += 1
            evaluations = list(scene.get("candidate_replay_evaluations", []))
            if evaluations and not _any_replay_safe_by_horizon(evaluations, horizon_s):
                no_replay_safe_token += 1
        rows.append(
            {
                "horizon_s": float(horizon_s),
                "available": available,
                "proxy_safe_selected_failures": selected_failures if available else None,
                "proxy_safe_stop_failures": stop_failures if available else None,
                "no_replay_safe_token": no_replay_safe_token if available else None,
            }
        )
    return rows


def _trace_near_miss(trace: list[dict[str, Any]], horizon_s: float) -> bool:
    if not _horizon_available(trace, horizon_s):
        return False
    clearances = [
        float(row["realized_clearance_m"])
        for row in trace
        if float(row["time_s"]) <= horizon_s + 1.0e-6
    ]
    return bool(clearances) and min(clearances) < DEFAULT_NEAR_MISS_THRESHOLD_M


def _horizon_available(trace: list[dict[str, Any]], horizon_s: float) -> bool:
    return bool(trace) and max(float(row["time_s"]) for row in trace) + 1.0e-6 >= horizon_s


def _any_replay_safe_by_horizon(evaluations: list[dict[str, Any]], horizon_s: float) -> bool:
    for evaluation in evaluations:
        trace = list(evaluation.get("realized_clearance_trace", []))
        if not _horizon_available(trace, horizon_s):
            continue
        clearances = [
            float(row["realized_clearance_m"])
            for row in trace
            if float(row["time_s"]) <= horizon_s + 1.0e-6
        ]
        if clearances and min(clearances) >= DEFAULT_NEAR_MISS_THRESHOLD_M:
            return True
    return False


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
        f"- Proxy-safe realized near/collision rate: `{report['proxy_safe_realized_near_or_collision_rate']:.3f}`",
        f"- Proxy-safe realized near/collision 95% CI: "
        f"`[{report['proxy_safe_realized_near_or_collision_ci95']['lower']:.3f}, "
        f"{report['proxy_safe_realized_near_or_collision_ci95']['upper']:.3f}]`",
        f"- Realized clearance source: `log_replay`",
        "",
        "## Failure Table",
        "",
        "| Failure rung | Count | Rate |",
        "|---|---:|---:|",
    ]
    for row in report["failure_rung_table"]:
        lines.append(f"| {row['failure_rung']} | {row['count']} | {row['rate']:.3f} |")
    lines.extend(
        [
            "",
            "## Token Failure Table",
            "",
            "| Token | Proxy-safe selected | Realized near/collision | Failure rate |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in report["token_failure_table"]:
        lines.append(
            f"| {row['token']} | {row['proxy_safe_selected']} | "
            f"{row['realized_near_or_collision']} | {row['failure_rate']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Threshold Sensitivity",
            "",
            "| Threshold | Proxy-safe + realized near/collision | Rate |",
            "|---:|---:|---:|",
        ]
    )
    for row in report["threshold_sensitivity_table"]:
        lines.append(
            f"| {row['threshold_m']:.1f} m | {row['proxy_safe_realized_near_or_collision_count']} | {row['rate']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Horizon Sensitivity",
            "",
            "| Horizon | Proxy-safe selected failures | Proxy-safe stop failures | No replay-safe token |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in report["horizon_sensitivity_table"]:
        if not row["available"]:
            lines.append(f"| {row['horizon_s']:.1f} s | n/a | n/a | n/a |")
            continue
        lines.append(
            f"| {row['horizon_s']:.1f} s | {row['proxy_safe_selected_failures']} | "
            f"{row['proxy_safe_stop_failures']} | {row['no_replay_safe_token']} |"
        )
    lines.extend(
        [
            "",
            "## No-Safe-Token Causes",
            "",
            "| Cause | Count |",
            "|---|---:|",
        ]
    )
    for row in report["no_safe_token_cause_table"]:
        lines.append(f"| {row['cause']} | {row['count']} |")
    lines.extend(
        [
            "",
            "## Replay Oracle Cases",
            "",
            "| Case | Count |",
            "|---|---:|",
        ]
    )
    for row in report["replay_oracle_case_table"]:
        lines.append(f"| {row['case']} | {row['count']} |")
    lines.extend(
        [
            "",
            "## Failed Stop Causes",
            "",
            "| Cause | Count |",
            "|---|---:|",
        ]
    )
    for row in report["failed_stop_cause_table"]:
        lines.append(f"| {row['cause']} | {row['count']} |")
    summary = report["stop_vs_evasive_right_summary"]
    lines.extend(
        [
            "",
            "## Stop Vs Evasive Right",
            "",
            f"- Failed stop scenes: `{summary['failed_stop_scene_count']}`",
            f"- `evasive_right` available: `{summary['evasive_right_available_count']}`",
            f"- `evasive_right` would avoid failure: `{summary['evasive_right_would_avoid_failure_count']}`",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
