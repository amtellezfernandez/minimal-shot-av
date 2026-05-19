#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from minimal_shot_av.simulator.alpasim_direct_actor_planner import (
    DirectPlannerConfig,
    _candidate_trajectory,
    _trajectory_cost,
)
from minimal_shot_av.simulator.alpasim_signal import scenario_from_command
from minimal_shot_av.simulator.alpasim_token_bc import (
    _adapter_spotlight_config,
    _current_ego_velocity_world,
    _load_oracle_actor_proxy,
    _world_actor_to_current_hazard,
)
from minimal_shot_av.simulator.perception import perceive_scene
from minimal_shot_av.simulator.spotlight_reflex import evaluate_maneuver_candidates
from minimal_shot_av.simulator.world_model import update_world_state


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = ROOT / "runs" / "alpasim_axis_constrained_30scene_eval"
DEFAULT_ORACLE_PROXY = ROOT / "artifacts" / "alpasim_oracle_actor_proxy_30scene.json"
DEFAULT_JSON = ROOT / "artifacts" / "alpasim_action_space_upper_bound_30scene.json"
DEFAULT_MD = ROOT / "artifacts" / "alpasim_action_space_upper_bound_30scene.md"

COLLISION_COMPONENTS = ("collision_front", "collision_lateral", "collision_rear")
ROUTE_STABLE_TOKENS = frozenset({"stop", "crawl", "maintain", "slow_yield", "lane_recover"})


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate a proxy upper bound for baseline AlpaSim collision clips by asking whether "
            "the token candidate set or selector-free direct grid contained no-overlap actions "
            "before first impact. This is an open-loop actor-proxy audit, not an AlpaSim replay."
        )
    )
    parser.add_argument("--baseline-run-dir", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--oracle-actor-proxy", type=Path, default=DEFAULT_ORACLE_PROXY)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_MD)
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Audit every Nth frame before first impact. The impact frame is always included.",
    )
    parser.add_argument(
        "--actionable-lead-frames",
        type=int,
        default=5,
        help="Minimum frame lead before first impact to count as actionable.",
    )
    parser.add_argument(
        "--clearance-threshold-m",
        type=float,
        default=0.0,
        help="Minimum horizon clearance for an action to count as proxy collision-free.",
    )
    parser.add_argument(
        "--margin-threshold-m",
        type=float,
        default=0.55,
        help="More conservative clearance margin reported separately from no-overlap safety.",
    )
    parser.add_argument(
        "--oracle-proxy-tolerance-us",
        type=int,
        default=50_000,
        help="Maximum timestamp mismatch when matching world-frame oracle actor proxy frames.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = audit_action_space_upper_bound(
        baseline_run_dir=args.baseline_run_dir,
        oracle_actor_proxy_path=args.oracle_actor_proxy,
        frame_stride=args.frame_stride,
        actionable_lead_frames=args.actionable_lead_frames,
        clearance_threshold_m=args.clearance_threshold_m,
        margin_threshold_m=args.margin_threshold_m,
        oracle_proxy_tolerance_us=args.oracle_proxy_tolerance_us,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = _markdown(report)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text(markdown + "\n", encoding="utf-8")
    print(markdown)
    return 0


def audit_action_space_upper_bound(
    *,
    baseline_run_dir: Path,
    oracle_actor_proxy_path: Path,
    frame_stride: int,
    actionable_lead_frames: int,
    clearance_threshold_m: float,
    margin_threshold_m: float,
    oracle_proxy_tolerance_us: int,
) -> dict[str, Any]:
    baseline_run_dir = baseline_run_dir.resolve()
    oracle_actor_proxy_path = oracle_actor_proxy_path.resolve()
    proxy_frames, _proxy_timestamps = _load_oracle_actor_proxy(oracle_actor_proxy_path)
    proxy_by_scene = _index_proxy_frames_by_scene(proxy_frames)

    scenes = []
    for scene_name, scene_dir in _load_scene_dirs(baseline_run_dir).items():
        timeline = _metric_timeline(scene_dir)
        first_collision = _first_collision_from_timeline(timeline)
        if first_collision is None:
            continue
        selection_rows, selection_summary = _load_selection_log(scene_dir / "driver" / "selection-log.jsonl")
        controller_rows = _load_controller_rows(scene_dir / "controller")
        scene_frames = _audit_scene_frames(
            scene_name=scene_name,
            timeline=timeline,
            first_collision=first_collision,
            selection_rows=selection_rows,
            selection_summary=selection_summary,
            controller_rows=controller_rows,
            proxy_scene_frames=proxy_by_scene.get(_clipgt_id(scene_name), {}),
            frame_stride=max(1, int(frame_stride)),
            actionable_lead_frames=actionable_lead_frames,
            clearance_threshold_m=clearance_threshold_m,
            margin_threshold_m=margin_threshold_m,
            oracle_proxy_tolerance_us=oracle_proxy_tolerance_us,
        )
        scenes.append(scene_frames)

    summary = _summary(
        scenes,
        baseline_run_dir=baseline_run_dir,
        oracle_actor_proxy_path=oracle_actor_proxy_path,
        frame_stride=frame_stride,
        actionable_lead_frames=actionable_lead_frames,
        clearance_threshold_m=clearance_threshold_m,
        margin_threshold_m=margin_threshold_m,
        oracle_proxy_tolerance_us=oracle_proxy_tolerance_us,
    )
    return {
        "schema": "alpasim_action_space_upper_bound_v1",
        "summary": summary,
        "scenes": scenes,
    }


def _audit_scene_frames(
    *,
    scene_name: str,
    timeline: list[dict[str, Any]],
    first_collision: dict[str, Any],
    selection_rows: dict[int, dict[str, Any]],
    selection_summary: dict[str, Any],
    controller_rows: list[dict[str, float | int]],
    proxy_scene_frames: dict[int, dict[str, Any]],
    frame_stride: int,
    actionable_lead_frames: int,
    clearance_threshold_m: float,
    margin_threshold_m: float,
    oracle_proxy_tolerance_us: int,
) -> dict[str, Any]:
    impact_frame = int(first_collision["frame_index"])
    frame_indexes = list(range(1, impact_frame + 1, frame_stride))
    if impact_frame not in frame_indexes:
        frame_indexes.append(impact_frame)
    frame_indexes = sorted(set(frame_indexes))
    frames = [
        _audit_frame(
            scene_name=scene_name,
            frame_index=frame_index,
            first_collision=first_collision,
            timeline=timeline,
            selection_row=selection_rows.get(frame_index),
            controller_rows=controller_rows,
            proxy_scene_frames=proxy_scene_frames,
            clearance_threshold_m=clearance_threshold_m,
            margin_threshold_m=margin_threshold_m,
            oracle_proxy_tolerance_us=oracle_proxy_tolerance_us,
        )
        for frame_index in frame_indexes
    ]
    bucket = _classify_scene(frames, actionable_lead_frames=actionable_lead_frames)
    return {
        "scene": scene_name,
        "clipgt_id": _clipgt_id(scene_name),
        "baseline_first_collision": first_collision,
        "baseline_selection_log": selection_summary,
        "frame_stride": frame_stride,
        "bucket": bucket,
        "frames": frames,
    }


def _audit_frame(
    *,
    scene_name: str,
    frame_index: int,
    first_collision: dict[str, Any],
    timeline: list[dict[str, Any]],
    selection_row: dict[str, Any] | None,
    controller_rows: list[dict[str, float | int]],
    proxy_scene_frames: dict[int, dict[str, Any]],
    clearance_threshold_m: float,
    margin_threshold_m: float,
    oracle_proxy_tolerance_us: int,
) -> dict[str, Any]:
    timestamp_us = _timestamp_for_frame(timeline, frame_index)
    lead_frames = max(0, int(first_collision["frame_index"]) - int(frame_index))
    base: dict[str, Any] = {
        "frame_index": frame_index,
        "lead_frames": lead_frames,
        "lead_seconds": round(lead_frames / 10.0, 3),
        "timestamp_us": timestamp_us,
        "status": "ok",
    }
    if timestamp_us is None:
        base["status"] = "missing_metric_timestamp"
        return base
    if selection_row is None:
        base["status"] = "missing_selection_log_row"
        return base
    controller_row = _nearest_controller_row(controller_rows, timestamp_us)
    if controller_row is None:
        base["status"] = "missing_controller_row"
        return base
    proxy_frame, proxy_delta_us = _nearest_scene_proxy_frame(
        proxy_scene_frames,
        timestamp_us,
        tolerance_us=oracle_proxy_tolerance_us,
    )
    if proxy_frame is None:
        base["status"] = "missing_oracle_proxy_frame"
        base["baseline_selected_token"] = selection_row.get("hybrid_token")
        base["oracle_proxy_delta_us"] = proxy_delta_us
        return base

    scenario, hazards = _scenario_with_world_frame_oracle(
        selection_row=selection_row,
        controller_row=controller_row,
        proxy_frame=proxy_frame,
    )
    speed_mps = _as_float(selection_row.get("speed_mps"), default=0.0)
    token = _token_action_space_summary(
        selection_row,
        scenario=scenario,
        speed_mps=speed_mps,
        clearance_threshold_m=clearance_threshold_m,
        margin_threshold_m=margin_threshold_m,
    )
    direct = _direct_grid_action_space_summary(
        scenario=scenario,
        speed_mps=speed_mps,
        clearance_threshold_m=clearance_threshold_m,
        margin_threshold_m=margin_threshold_m,
    )

    base.update(
        {
            "scene": scene_name,
            "baseline_selected_token": selection_row.get("hybrid_token"),
            "baseline_spotlight_token": selection_row.get("spotlight_token"),
            "baseline_dagger_argmax_token": selection_row.get("dagger_argmax_token"),
            "command": selection_row.get("command"),
            "speed_mps": _round(speed_mps),
            "oracle_proxy_hit": True,
            "oracle_proxy_delta_us": proxy_delta_us,
            "oracle_proxy_actor_count": len(proxy_frame.get("world_actors", []) or []),
            "hazard_count": len(hazards),
            "closest_hazard": _closest_hazard(hazards),
            "token_candidate_set": token,
            "direct_grid": direct,
        }
    )
    return base


def _scenario_with_world_frame_oracle(
    *,
    selection_row: dict[str, Any],
    controller_row: dict[str, float | int],
    proxy_frame: dict[str, Any],
) -> tuple[Any, list[dict[str, float | str]]]:
    signal = dict(selection_row.get("alpasim_signal") or {})
    ego_pose = {
        "world_x": float(controller_row["world_x"]),
        "world_y": float(controller_row["world_y"]),
        "world_heading": float(controller_row["world_heading"]),
        "world_vx": float(controller_row["world_vx"]),
        "world_vy": float(controller_row["world_vy"]),
    }
    speed_mps = _as_float(selection_row.get("speed_mps"), default=0.0)
    ego_velocity = _current_ego_velocity_world(ego_pose, speed_mps=speed_mps)
    hazards = []
    for index, actor in enumerate(proxy_frame.get("world_actors", []) or []):
        hazard = _world_actor_to_current_hazard(actor, ego_pose, ego_velocity, index)
        if hazard is not None:
            hazards.append(hazard)
    signal["structured_hazards"] = list(signal.get("structured_hazards") or []) + hazards
    signal["oracle_actor_proxy_hit"] = True
    signal["oracle_actor_proxy_count"] = len(hazards)
    signal["oracle_actor_proxy_scene_id"] = proxy_frame.get("scene_id")
    command = str(selection_row.get("command", "straight"))
    return scenario_from_command(command, signal), hazards


def _token_action_space_summary(
    selection_row: dict[str, Any],
    *,
    scenario: Any,
    speed_mps: float,
    clearance_threshold_m: float,
    margin_threshold_m: float,
) -> dict[str, Any]:
    command = str(selection_row.get("command", "straight"))
    trajectory_mode = str(selection_row.get("trajectory_mode", "clamped_lateral"))
    max_lateral_offset_m = _as_float(selection_row.get("max_lateral_offset_m"), default=2.0)
    config = _adapter_spotlight_config(
        trajectory_mode=trajectory_mode,
        max_lateral_offset_m=max_lateral_offset_m,
        selection_mode="actor_axis_constrained",
    )
    scenario_now = scenario
    position = scenario_now.start
    perception = perceive_scene(scenario_now, position)
    world_state = update_world_state(scenario_now, position, perception)
    evaluations, reference_count = evaluate_maneuver_candidates(
        scenario_now,
        position,
        world_state,
        perception,
        speed_mps=speed_mps,
        config=config,
    )
    rows = []
    for evaluation in evaluations:
        explanation = evaluation.explanation
        trajectory = evaluation.candidate.trajectory
        horizon_clearance = _as_float(explanation.horizon_clearance_m)
        action_clearance = _as_float(explanation.action_clearance_m)
        progress = 0.0
        if trajectory:
            progress = _as_float(trajectory[-1][0], default=0.0)
        rows.append(
            {
                "candidate": evaluation.candidate.name,
                "effective_score": _round(explanation.effective_score),
                "selector_score": _round(explanation.selector_score),
                "action_clearance_m": _round(action_clearance),
                "horizon_clearance_m": _round(horizon_clearance),
                "progress_m": _round(progress),
                "collision_free_proxy": horizon_clearance >= clearance_threshold_m,
                "margin_safe_proxy": horizon_clearance >= margin_threshold_m,
                "route_stable": evaluation.candidate.name in ROUTE_STABLE_TOKENS,
            }
        )
    rows.sort(key=lambda item: _sort_float(item["effective_score"]), reverse=True)
    selected_token = str(selection_row.get("hybrid_token"))
    selected = next((row for row in rows if row["candidate"] == selected_token), None)
    safe_rows = [row for row in rows if row["collision_free_proxy"]]
    margin_rows = [row for row in rows if row["margin_safe_proxy"]]
    best_clearance = max(rows, key=lambda row: _sort_float(row["horizon_clearance_m"]), default=None)
    best_safe_progress = max(safe_rows, key=lambda row: _sort_float(row["progress_m"]), default=None)
    return {
        "command": command,
        "actor_forecast_mode": "time_swept",
        "reference_count": reference_count,
        "candidate_count": len(rows),
        "selected_token": selected_token,
        "selected_collision_free_proxy": bool(selected and selected["collision_free_proxy"]),
        "selected_margin_safe_proxy": bool(selected and selected["margin_safe_proxy"]),
        "selected_horizon_clearance_m": selected["horizon_clearance_m"] if selected else None,
        "any_collision_free_proxy": bool(safe_rows),
        "any_margin_safe_proxy": bool(margin_rows),
        "collision_free_token_count": len(safe_rows),
        "margin_safe_token_count": len(margin_rows),
        "best_clearance_token": best_clearance["candidate"] if best_clearance else None,
        "best_clearance_m": best_clearance["horizon_clearance_m"] if best_clearance else None,
        "best_safe_progress_token": best_safe_progress["candidate"] if best_safe_progress else None,
        "best_safe_progress_m": best_safe_progress["progress_m"] if best_safe_progress else None,
        "safe_tokens": [row["candidate"] for row in safe_rows],
        "margin_safe_tokens": [row["candidate"] for row in margin_rows],
        "top_candidates": rows[:5],
    }


def _direct_grid_action_space_summary(
    *,
    scenario: Any,
    speed_mps: float,
    clearance_threshold_m: float,
    margin_threshold_m: float,
) -> dict[str, Any]:
    config = DirectPlannerConfig()
    speed_scales = tuple(
        scale for scale in config.speed_scales if 0.0 <= scale <= max(0.0, config.max_accel_speed_scale)
    )
    lateral_offsets = tuple(
        max(-config.max_lateral_offset_m, min(config.max_lateral_offset_m, offset))
        for offset in config.lateral_offsets_m
    )
    rows = []
    for speed_scale in speed_scales:
        for lateral_offset in lateral_offsets:
            trajectory = _candidate_trajectory(
                scenario,
                speed_mps=speed_mps,
                speed_scale=speed_scale,
                lateral_offset_m=lateral_offset,
                config=config,
            )
            cost, metrics = _trajectory_cost(
                trajectory,
                scenario=scenario,
                speed_mps=speed_mps,
                speed_scale=speed_scale,
                lateral_offset_m=lateral_offset,
                config=config,
            )
            clearance = _metric_float(metrics.get("min_clearance_m"))
            progress = _metric_float(metrics.get("progress_m"))
            row = {
                "candidate": f"scale={speed_scale:.2f},lat={lateral_offset:.2f}",
                "cost": _round(cost),
                "speed_scale": _round(speed_scale),
                "lateral_offset_m": _round(lateral_offset),
                "min_clearance_m": _round(clearance),
                "progress_m": _round(progress),
                "rear_flow_penalty": metrics.get("rear_flow_penalty"),
                "rear_actor_count": metrics.get("rear_actor_count"),
                "rear_closing_actor_count": metrics.get("rear_closing_actor_count"),
                "collision_free_proxy": clearance >= clearance_threshold_m,
                "margin_safe_proxy": clearance >= margin_threshold_m,
            }
            rows.append(row)
    rows.sort(key=lambda item: _sort_float(item["cost"]))
    selected = rows[0] if rows else None
    safe_rows = [row for row in rows if row["collision_free_proxy"]]
    margin_rows = [row for row in rows if row["margin_safe_proxy"]]
    best_clearance = max(rows, key=lambda row: _sort_float(row["min_clearance_m"]), default=None)
    best_safe_progress = max(safe_rows, key=lambda row: _sort_float(row["progress_m"]), default=None)
    cost_missed_safe = bool(safe_rows and selected and not selected["collision_free_proxy"])
    cost_missed_margin = bool(margin_rows and selected and not selected["margin_safe_proxy"])
    return {
        "candidate_count": len(rows),
        "selected_candidate": selected["candidate"] if selected else None,
        "selected_collision_free_proxy": bool(selected and selected["collision_free_proxy"]),
        "selected_margin_safe_proxy": bool(selected and selected["margin_safe_proxy"]),
        "selected_min_clearance_m": selected["min_clearance_m"] if selected else None,
        "selected_progress_m": selected["progress_m"] if selected else None,
        "any_collision_free_proxy": bool(safe_rows),
        "any_margin_safe_proxy": bool(margin_rows),
        "collision_free_candidate_count": len(safe_rows),
        "margin_safe_candidate_count": len(margin_rows),
        "cost_missed_collision_free_proxy": cost_missed_safe,
        "cost_missed_margin_safe_proxy": cost_missed_margin,
        "best_clearance_candidate": best_clearance["candidate"] if best_clearance else None,
        "best_clearance_m": best_clearance["min_clearance_m"] if best_clearance else None,
        "best_safe_progress_candidate": best_safe_progress["candidate"] if best_safe_progress else None,
        "best_safe_progress_m": best_safe_progress["progress_m"] if best_safe_progress else None,
        "top_cost_candidates": rows[:5],
        "top_clearance_candidates": sorted(
            rows,
            key=lambda item: _sort_float(item["min_clearance_m"]),
            reverse=True,
        )[:5],
    }


def _classify_scene(frames: list[dict[str, Any]], *, actionable_lead_frames: int) -> dict[str, Any]:
    evaluable = [frame for frame in frames if frame.get("status") == "ok"]
    actionable = [frame for frame in evaluable if int(frame.get("lead_frames", 0)) >= actionable_lead_frames]
    token_safe = [frame for frame in actionable if (frame.get("token_candidate_set") or {}).get("any_collision_free_proxy")]
    token_margin = [frame for frame in actionable if (frame.get("token_candidate_set") or {}).get("any_margin_safe_proxy")]
    token_selected_safe = [
        frame for frame in token_safe if (frame.get("token_candidate_set") or {}).get("selected_collision_free_proxy")
    ]
    token_selected_margin = [
        frame for frame in token_margin if (frame.get("token_candidate_set") or {}).get("selected_margin_safe_proxy")
    ]
    direct_safe = [frame for frame in actionable if (frame.get("direct_grid") or {}).get("any_collision_free_proxy")]
    direct_margin = [frame for frame in actionable if (frame.get("direct_grid") or {}).get("any_margin_safe_proxy")]
    direct_selected_safe = [
        frame for frame in direct_safe if (frame.get("direct_grid") or {}).get("selected_collision_free_proxy")
    ]
    direct_selected_margin = [
        frame for frame in direct_margin if (frame.get("direct_grid") or {}).get("selected_margin_safe_proxy")
    ]
    direct_cost_miss = [
        frame for frame in direct_safe if (frame.get("direct_grid") or {}).get("cost_missed_collision_free_proxy")
    ]
    direct_cost_miss_margin = [
        frame for frame in direct_margin if (frame.get("direct_grid") or {}).get("cost_missed_margin_safe_proxy")
    ]
    impact = next((frame for frame in evaluable if int(frame.get("lead_frames", -1)) == 0), None)

    if not evaluable:
        label = "insufficient_logs"
    elif not direct_safe and not token_safe:
        label = "no_proxy_safe_action_in_token_or_direct_action_space"
    elif direct_cost_miss:
        label = "direct_grid_contains_proxy_safe_action_but_cost_rejects_it"
    elif direct_selected_safe:
        label = "direct_grid_cost_selects_proxy_safe_action_before_collision"
    elif token_safe and not token_selected_safe:
        label = "token_candidate_set_contains_proxy_safe_action_but_selector_rejects_it"
    else:
        label = "proxy_safe_action_exists_but_diagnosis_mixed"

    return {
        "label": label,
        "evaluable_frame_count": len(evaluable),
        "actionable_frame_count": len(actionable),
        "token_proxy_safe_actionable_frame_count": len(token_safe),
        "token_selected_proxy_safe_actionable_frame_count": len(token_selected_safe),
        "token_margin_safe_actionable_frame_count": len(token_margin),
        "token_selected_margin_safe_actionable_frame_count": len(token_selected_margin),
        "direct_proxy_safe_actionable_frame_count": len(direct_safe),
        "direct_selected_proxy_safe_actionable_frame_count": len(direct_selected_safe),
        "direct_cost_missed_proxy_safe_actionable_frame_count": len(direct_cost_miss),
        "direct_margin_safe_actionable_frame_count": len(direct_margin),
        "direct_selected_margin_safe_actionable_frame_count": len(direct_selected_margin),
        "direct_cost_missed_margin_safe_actionable_frame_count": len(direct_cost_miss_margin),
        "first_token_proxy_safe_lead_frames": _max_lead(token_safe),
        "first_direct_proxy_safe_lead_frames": _max_lead(direct_safe),
        "first_direct_cost_miss_lead_frames": _max_lead(direct_cost_miss),
        "impact_token_proxy_safe": bool(
            impact and (impact.get("token_candidate_set") or {}).get("any_collision_free_proxy")
        ),
        "impact_direct_proxy_safe": bool(impact and (impact.get("direct_grid") or {}).get("any_collision_free_proxy")),
        "impact_direct_selected_proxy_safe": bool(
            impact and (impact.get("direct_grid") or {}).get("selected_collision_free_proxy")
        ),
    }


def _summary(
    scenes: list[dict[str, Any]],
    *,
    baseline_run_dir: Path,
    oracle_actor_proxy_path: Path,
    frame_stride: int,
    actionable_lead_frames: int,
    clearance_threshold_m: float,
    margin_threshold_m: float,
    oracle_proxy_tolerance_us: int,
) -> dict[str, Any]:
    buckets = Counter(scene["bucket"]["label"] for scene in scenes)
    all_frames = [frame for scene in scenes for frame in scene["frames"]]
    evaluable = [frame for frame in all_frames if frame.get("status") == "ok"]
    actionable = [frame for frame in evaluable if int(frame.get("lead_frames", 0)) >= actionable_lead_frames]
    token_safe = [frame for frame in actionable if (frame.get("token_candidate_set") or {}).get("any_collision_free_proxy")]
    token_margin = [frame for frame in actionable if (frame.get("token_candidate_set") or {}).get("any_margin_safe_proxy")]
    token_selected_safe = [
        frame for frame in token_safe if (frame.get("token_candidate_set") or {}).get("selected_collision_free_proxy")
    ]
    token_selected_margin = [
        frame for frame in token_margin if (frame.get("token_candidate_set") or {}).get("selected_margin_safe_proxy")
    ]
    direct_safe = [frame for frame in actionable if (frame.get("direct_grid") or {}).get("any_collision_free_proxy")]
    direct_margin = [frame for frame in actionable if (frame.get("direct_grid") or {}).get("any_margin_safe_proxy")]
    direct_selected_safe = [
        frame for frame in direct_safe if (frame.get("direct_grid") or {}).get("selected_collision_free_proxy")
    ]
    direct_selected_margin = [
        frame for frame in direct_margin if (frame.get("direct_grid") or {}).get("selected_margin_safe_proxy")
    ]
    direct_cost_miss = [
        frame for frame in direct_safe if (frame.get("direct_grid") or {}).get("cost_missed_collision_free_proxy")
    ]
    direct_cost_miss_margin = [
        frame for frame in direct_margin if (frame.get("direct_grid") or {}).get("cost_missed_margin_safe_proxy")
    ]
    components = Counter(
        component
        for scene in scenes
        for component in scene["baseline_first_collision"].get("components", [])
    )
    status_counts = Counter(frame.get("status", "unknown") for frame in all_frames)
    return {
        "baseline_run_dir": str(baseline_run_dir),
        "oracle_actor_proxy_path": str(oracle_actor_proxy_path),
        "frame_stride": int(frame_stride),
        "actionable_lead_frames": int(actionable_lead_frames),
        "actionable_lead_seconds": round(actionable_lead_frames / 10.0, 3),
        "clearance_threshold_m": float(clearance_threshold_m),
        "margin_threshold_m": float(margin_threshold_m),
        "oracle_proxy_tolerance_us": int(oracle_proxy_tolerance_us),
        "baseline_collision_scene_count": len(scenes),
        "baseline_first_collision_component_counts": dict(sorted(components.items())),
        "frame_status_counts": dict(sorted(status_counts.items())),
        "bucket_counts": dict(sorted(buckets.items())),
        "evaluable_frame_count": len(evaluable),
        "actionable_frame_count": len(actionable),
        "token_candidate_set": {
            "proxy_safe_actionable_frame_count": len(token_safe),
            "selected_proxy_safe_actionable_frame_count": len(token_selected_safe),
            "margin_safe_actionable_frame_count": len(token_margin),
            "selected_margin_safe_actionable_frame_count": len(token_selected_margin),
            "scene_count_with_proxy_safe_actionable_frame": sum(
                scene["bucket"]["token_proxy_safe_actionable_frame_count"] > 0 for scene in scenes
            ),
            "scene_count_with_selected_proxy_safe_actionable_frame": sum(
                scene["bucket"]["token_selected_proxy_safe_actionable_frame_count"] > 0 for scene in scenes
            ),
            "scene_count_with_margin_safe_actionable_frame": sum(
                scene["bucket"]["token_margin_safe_actionable_frame_count"] > 0 for scene in scenes
            ),
            "scene_count_with_selected_margin_safe_actionable_frame": sum(
                scene["bucket"]["token_selected_margin_safe_actionable_frame_count"] > 0 for scene in scenes
            ),
        },
        "direct_grid": {
            "proxy_safe_actionable_frame_count": len(direct_safe),
            "selected_proxy_safe_actionable_frame_count": len(direct_selected_safe),
            "cost_missed_proxy_safe_actionable_frame_count": len(direct_cost_miss),
            "margin_safe_actionable_frame_count": len(direct_margin),
            "selected_margin_safe_actionable_frame_count": len(direct_selected_margin),
            "cost_missed_margin_safe_actionable_frame_count": len(direct_cost_miss_margin),
            "scene_count_with_proxy_safe_actionable_frame": sum(
                scene["bucket"]["direct_proxy_safe_actionable_frame_count"] > 0 for scene in scenes
            ),
            "scene_count_with_selected_proxy_safe_actionable_frame": sum(
                scene["bucket"]["direct_selected_proxy_safe_actionable_frame_count"] > 0 for scene in scenes
            ),
            "scene_count_with_cost_missed_proxy_safe_actionable_frame": sum(
                scene["bucket"]["direct_cost_missed_proxy_safe_actionable_frame_count"] > 0 for scene in scenes
            ),
            "scene_count_with_margin_safe_actionable_frame": sum(
                scene["bucket"]["direct_margin_safe_actionable_frame_count"] > 0 for scene in scenes
            ),
            "scene_count_with_selected_margin_safe_actionable_frame": sum(
                scene["bucket"]["direct_selected_margin_safe_actionable_frame_count"] > 0 for scene in scenes
            ),
            "scene_count_with_cost_missed_margin_safe_actionable_frame": sum(
                scene["bucket"]["direct_cost_missed_margin_safe_actionable_frame_count"] > 0 for scene in scenes
            ),
        },
        "hazards_per_evaluable_frame": _distribution([frame.get("hazard_count") for frame in evaluable]),
        "oracle_proxy_delta_us": _distribution([frame.get("oracle_proxy_delta_us") for frame in evaluable]),
        "first_direct_proxy_safe_lead_frames": _distribution(
            [scene["bucket"].get("first_direct_proxy_safe_lead_frames") for scene in scenes]
        ),
        "first_token_proxy_safe_lead_frames": _distribution(
            [scene["bucket"].get("first_token_proxy_safe_lead_frames") for scene in scenes]
        ),
    }


def _load_scene_dirs(run_dir: Path) -> dict[str, Path]:
    return {
        path.name: path
        for path in sorted(run_dir.glob("[0-9][0-9][0-9]_*"))
        if path.is_dir()
    }


def _metric_timeline(scene_dir: Path) -> list[dict[str, Any]]:
    table = pq.read_table(
        scene_dir / "aggregate" / "metrics_unprocessed.parquet",
        columns=["name", "timestamps_us", "values", "valid"],
    )
    payload = table.to_pydict()
    by_timestamp: dict[int, dict[str, float]] = {}
    collision_timestamps = []
    for name, timestamp, value, valid in zip(
        payload["name"], payload["timestamps_us"], payload["values"], payload["valid"]
    ):
        if not valid or value is None:
            continue
        timestamp_us = int(timestamp)
        by_timestamp.setdefault(timestamp_us, {})[str(name)] = float(value)
        if name == "collision_any":
            collision_timestamps.append(timestamp_us)
    return [
        {
            "frame_index": index,
            "timestamp_us": timestamp_us,
            "metrics": by_timestamp.get(timestamp_us, {}),
        }
        for index, timestamp_us in enumerate(collision_timestamps, start=1)
    ]


def _first_collision_from_timeline(timeline: list[dict[str, Any]]) -> dict[str, Any] | None:
    for frame in timeline:
        metrics = frame["metrics"]
        if metrics.get("collision_any", 0.0) <= 0.0:
            continue
        components = [
            component
            for component in COLLISION_COMPONENTS
            if metrics.get(component, 0.0) > 0.0
        ]
        return {
            "frame_index": frame["frame_index"],
            "timestamp_us": frame["timestamp_us"],
            "components": components,
        }
    return None


def _timestamp_for_frame(timeline: list[dict[str, Any]], frame_index: int) -> int | None:
    if frame_index < 1 or frame_index > len(timeline):
        return None
    return int(timeline[frame_index - 1]["timestamp_us"])


def _load_selection_log(path: Path) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    summary = {
        "exists": path.is_file(),
        "good_json_lines": 0,
        "bad_json_lines": 0,
        "rows_by_frame": 0,
    }
    if not path.is_file():
        return rows, summary
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                summary["bad_json_lines"] += 1
                continue
            summary["good_json_lines"] += 1
            frame = row.get("frame_index")
            if frame is None:
                continue
            rows[int(frame)] = row
    summary["rows_by_frame"] = len(rows)
    return rows, summary


def _load_controller_rows(controller_dir: Path) -> list[dict[str, float | int]]:
    files = sorted(controller_dir.glob("*.csv"))
    if not files:
        return []
    rows: list[dict[str, float | int]] = []
    with files[0].open(newline="", encoding="utf-8") as handle:
        for item in csv.DictReader(handle):
            timestamp = item.get("timestamp_us")
            if timestamp is None:
                continue
            qx = _as_float(item.get("qx"), default=0.0)
            qy = _as_float(item.get("qy"), default=0.0)
            qz = _as_float(item.get("qz"), default=0.0)
            qw = _as_float(item.get("qw"), default=1.0)
            yaw = math.atan2(
                2.0 * (qw * qz + qx * qy),
                1.0 - 2.0 * (qy * qy + qz * qz),
            )
            rows.append(
                {
                    "timestamp_us": int(float(timestamp)),
                    "world_x": _as_float(item.get("x"), default=0.0),
                    "world_y": _as_float(item.get("y"), default=0.0),
                    "world_heading": yaw,
                    "world_vx": _as_float(item.get("vx"), default=0.0),
                    "world_vy": _as_float(item.get("vy"), default=0.0),
                }
            )
    rows.sort(key=lambda row: int(row["timestamp_us"]))
    return rows


def _nearest_controller_row(
    rows: list[dict[str, float | int]],
    timestamp_us: int | None,
) -> dict[str, float | int] | None:
    if timestamp_us is None or not rows:
        return None
    return min(rows, key=lambda row: abs(int(row["timestamp_us"]) - int(timestamp_us)))


def _index_proxy_frames_by_scene(
    frames: dict[int, dict[str, Any]],
) -> dict[str, dict[int, dict[str, Any]]]:
    by_scene: dict[str, dict[int, dict[str, Any]]] = {}
    for timestamp_us, frame in frames.items():
        scene_id = frame.get("scene_id")
        if not scene_id:
            continue
        by_scene.setdefault(str(scene_id), {})[int(timestamp_us)] = frame
    return by_scene


def _nearest_scene_proxy_frame(
    scene_frames: dict[int, dict[str, Any]],
    timestamp_us: int,
    *,
    tolerance_us: int,
) -> tuple[dict[str, Any] | None, int | None]:
    if not scene_frames:
        return None, None
    nearest_timestamp = min(scene_frames, key=lambda item: abs(int(item) - int(timestamp_us)))
    delta = abs(int(nearest_timestamp) - int(timestamp_us))
    if delta > tolerance_us:
        return None, delta
    return scene_frames[nearest_timestamp], delta


def _closest_hazard(hazards: list[dict[str, float | str]]) -> dict[str, Any] | None:
    if not hazards:
        return None
    hazard = min(hazards, key=lambda item: math.hypot(float(item["x"]), float(item["y"])))
    return {
        "label": hazard.get("label"),
        "x": _round(hazard.get("x")),
        "y": _round(hazard.get("y")),
        "radius": _round(hazard.get("radius")),
        "distance_m": _round(math.hypot(float(hazard["x"]), float(hazard["y"]))),
        "vx": _round(hazard.get("vx")),
        "vy": _round(hazard.get("vy")),
    }


def _clipgt_id(scene_name: str) -> str:
    return scene_name.split("_", 1)[1] if "_" in scene_name else scene_name


def _max_lead(frames: list[dict[str, Any]]) -> int | None:
    if not frames:
        return None
    return max(int(frame.get("lead_frames", 0)) for frame in frames)


def _metric_float(value: Any) -> float:
    if value == "inf":
        return math.inf
    if value == "-inf":
        return -math.inf
    return _as_float(value)


def _as_float(value: Any, *, default: float = math.nan) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else parsed


def _sort_float(value: Any) -> float:
    parsed = _metric_float(value)
    if math.isnan(parsed):
        return -math.inf
    return parsed


def _round(value: Any, digits: int = 4) -> float | str | None:
    parsed = _metric_float(value)
    if math.isnan(parsed):
        return None
    if math.isinf(parsed):
        return "inf" if parsed > 0 else "-inf"
    return round(parsed, digits)


def _distribution(values: list[Any]) -> dict[str, Any]:
    parsed = [_metric_float(value) for value in values]
    parsed = [value for value in parsed if not math.isnan(value)]
    if not parsed:
        return {"count": 0}
    finite = [value for value in parsed if math.isfinite(value)]
    if not finite:
        return {"count": len(parsed), "min": "inf", "median": "inf", "mean": "inf", "max": "inf"}
    return {
        "count": len(parsed),
        "min": round(min(parsed), 4) if math.isfinite(min(parsed)) else "-inf",
        "median": round(statistics.median(finite), 4),
        "mean": round(statistics.fmean(finite), 4),
        "max": round(max(parsed), 4) if math.isfinite(max(parsed)) else "inf",
    }


def _markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    token = summary["token_candidate_set"]
    direct = summary["direct_grid"]
    lines = [
        "# AlpaSim action-space upper-bound audit",
        "",
        "This audit reconstructs the token candidate set and the selector-free direct grid on the "
        "baseline collision timeline after injecting the world-frame oracle actor proxy. It is an "
        "open-loop proxy upper bound, not a counterfactual AlpaSim physics replay.",
        "",
        "## Summary",
        "",
        f"- Baseline collision scenes audited: `{summary['baseline_collision_scene_count']}`.",
        f"- Baseline first-impact components: `{summary['baseline_first_collision_component_counts']}`.",
        f"- Frame status counts: `{summary['frame_status_counts']}`.",
        f"- Actionable window: at least `{summary['actionable_lead_frames']}` frames "
        f"(`{summary['actionable_lead_seconds']:.1f}s`) before impact.",
        f"- Proxy no-overlap threshold: `{summary['clearance_threshold_m']:.2f}m`; "
        f"reported margin threshold: `{summary['margin_threshold_m']:.2f}m`.",
        f"- Token candidate set proxy-safe actionable frames: "
        f"`{token['proxy_safe_actionable_frame_count']}/{summary['actionable_frame_count']}` "
        f"across `{token['scene_count_with_proxy_safe_actionable_frame']}` scenes.",
        f"- Token selected proxy-safe actionable frames: "
        f"`{token['selected_proxy_safe_actionable_frame_count']}` "
        f"across `{token['scene_count_with_selected_proxy_safe_actionable_frame']}` scenes.",
        f"- Token selected margin-safe actionable frames: "
        f"`{token['selected_margin_safe_actionable_frame_count']}` "
        f"across `{token['scene_count_with_selected_margin_safe_actionable_frame']}` scenes.",
        f"- Direct grid proxy-safe actionable frames: "
        f"`{direct['proxy_safe_actionable_frame_count']}/{summary['actionable_frame_count']}` "
        f"across `{direct['scene_count_with_proxy_safe_actionable_frame']}` scenes.",
        f"- Direct grid cost-selected proxy-safe actionable frames: "
        f"`{direct['selected_proxy_safe_actionable_frame_count']}` "
        f"across `{direct['scene_count_with_selected_proxy_safe_actionable_frame']}` scenes.",
        f"- Direct grid cost-selected margin-safe actionable frames: "
        f"`{direct['selected_margin_safe_actionable_frame_count']}` "
        f"across `{direct['scene_count_with_selected_margin_safe_actionable_frame']}` scenes.",
        f"- Direct grid cost-missed proxy-safe actionable frames: "
        f"`{direct['cost_missed_proxy_safe_actionable_frame_count']}` "
        f"across `{direct['scene_count_with_cost_missed_proxy_safe_actionable_frame']}` scenes.",
        f"- Direct grid cost-missed margin-safe actionable frames: "
        f"`{direct['cost_missed_margin_safe_actionable_frame_count']}` "
        f"across `{direct['scene_count_with_cost_missed_margin_safe_actionable_frame']}` scenes.",
        f"- Bucket counts: `{summary['bucket_counts']}`.",
        "",
        "## Interpretation",
        "",
        "If the direct grid contains proxy-safe actions but its cost rejects them, the selector-free "
        "planner is likely under-tuned or incorrectly weighted. If the direct grid cost selects "
        "proxy-safe actions before impact but the actual closed-loop run still collides, the likely "
        "bottleneck moves to controller execution, receding-horizon dynamics, or simulator mismatch. "
        "If neither token candidates nor the direct grid contain proxy-safe actions, the available "
        "action space is too weak for this collision surface.",
        "",
        "## Per-scene buckets",
        "",
        "| Scene | Component | Bucket | Token safe frames | Token selected safe | Direct safe frames | Direct selected safe | Direct cost misses | First direct safe lead |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for scene in report["scenes"]:
        bucket = scene["bucket"]
        component = "+".join(scene["baseline_first_collision"].get("components", [])) or "collision"
        lines.append(
            "| "
            f"{scene['scene']} | "
            f"{component} | "
            f"{bucket['label']} | "
            f"{bucket['token_proxy_safe_actionable_frame_count']} | "
            f"{bucket['token_selected_proxy_safe_actionable_frame_count']} | "
            f"{bucket['direct_proxy_safe_actionable_frame_count']} | "
            f"{bucket['direct_selected_proxy_safe_actionable_frame_count']} | "
            f"{bucket['direct_cost_missed_proxy_safe_actionable_frame_count']} | "
            f"{bucket['first_direct_proxy_safe_lead_frames']} |"
        )
    lines.extend(["", "## Impact-frame snapshot", ""])
    lines.append(
        "| Scene | Lead | Token selected safe | Token safe count | Direct selected safe | Direct safe count | Direct selected clearance | Direct best clearance |"
    )
    lines.append("| --- | ---: | --- | ---: | --- | ---: | ---: | ---: |")
    for scene in report["scenes"]:
        impact = next((frame for frame in scene["frames"] if int(frame.get("lead_frames", -1)) == 0), None)
        if impact is None or impact.get("status") != "ok":
            continue
        token_frame = impact.get("token_candidate_set") or {}
        direct_frame = impact.get("direct_grid") or {}
        lines.append(
            "| "
            f"{scene['scene']} | "
            f"{impact.get('lead_frames')} | "
            f"{token_frame.get('selected_collision_free_proxy')} | "
            f"{token_frame.get('collision_free_token_count')} | "
            f"{direct_frame.get('selected_collision_free_proxy')} | "
            f"{direct_frame.get('collision_free_candidate_count')} | "
            f"{direct_frame.get('selected_min_clearance_m')} | "
            f"{direct_frame.get('best_clearance_m')} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
