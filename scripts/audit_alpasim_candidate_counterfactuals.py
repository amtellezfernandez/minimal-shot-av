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

from minimal_shot_av.simulator.alpasim_signal import scenario_from_command
from minimal_shot_av.simulator.alpasim_token_bc import (
    _actor_axis_constraint_violation,
    _adapter_spotlight_config,
    _axis_constraint_violation,
    _candidate_axis_signals,
    _current_ego_velocity_world,
    _load_oracle_actor_proxy,
    _world_actor_to_current_hazard,
)
from minimal_shot_av.simulator.environment import scenario_at_tick
from minimal_shot_av.simulator.perception import perceive_scene
from minimal_shot_av.simulator.spotlight_reflex import evaluate_maneuver_candidates
from minimal_shot_av.simulator.world_model import update_world_state


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = ROOT / "runs" / "alpasim_axis_constrained_30scene_eval"
DEFAULT_CANDIDATE = (
    ROOT
    / "runs"
    / "alpasim_actor_blindness_30scene_raw"
    / "token_dagger_iter2_axis_constrained_oracle_actor_clamped__front_camera_30scene_merged"
)
DEFAULT_ORACLE_PROXY = ROOT / "artifacts" / "alpasim_oracle_actor_proxy_30scene.json"
DEFAULT_JSON = ROOT / "artifacts" / "alpasim_candidate_counterfactual_30scene.json"
DEFAULT_MD = ROOT / "artifacts" / "alpasim_candidate_counterfactual_30scene.md"

COLLISION_COMPONENTS = ("collision_front", "collision_lateral", "collision_rear")
DEFAULT_LEAD_FRAMES = (20, 10, 5, 2, 1, 0)
ROUTE_STABLE_TOKENS = frozenset({"stop", "crawl", "maintain", "slow_yield", "lane_recover"})


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct selector-side candidate counterfactuals for baseline AlpaSim collision "
            "frames by injecting the world-frame oracle actor proxy into the baseline timeline. "
            "This is not a physics/controller replay."
        )
    )
    parser.add_argument("--baseline-run-dir", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--candidate-run-dir", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--oracle-actor-proxy", type=Path, default=DEFAULT_ORACLE_PROXY)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_MD)
    parser.add_argument(
        "--lead-frames",
        default=",".join(str(item) for item in DEFAULT_LEAD_FRAMES),
        help="Comma-separated frame leads before first impact to audit, assuming 10 Hz AlpaSim logs.",
    )
    parser.add_argument(
        "--actionable-lead-frames",
        type=int,
        default=5,
        help="Minimum lead in frames before impact to count as actionable for bucket summaries.",
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
    lead_frames = _parse_lead_frames(args.lead_frames)
    report = audit_candidate_counterfactuals(
        baseline_run_dir=args.baseline_run_dir,
        candidate_run_dir=args.candidate_run_dir,
        oracle_actor_proxy_path=args.oracle_actor_proxy,
        lead_frames=lead_frames,
        actionable_lead_frames=args.actionable_lead_frames,
        oracle_proxy_tolerance_us=args.oracle_proxy_tolerance_us,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = _markdown(report)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text(markdown + "\n", encoding="utf-8")
    print(markdown)
    return 0


def audit_candidate_counterfactuals(
    *,
    baseline_run_dir: Path,
    candidate_run_dir: Path,
    oracle_actor_proxy_path: Path,
    lead_frames: tuple[int, ...],
    actionable_lead_frames: int,
    oracle_proxy_tolerance_us: int,
) -> dict[str, Any]:
    baseline_run_dir = baseline_run_dir.resolve()
    candidate_run_dir = candidate_run_dir.resolve()
    oracle_actor_proxy_path = oracle_actor_proxy_path.resolve()
    proxy_frames, _proxy_timestamps = _load_oracle_actor_proxy(oracle_actor_proxy_path)
    proxy_by_scene = _index_proxy_frames_by_scene(proxy_frames)

    baseline_scenes = _load_scene_dirs(baseline_run_dir)
    candidate_scenes = _load_scene_dirs(candidate_run_dir)
    rows: list[dict[str, Any]] = []
    for scene_name, baseline_scene_dir in baseline_scenes.items():
        timeline = _metric_timeline(baseline_scene_dir)
        first_collision = _first_collision_from_timeline(timeline)
        if first_collision is None:
            continue
        selection_rows, selection_summary = _load_selection_log(
            baseline_scene_dir / "driver" / "selection-log.jsonl"
        )
        controller_rows = _load_controller_rows(baseline_scene_dir / "controller")
        candidate_first_collision = None
        if scene_name in candidate_scenes:
            candidate_timeline = _metric_timeline(candidate_scenes[scene_name])
            candidate_first_collision = _first_collision_from_timeline(candidate_timeline)

        frame_rows = []
        for lead in lead_frames:
            frame_index = max(1, int(first_collision["frame_index"]) - int(lead))
            frame_timestamp = _timestamp_for_frame(timeline, frame_index)
            selection_row = selection_rows.get(frame_index)
            controller_row = _nearest_controller_row(controller_rows, frame_timestamp)
            frame_rows.append(
                _audit_frame(
                    scene_name=scene_name,
                    frame_index=frame_index,
                    lead_frames=int(first_collision["frame_index"]) - frame_index,
                    timestamp_us=frame_timestamp,
                    selection_row=selection_row,
                    controller_row=controller_row,
                    proxy_scene_frames=proxy_by_scene.get(_clipgt_id(scene_name), {}),
                    oracle_proxy_tolerance_us=oracle_proxy_tolerance_us,
                )
            )

        rows.append(
            {
                "scene": scene_name,
                "clipgt_id": _clipgt_id(scene_name),
                "baseline_first_collision": first_collision,
                "candidate_first_collision": candidate_first_collision,
                "baseline_selection_log": selection_summary,
                "frames": frame_rows,
                "bucket": _bucket_scene(frame_rows, actionable_lead_frames=actionable_lead_frames),
            }
        )

    summary = _summary(
        rows,
        baseline_run_dir=baseline_run_dir,
        candidate_run_dir=candidate_run_dir,
        oracle_actor_proxy_path=oracle_actor_proxy_path,
        lead_frames=lead_frames,
        actionable_lead_frames=actionable_lead_frames,
        oracle_proxy_tolerance_us=oracle_proxy_tolerance_us,
    )
    return {
        "schema": "alpasim_candidate_counterfactual_audit_v1",
        "summary": summary,
        "scenes": rows,
    }


def _audit_frame(
    *,
    scene_name: str,
    frame_index: int,
    lead_frames: int,
    timestamp_us: int | None,
    selection_row: dict[str, Any] | None,
    controller_row: dict[str, Any] | None,
    proxy_scene_frames: dict[int, dict[str, Any]],
    oracle_proxy_tolerance_us: int,
) -> dict[str, Any]:
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
        return base

    ego_pose = {
        "world_x": controller_row["world_x"],
        "world_y": controller_row["world_y"],
        "world_heading": controller_row["world_heading"],
        "world_vx": controller_row["world_vx"],
        "world_vy": controller_row["world_vy"],
    }
    speed_mps = _as_float(selection_row.get("speed_mps"), default=0.0)
    ego_velocity = _current_ego_velocity_world(ego_pose, speed_mps=speed_mps)
    hazards = []
    for index, actor in enumerate(proxy_frame.get("world_actors", []) or []):
        hazard = _world_actor_to_current_hazard(actor, ego_pose, ego_velocity, index)
        if hazard is not None:
            hazards.append(hazard)

    axis_eval = _evaluate_selection_frame(
        selection_row,
        hazards,
        selection_mode=str(selection_row.get("selection_mode", "axis_constrained")),
        safety_mode="axis",
    )
    actor_axis_eval = _evaluate_selection_frame(
        selection_row,
        hazards,
        selection_mode="actor_axis_constrained",
        safety_mode="actor_axis",
    )
    closest_hazard = _closest_hazard(hazards)
    base.update(
        {
            "baseline_selected_token": selection_row.get("hybrid_token"),
            "baseline_spotlight_token": selection_row.get("spotlight_token"),
            "baseline_dagger_argmax_token": selection_row.get("dagger_argmax_token"),
            "speed_mps": speed_mps,
            "oracle_proxy_hit": True,
            "oracle_proxy_delta_us": proxy_delta_us,
            "oracle_proxy_actor_count": len(proxy_frame.get("world_actors", []) or []),
            "hazard_count": len(hazards),
            "closest_hazard": closest_hazard,
            "ego_pose": {
                "world_x": round(float(ego_pose["world_x"]), 4),
                "world_y": round(float(ego_pose["world_y"]), 4),
                "world_heading": round(float(ego_pose["world_heading"]), 6),
            },
            "axis_proxy": axis_eval,
            "actor_axis_proxy": actor_axis_eval,
        }
    )
    return base


def _evaluate_selection_frame(
    selection_row: dict[str, Any],
    hazards: list[dict[str, float | str]],
    *,
    selection_mode: str,
    safety_mode: str,
) -> dict[str, Any]:
    signal = dict(selection_row.get("alpasim_signal") or {})
    signal["structured_hazards"] = list(signal.get("structured_hazards") or []) + list(hazards)
    signal["oracle_actor_proxy_hit"] = True
    signal["oracle_actor_proxy_count"] = len(hazards)
    command = str(selection_row.get("command", "straight"))
    speed_mps = _as_float(selection_row.get("speed_mps"), default=0.0)
    trajectory_mode = str(selection_row.get("trajectory_mode", "clamped_lateral"))
    max_lateral_offset_m = _as_float(selection_row.get("max_lateral_offset_m"), default=2.0)
    config = _adapter_spotlight_config(
        trajectory_mode=trajectory_mode,
        max_lateral_offset_m=max_lateral_offset_m,
        selection_mode=selection_mode,
    )
    scenario = scenario_at_tick(scenario_from_command(command, signal), 0)
    position = scenario.start
    perception = perceive_scene(scenario, position)
    world_state = update_world_state(scenario, position, perception)
    evaluations, reference_count = evaluate_maneuver_candidates(
        scenario,
        position,
        world_state,
        perception,
        speed_mps=speed_mps,
        config=config,
    )
    axis_signals = (
        _candidate_axis_signals(
            evaluations,
            scenario=scenario,
            position=position,
            speed_mps=speed_mps,
            config=config,
        )
        if safety_mode == "actor_axis"
        else {}
    )

    selected_token = str(selection_row.get("hybrid_token"))
    candidate_rows = []
    for evaluation in evaluations:
        token = evaluation.candidate.name
        signal_row = axis_signals.get(token)
        if safety_mode == "actor_axis":
            violation = _actor_axis_constraint_violation(token, evaluation, signal_row)
        else:
            violation = _axis_constraint_violation(token, evaluation)
        candidate_rows.append(_candidate_summary(evaluation, signal_row, violation))
    candidate_rows.sort(key=lambda item: item["effective_score"], reverse=True)
    by_token = {row["candidate"]: row for row in candidate_rows}
    safe_tokens = [row["candidate"] for row in candidate_rows if row["violation"] is None]
    moving_safe_tokens = [
        row["candidate"]
        for row in candidate_rows
        if row["violation"] is None and row["candidate"] not in {"stop", "crawl"}
    ]
    selected = by_token.get(selected_token)
    best_safe = next((row for row in candidate_rows if row["violation"] is None), None)
    route_stable_safe = [
        row["candidate"]
        for row in candidate_rows
        if row["violation"] is None and row["candidate"] in ROUTE_STABLE_TOKENS
    ]
    return {
        "safety_mode": safety_mode,
        "selection_mode": selection_mode,
        "reference_count": reference_count,
        "safe_tokens": safe_tokens,
        "moving_safe_tokens": moving_safe_tokens,
        "route_stable_safe_tokens": route_stable_safe,
        "best_token": candidate_rows[0]["candidate"] if candidate_rows else None,
        "best_safe_token": best_safe["candidate"] if best_safe else None,
        "selected_token": selected_token,
        "selected_violation": selected["violation"] if selected else "missing_candidate",
        "selected_is_safe": selected is not None and selected["violation"] is None,
        "selected_candidate": selected,
        "candidate_summaries": candidate_rows,
    }


def _candidate_summary(
    evaluation: Any,
    axis_signal: dict[str, Any] | None,
    violation: str | None,
) -> dict[str, Any]:
    explanation = evaluation.explanation
    row: dict[str, Any] = {
        "candidate": evaluation.candidate.name,
        "effective_score": _round(explanation.effective_score),
        "selector_score": _round(explanation.selector_score),
        "action_clearance_m": _round(explanation.action_clearance_m),
        "horizon_clearance_m": _round(explanation.horizon_clearance_m),
        "safety_penalty": _round(explanation.safety_penalty),
        "horizon_clearance_penalty": _round(explanation.horizon_clearance_penalty),
        "near_clearance_penalty": _round(explanation.near_clearance_penalty),
        "stop_penalty": _round(explanation.stop_penalty),
        "progress_bonus": _round(explanation.progress_bonus),
        "inside_3s_region": bool(explanation.inside_3s_region),
        "inside_5s_region": bool(explanation.inside_5s_region),
        "violation": violation,
    }
    if axis_signal:
        row.update(
            {
                "actor_action_clearance_m": _round(axis_signal.get("actor_action_clearance_m")),
                "actor_horizon_clearance_m": _round(axis_signal.get("actor_horizon_clearance_m")),
                "static_action_clearance_m": _round(axis_signal.get("static_action_clearance_m")),
                "static_horizon_clearance_m": _round(axis_signal.get("static_horizon_clearance_m")),
                "lane_margin_m": _round(axis_signal.get("lane_margin_m")),
                "route_deviation_m": _round(axis_signal.get("route_deviation_m")),
                "rear_actor_count": int(_as_float(axis_signal.get("rear_actor_count"), default=0.0)),
                "rear_closing_actor_count": int(
                    _as_float(axis_signal.get("rear_closing_actor_count"), default=0.0)
                ),
                "rear_flow_gap_m": _round(axis_signal.get("rear_flow_gap_m")),
                "rear_flow_ttc_s": _round(axis_signal.get("rear_flow_ttc_s")),
                "rear_flow_risk": bool(axis_signal.get("rear_flow_risk", False)),
                "candidate_mean_forward_speed_mps": _round(
                    axis_signal.get("candidate_mean_forward_speed_mps")
                ),
            }
        )
    return row


def _bucket_scene(frame_rows: list[dict[str, Any]], *, actionable_lead_frames: int) -> dict[str, Any]:
    evaluable = [row for row in frame_rows if row.get("status") == "ok"]
    actionable = [row for row in evaluable if int(row.get("lead_frames", 0)) >= actionable_lead_frames]
    actor_safe = [
        row
        for row in actionable
        if (row.get("actor_axis_proxy") or {}).get("safe_tokens")
    ]
    actor_miss = [
        row
        for row in actor_safe
        if (row.get("actor_axis_proxy") or {}).get("selected_violation") is not None
    ]
    actor_selected_safe = [
        row
        for row in actor_safe
        if (row.get("actor_axis_proxy") or {}).get("selected_is_safe") is True
    ]
    axis_safe = [
        row
        for row in actionable
        if (row.get("axis_proxy") or {}).get("safe_tokens")
    ]
    impact = next((row for row in evaluable if int(row.get("lead_frames", -1)) == 0), None)

    if not evaluable:
        label = "insufficient_logs"
    elif not actor_safe:
        label = "no_actor_axis_safe_candidate_actionable"
    elif actor_miss and not actor_selected_safe:
        label = "actor_axis_safe_candidate_missed_by_baseline_selector"
    elif actor_miss:
        label = "mixed_actor_axis_safe_candidate_intermittently_missed"
    else:
        label = "baseline_selected_actor_axis_safe_candidate_but_collision_persisted"

    return {
        "label": label,
        "evaluable_frame_count": len(evaluable),
        "actionable_frame_count": len(actionable),
        "axis_proxy_actionable_safe_frame_count": len(axis_safe),
        "actor_axis_proxy_actionable_safe_frame_count": len(actor_safe),
        "actor_axis_proxy_missed_safe_frame_count": len(actor_miss),
        "actor_axis_proxy_selected_safe_frame_count": len(actor_selected_safe),
        "actor_axis_safe_preimpact": bool(actor_safe),
        "axis_safe_preimpact": bool(axis_safe),
        "actor_axis_safe_at_impact": bool(
            impact and (impact.get("actor_axis_proxy") or {}).get("safe_tokens")
        ),
        "axis_safe_at_impact": bool(impact and (impact.get("axis_proxy") or {}).get("safe_tokens")),
    }


def _summary(
    rows: list[dict[str, Any]],
    *,
    baseline_run_dir: Path,
    candidate_run_dir: Path,
    oracle_actor_proxy_path: Path,
    lead_frames: tuple[int, ...],
    actionable_lead_frames: int,
    oracle_proxy_tolerance_us: int,
) -> dict[str, Any]:
    buckets = Counter(row["bucket"]["label"] for row in rows)
    baseline_components = Counter(
        component
        for row in rows
        for component in row["baseline_first_collision"].get("components", [])
    )
    candidate_collision_count = sum(row["candidate_first_collision"] is not None for row in rows)
    evaluable_frames = [
        frame
        for row in rows
        for frame in row["frames"]
        if frame.get("status") == "ok"
    ]
    actionable_frames = [
        frame
        for frame in evaluable_frames
        if int(frame.get("lead_frames", 0)) >= actionable_lead_frames
    ]
    axis_safe_frames = [
        frame
        for frame in actionable_frames
        if (frame.get("axis_proxy") or {}).get("safe_tokens")
    ]
    actor_safe_frames = [
        frame
        for frame in actionable_frames
        if (frame.get("actor_axis_proxy") or {}).get("safe_tokens")
    ]
    actor_selected_safe_frames = [
        frame
        for frame in actor_safe_frames
        if (frame.get("actor_axis_proxy") or {}).get("selected_is_safe") is True
    ]
    actor_miss_frames = [
        frame
        for frame in actor_safe_frames
        if (frame.get("actor_axis_proxy") or {}).get("selected_violation") is not None
    ]
    impact_frames = [
        frame
        for frame in evaluable_frames
        if int(frame.get("lead_frames", -1)) == 0
    ]
    status_counts = Counter(
        frame.get("status", "unknown")
        for row in rows
        for frame in row["frames"]
    )
    return {
        "baseline_run_dir": str(baseline_run_dir),
        "candidate_run_dir": str(candidate_run_dir),
        "oracle_actor_proxy_path": str(oracle_actor_proxy_path),
        "lead_frames": list(lead_frames),
        "actionable_lead_frames": actionable_lead_frames,
        "actionable_lead_seconds": round(actionable_lead_frames / 10.0, 3),
        "oracle_proxy_tolerance_us": oracle_proxy_tolerance_us,
        "baseline_collision_scene_count": len(rows),
        "candidate_collision_scene_count_for_same_baseline_collision_set": candidate_collision_count,
        "baseline_first_collision_component_counts": dict(sorted(baseline_components.items())),
        "frame_status_counts": dict(sorted(status_counts.items())),
        "bucket_counts": dict(sorted(buckets.items())),
        "actionable_frame_count": len(actionable_frames),
        "axis_proxy_actionable_safe_frame_count": len(axis_safe_frames),
        "actor_axis_proxy_actionable_safe_frame_count": len(actor_safe_frames),
        "actor_axis_proxy_selected_safe_frame_count": len(actor_selected_safe_frames),
        "actor_axis_proxy_missed_safe_frame_count": len(actor_miss_frames),
        "impact_frame_count": len(impact_frames),
        "axis_proxy_safe_at_impact_count": sum(
            bool((frame.get("axis_proxy") or {}).get("safe_tokens")) for frame in impact_frames
        ),
        "actor_axis_proxy_safe_at_impact_count": sum(
            bool((frame.get("actor_axis_proxy") or {}).get("safe_tokens")) for frame in impact_frames
        ),
        "hazards_per_evaluable_frame": _distribution(
            [frame.get("hazard_count") for frame in evaluable_frames]
        ),
        "oracle_proxy_delta_us": _distribution(
            [frame.get("oracle_proxy_delta_us") for frame in evaluable_frames]
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
    timeline = []
    for index, timestamp_us in enumerate(collision_timestamps, start=1):
        metrics = by_timestamp.get(timestamp_us, {})
        timeline.append(
            {
                "frame_index": index,
                "timestamp_us": timestamp_us,
                "metrics": metrics,
            }
        )
    return timeline


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
    with path.open(encoding="utf-8") as handle:
        for line in handle:
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
        reader = csv.DictReader(handle)
        for item in reader:
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


def _parse_lead_frames(raw: str) -> tuple[int, ...]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(max(0, int(item)))
    return tuple(sorted(set(values), reverse=True))


def _as_float(value: Any, *, default: float = math.nan) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _round(value: Any, digits: int = 4) -> float | str | None:
    parsed = _as_float(value)
    if math.isnan(parsed):
        return None
    if math.isinf(parsed):
        return "inf" if parsed > 0 else "-inf"
    return round(parsed, digits)


def _distribution(values: list[Any]) -> dict[str, Any]:
    parsed = [_as_float(value) for value in values]
    parsed = [value for value in parsed if not math.isnan(value)]
    if not parsed:
        return {"count": 0}
    return {
        "count": len(parsed),
        "min": round(min(parsed), 4),
        "median": round(statistics.median(parsed), 4),
        "mean": round(statistics.fmean(parsed), 4),
        "max": round(max(parsed), 4),
    }


def _markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# AlpaSim 30-scene candidate-counterfactual audit",
        "",
        "This audit injects the world-frame oracle actor proxy into the baseline timeline and "
        "reconstructs selector-side candidate feasibility. It does not replay candidates through "
        "the AlpaSim controller or traffic simulator.",
        "",
        "## Summary",
        "",
        f"- Baseline collision scenes audited: `{summary['baseline_collision_scene_count']}`.",
        f"- Same-scene oracle collisions among those scenes: "
        f"`{summary['candidate_collision_scene_count_for_same_baseline_collision_set']}`.",
        f"- Baseline first-impact components: `{summary['baseline_first_collision_component_counts']}`.",
        f"- Frame status counts: `{summary['frame_status_counts']}`.",
        f"- Actionable window: at least `{summary['actionable_lead_frames']}` frames "
        f"(`{summary['actionable_lead_seconds']:.1f}s`) before first impact.",
        f"- Axis-proxy actionable safe frames: "
        f"`{summary['axis_proxy_actionable_safe_frame_count']}/{summary['actionable_frame_count']}`.",
        f"- Actor-axis actionable safe frames: "
        f"`{summary['actor_axis_proxy_actionable_safe_frame_count']}/{summary['actionable_frame_count']}`.",
        f"- Actor-axis safe frame selected by baseline policy: "
        f"`{summary['actor_axis_proxy_selected_safe_frame_count']}`.",
        f"- Actor-axis safe frame missed by baseline policy: "
        f"`{summary['actor_axis_proxy_missed_safe_frame_count']}`.",
        f"- Axis-proxy safe at impact: "
        f"`{summary['axis_proxy_safe_at_impact_count']}/{summary['impact_frame_count']}`.",
        f"- Actor-axis safe at impact: "
        f"`{summary['actor_axis_proxy_safe_at_impact_count']}/{summary['impact_frame_count']}`.",
        f"- Bucket counts: `{summary['bucket_counts']}`.",
        "",
        "## Interpretation",
        "",
        "The ordinary axis scorer often still marks candidates safe after actor injection, which "
        "explains why actor visibility alone did not remove rear collisions. The stricter actor-axis "
        "test is more diagnostic: when it finds no actionable safe candidate, the existing candidate "
        "set/controller envelope is the likely bottleneck; when it finds a safe candidate that the "
        "baseline-selected token violates, the result is only a selector-side miss until a controller "
        "replay confirms that candidate actually avoids impact.",
        "",
        "## Per-scene buckets",
        "",
        "| Scene | Component | Bucket | Actor-axis safe frames | Missed safe frames | Impact actor-safe | Notes |",
        "| --- | --- | --- | ---: | ---: | --- | --- |",
    ]
    for row in report["scenes"]:
        bucket = row["bucket"]
        component = "+".join(row["baseline_first_collision"].get("components", [])) or "collision"
        notes = _scene_notes(row)
        lines.append(
            "| "
            f"{row['scene']} | "
            f"{component} | "
            f"{bucket['label']} | "
            f"{bucket['actor_axis_proxy_actionable_safe_frame_count']} | "
            f"{bucket['actor_axis_proxy_missed_safe_frame_count']} | "
            f"{bucket['actor_axis_safe_at_impact']} | "
            f"{notes} |"
        )
    lines.extend(
        [
            "",
            "## Frame Detail",
            "",
        ]
    )
    for row in report["scenes"]:
        lines.append(f"### {row['scene']}")
        lines.append("")
        lines.append("| Lead | Selected | Axis safe | Actor-axis safe | Actor selected violation | Best actor-safe | Status |")
        lines.append("| ---: | --- | --- | --- | --- | --- | --- |")
        for frame in row["frames"]:
            axis = frame.get("axis_proxy") or {}
            actor = frame.get("actor_axis_proxy") or {}
            lines.append(
                "| "
                f"{frame.get('lead_frames')} | "
                f"{frame.get('baseline_selected_token', '')} | "
                f"{_token_list(axis.get('safe_tokens'))} | "
                f"{_token_list(actor.get('safe_tokens'))} | "
                f"{actor.get('selected_violation')} | "
                f"{actor.get('best_safe_token')} | "
                f"{frame.get('status')} |"
            )
        lines.append("")
    return "\n".join(lines)


def _scene_notes(row: dict[str, Any]) -> str:
    frames = [frame for frame in row["frames"] if frame.get("status") == "ok"]
    if not frames:
        return "no evaluable proxy frames"
    first_safe = next(
        (
            frame
            for frame in frames
            if (frame.get("actor_axis_proxy") or {}).get("safe_tokens")
        ),
        None,
    )
    if first_safe is None:
        return "no actor-axis-safe candidate in audited frames"
    actor = first_safe.get("actor_axis_proxy") or {}
    return (
        f"earliest actor-safe lead {first_safe.get('lead_frames')}f; "
        f"best safe {actor.get('best_safe_token')}; selected violation {actor.get('selected_violation')}"
    )


def _token_list(value: Any) -> str:
    if not value:
        return "--"
    if isinstance(value, list):
        return ",".join(str(item) for item in value[:4])
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
