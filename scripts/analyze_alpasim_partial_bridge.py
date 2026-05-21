#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ACTION_SPACE = ROOT / "artifacts" / "alpasim_action_space_upper_bound_30scene.json"
DEFAULT_COLLISION_SURFACE = ROOT / "artifacts" / "corl2027" / "alpasim_collision_surface_30scene_audit.json"
DEFAULT_ORACLE_PROXY = ROOT / "artifacts" / "alpasim_oracle_actor_proxy_30scene.json"
DEFAULT_INTERNAL = ROOT / "artifacts" / "internal_proxy_transfer_medium.json"
DEFAULT_JSON = ROOT / "artifacts" / "corl2027" / "alpasim_partial_bridge_preimpact.json"
DEFAULT_MD = ROOT / "artifacts" / "corl2027" / "alpasim_partial_bridge_preimpact.md"
DEFAULT_SVG = ROOT / "artifacts" / "corl2027" / "alpasim_partial_bridge_preimpact.svg"

EGO_RADIUS_M = 0.35
ACTIONABLE_LEAD_FRAMES = 5


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Bridge committed AlpaSim pre-impact oracle-proxy audits to the controlled "
            "internal proxy-perturbation regimes without rerunning AlpaSim."
        )
    )
    parser.add_argument("--action-space-json", type=Path, default=DEFAULT_ACTION_SPACE)
    parser.add_argument("--collision-surface-json", type=Path, default=DEFAULT_COLLISION_SURFACE)
    parser.add_argument("--oracle-proxy-json", type=Path, default=DEFAULT_ORACLE_PROXY)
    parser.add_argument("--internal-proxy-json", type=Path, default=DEFAULT_INTERNAL)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_MD)
    parser.add_argument("--output-svg", type=Path, default=DEFAULT_SVG)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = analyze_partial_bridge(
        action_space_path=args.action_space_json,
        collision_surface_path=args.collision_surface_json,
        oracle_proxy_path=args.oracle_proxy_json,
        internal_proxy_path=args.internal_proxy_json,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_markdown.write_text(_markdown(report) + "\n", encoding="utf-8")
    args.output_svg.write_text(_svg(report), encoding="utf-8")
    print(_markdown(report))
    return 0


def analyze_partial_bridge(
    *,
    action_space_path: Path,
    collision_surface_path: Path,
    oracle_proxy_path: Path,
    internal_proxy_path: Path,
) -> dict[str, Any]:
    action_space = _load_json(action_space_path)
    collision_surface = _load_json(collision_surface_path)
    oracle_proxy = _load_json(oracle_proxy_path)
    internal_proxy = _load_json(internal_proxy_path)

    collision_frames = _collision_preimpact_frames(action_space)
    impact_frames = [frame for frame in collision_frames if frame["lead_frames"] == 0]
    actionable_frames = [frame for frame in collision_frames if frame["lead_frames"] >= ACTIONABLE_LEAD_FRAMES]
    collision_scene_ids = {
        _clipgt_id_from_scene(scene["scene"])
        for scene in collision_surface["scenes"]
        if scene.get("baseline_collision")
    }
    noncollision_scene_ids = {
        _clipgt_id_from_scene(scene["scene"])
        for scene in collision_surface["scenes"]
        if not scene.get("baseline_collision")
    }

    proxy_frames_by_scene = _oracle_proxy_frames_by_scene(oracle_proxy)
    terminal_control_frames = _terminal_matched_control_frames(
        proxy_frames_by_scene=proxy_frames_by_scene,
        noncollision_scene_ids=noncollision_scene_ids,
        lead_frames=[frame["lead_frames"] for frame in actionable_frames],
    )

    first_impact_adapter = _first_impact_adapter_summary(
        collision_surface=collision_surface,
        action_space=action_space,
    )
    collision_actionable = _frame_summary(actionable_frames)
    collision_impact = _frame_summary(impact_frames)
    noncollision_terminal = _control_summary(terminal_control_frames)
    perturbation_map = _internal_perturbation_map(internal_proxy)

    bridge = _bridge_interpretation(
        first_impact_adapter=first_impact_adapter,
        collision_actionable=collision_actionable,
        noncollision_terminal=noncollision_terminal,
        perturbation_map=perturbation_map,
    )

    return {
        "schema": "alpasim_partial_bridge_preimpact_v1",
        "inputs": {
            "action_space_json": str(action_space_path),
            "collision_surface_json": str(collision_surface_path),
            "oracle_proxy_json": str(oracle_proxy_path),
            "internal_proxy_json": str(internal_proxy_path),
        },
        "limitations": {
            "structured_hazard_definition": (
                "The baseline count uses the logged adapter field alpasim_signal.structured_hazards. "
                "In the adapter code path this field is forwarded directly from prediction_input when present; "
                "the bridge therefore measures hazard omission in the logged structured-hazard interface, "
                "not a downstream score threshold."
            ),
            "adapter_state_scope": (
                "Committed audits preserve exact adapter-vs-oracle actor omission at first impact, "
                "but not full raw adapter alpasim_signal for every 30-scene frame."
            ),
            "control_matching": (
                "Non-collision controls are matched by lead-to-terminal frame offset using the same 30-scene "
                "oracle actor proxy. They are not additionally matched on scene class or ego speed."
            ),
            "preimpact_scope": (
                "Actionable pre-impact frames are reconstructed for the 18 baseline collision clips. "
                "Non-collision controls use terminal-frame oracle-proxy actor load from the same 30-scene proxy."
            ),
        },
        "scene_counts": {
            "collision_scenes": len(collision_scene_ids),
            "noncollision_scenes": len(noncollision_scene_ids),
        },
        "frame_counts": {
            "collision_preimpact_frames": len(collision_frames),
            "collision_impact_frames": len(impact_frames),
            "collision_actionable_frames": len(actionable_frames),
            "terminal_matched_noncollision_control_frames": len(terminal_control_frames),
        },
        "first_impact_adapter_residual": first_impact_adapter,
        "collision_actionable_oracle_load": collision_actionable,
        "collision_impact_oracle_load": collision_impact,
        "noncollision_terminal_oracle_load": noncollision_terminal,
        "internal_perturbation_regimes": perturbation_map,
        "bridge_interpretation": bridge,
    }


def _collision_preimpact_frames(action_space: dict[str, Any]) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    for scene in action_space.get("scenes", []):
        component = (scene.get("baseline_first_collision") or {}).get("components", ["unknown"])[0]
        for frame in scene.get("frames", []):
            if frame.get("status") != "ok":
                continue
            closest = frame.get("closest_hazard") or {}
            direct_grid = frame.get("direct_grid") or {}
            token_set = frame.get("token_candidate_set") or {}
            frames.append(
                {
                    "scene": scene.get("scene"),
                    "clipgt_id": scene.get("clipgt_id"),
                    "component": component,
                    "frame_index": _to_int(frame.get("frame_index")),
                    "lead_frames": _to_int(frame.get("lead_frames")),
                    "lead_seconds": _to_float(frame.get("lead_seconds")),
                    "hazard_count": _to_int(frame.get("hazard_count")),
                    "oracle_proxy_actor_count": _to_int(frame.get("oracle_proxy_actor_count")),
                    "closest_distance_m": _to_float(closest.get("distance_m")),
                    "closest_clearance_m": _closest_clearance(closest),
                    "closest_x_m": _to_float(closest.get("x")),
                    "closest_y_m": _to_float(closest.get("y")),
                    "closest_vx_mps": _to_float(closest.get("vx")),
                    "closest_vy_mps": _to_float(closest.get("vy")),
                    "closest_is_rear": _is_rear(closest),
                    "baseline_selected_token": frame.get("baseline_selected_token"),
                    "direct_selected_clearance_m": _to_float(direct_grid.get("selected_min_clearance_m")),
                    "direct_best_clearance_m": _to_float(direct_grid.get("best_clearance_m")),
                    "direct_selected_collision_free_proxy": direct_grid.get("selected_collision_free_proxy"),
                    "token_selected_clearance_m": _to_float(token_set.get("selected_horizon_clearance_m")),
                    "token_best_clearance_m": _to_float(token_set.get("best_clearance_m")),
                    "token_selected_collision_free_proxy": token_set.get("selected_collision_free_proxy"),
                }
            )
    return frames


def _terminal_matched_control_frames(
    *,
    proxy_frames_by_scene: dict[str, list[dict[str, Any]]],
    noncollision_scene_ids: set[str],
    lead_frames: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scene_id in sorted(noncollision_scene_ids):
        frames = proxy_frames_by_scene.get(scene_id, [])
        if not frames:
            continue
        last_index = len(frames) - 1
        for lead in lead_frames:
            index = last_index - lead
            if index < 0 or index >= len(frames):
                continue
            frame = frames[index]
            rows.append(_oracle_proxy_frame_metrics(frame, scene_id=scene_id, lead_frames=lead))
    return rows


def _oracle_proxy_frames_by_scene(oracle_proxy: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    frames_by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for frame in oracle_proxy.get("frames", {}).values():
        scene_id = frame.get("scene_id")
        if not scene_id:
            continue
        frames_by_scene[str(scene_id)].append(frame)
    for frames in frames_by_scene.values():
        frames.sort(key=lambda item: int(item.get("timestamp_us", 0)))
    return dict(frames_by_scene)


def _oracle_proxy_frame_metrics(frame: dict[str, Any], *, scene_id: str, lead_frames: int) -> dict[str, Any]:
    actors = frame.get("world_actors") or []
    closest: dict[str, Any] | None = None
    closest_clearance = math.inf
    rear_count = 0
    for actor in actors:
        rel_x = _to_float(actor.get("source_rel_x"))
        rel_y = _to_float(actor.get("source_rel_y"))
        radius = _to_float(actor.get("radius"), default=0.0)
        if rel_x is None or rel_y is None:
            continue
        clearance = math.hypot(rel_x, rel_y) - radius - EGO_RADIUS_M
        if rel_x < 0.0 and abs(rel_y) <= 3.0:
            rear_count += 1
        if clearance < closest_clearance:
            closest_clearance = clearance
            closest = actor
    return {
        "scene_id": scene_id,
        "lead_frames": lead_frames,
        "hazard_count": len(actors),
        "rear_lane_actor_count": rear_count,
        "closest_clearance_m": closest_clearance if math.isfinite(closest_clearance) else None,
        "closest_distance_m": (
            math.hypot(float(closest.get("source_rel_x")), float(closest.get("source_rel_y")))
            if closest is not None
            and closest.get("source_rel_x") is not None
            and closest.get("source_rel_y") is not None
            else None
        ),
        "closest_is_rear": (
            bool(
                closest is not None
                and _to_float(closest.get("source_rel_x"), default=0.0) < 0.0
                and abs(_to_float(closest.get("source_rel_y"), default=math.inf)) <= 3.0
            )
        ),
    }


def _first_impact_adapter_summary(
    *,
    collision_surface: dict[str, Any],
    action_space: dict[str, Any],
) -> dict[str, Any]:
    impact_by_clip = {}
    for scene in action_space.get("scenes", []):
        clip_id = scene.get("clipgt_id")
        impact = [frame for frame in scene.get("frames", []) if frame.get("lead_frames") == 0]
        if clip_id and impact:
            impact_by_clip[clip_id] = impact[0]

    rows = []
    for scene in collision_surface.get("scenes", []):
        if not scene.get("baseline_collision"):
            continue
        clip_id = _clipgt_id_from_scene(scene.get("scene", ""))
        baseline_sel = scene.get("baseline_selection_at_first_collision") or {}
        impact = impact_by_clip.get(clip_id) or {}
        rows.append(
            {
                "scene": scene.get("scene"),
                "clipgt_id": clip_id,
                "adapter_structured_hazard_count": _to_int(
                    baseline_sel.get("structured_hazard_count"), default=None
                ),
                "oracle_hazard_count_at_same_impact": _to_int(impact.get("hazard_count"), default=None),
                "oracle_closest_clearance_m": _closest_clearance(impact.get("closest_hazard") or {}),
                "oracle_closest_is_rear": _is_rear(impact.get("closest_hazard") or {}),
                "baseline_top_candidate": baseline_sel.get("top_candidate"),
                "baseline_selected_token": baseline_sel.get("selected_token"),
            }
        )
    adapter_counts = [row["adapter_structured_hazard_count"] for row in rows]
    oracle_counts = [row["oracle_hazard_count_at_same_impact"] for row in rows]
    return {
        "scene_count": len(rows),
        "adapter_zero_structured_hazards": sum(count == 0 for count in adapter_counts),
        "oracle_positive_hazards_at_same_impact": sum((count or 0) > 0 for count in oracle_counts),
        "adapter_hazard_count": _distribution(adapter_counts),
        "oracle_hazard_count_at_same_impact": _distribution(oracle_counts),
        "oracle_closest_clearance_m": _distribution(
            [row["oracle_closest_clearance_m"] for row in rows]
        ),
        "oracle_closest_rear_count": sum(row["oracle_closest_is_rear"] for row in rows),
        "top_candidate_maintain": sum(row["baseline_top_candidate"] == "maintain" for row in rows),
        "rows": rows,
    }


def _frame_summary(frames: list[dict[str, Any]]) -> dict[str, Any]:
    if not frames:
        return {"count": 0}
    component_counts = Counter(str(frame.get("component")) for frame in frames)
    scene_count = len({frame.get("clipgt_id") for frame in frames})
    low_direct = [
        frame for frame in frames
        if _is_number(frame.get("direct_selected_clearance_m"))
        and float(frame["direct_selected_clearance_m"]) < 0.55
    ]
    low_token = [
        frame for frame in frames
        if _is_number(frame.get("token_selected_clearance_m"))
        and float(frame["token_selected_clearance_m"]) < 0.55
    ]
    return {
        "count": len(frames),
        "scene_count": scene_count,
        "component_counts": dict(sorted(component_counts.items())),
        "lead_frames": _distribution([frame.get("lead_frames") for frame in frames]),
        "hazard_count": _distribution([frame.get("hazard_count") for frame in frames]),
        "oracle_proxy_actor_count": _distribution([frame.get("oracle_proxy_actor_count") for frame in frames]),
        "closest_distance_m": _distribution([frame.get("closest_distance_m") for frame in frames]),
        "closest_clearance_m": _distribution([frame.get("closest_clearance_m") for frame in frames]),
        "closest_rear_rate": sum(bool(frame.get("closest_is_rear")) for frame in frames) / len(frames),
        "direct_selected_clearance_m": _distribution(
            [frame.get("direct_selected_clearance_m") for frame in frames]
        ),
        "direct_best_clearance_m": _distribution([frame.get("direct_best_clearance_m") for frame in frames]),
        "direct_selected_low_margin_rate_lt_0p55": len(low_direct) / len(frames),
        "direct_selected_collision_free_rate": _mean_bool(
            [frame.get("direct_selected_collision_free_proxy") for frame in frames]
        ),
        "token_selected_clearance_m": _distribution(
            [frame.get("token_selected_clearance_m") for frame in frames]
        ),
        "token_best_clearance_m": _distribution([frame.get("token_best_clearance_m") for frame in frames]),
        "token_selected_low_margin_rate_lt_0p55": len(low_token) / len(frames),
        "token_selected_collision_free_rate": _mean_bool(
            [frame.get("token_selected_collision_free_proxy") for frame in frames]
        ),
    }


def _control_summary(frames: list[dict[str, Any]]) -> dict[str, Any]:
    if not frames:
        return {"count": 0}
    return {
        "count": len(frames),
        "scene_count": len({frame.get("scene_id") for frame in frames}),
        "lead_frames": _distribution([frame.get("lead_frames") for frame in frames]),
        "hazard_count": _distribution([frame.get("hazard_count") for frame in frames]),
        "rear_lane_actor_count": _distribution([frame.get("rear_lane_actor_count") for frame in frames]),
        "closest_distance_m": _distribution([frame.get("closest_distance_m") for frame in frames]),
        "closest_clearance_m": _distribution([frame.get("closest_clearance_m") for frame in frames]),
        "closest_rear_rate": sum(bool(frame.get("closest_is_rear")) for frame in frames) / len(frames),
    }


def _internal_perturbation_map(internal_proxy: dict[str, Any]) -> dict[str, Any]:
    regimes: dict[str, Any] = {}
    for perturbation, payload in sorted(internal_proxy.get("by_perturbation", {}).items()):
        summary = payload.get("summary", {})
        raw = summary.get("raw_iter2", {})
        hybrid = summary.get("hybrid_veto_iter2", {})
        clamped = summary.get("clamped_iter2", {})
        regimes[perturbation] = {
            "raw_iter2": _agent_metrics(raw),
            "hybrid_veto_iter2": _agent_metrics(hybrid),
            "clamped_iter2": _agent_metrics(clamped),
            "hybrid_delta_vs_raw": {
                "collision": _delta(hybrid.get("collision"), raw.get("collision")),
                "lane_violation_any": _delta(hybrid.get("lane_violation_any"), raw.get("lane_violation_any")),
                "pass": _delta(hybrid.get("pass"), raw.get("pass")),
            },
            "clamped_delta_vs_raw": {
                "collision": _delta(clamped.get("collision"), raw.get("collision")),
                "lane_violation_any": _delta(clamped.get("lane_violation_any"), raw.get("lane_violation_any")),
                "pass": _delta(clamped.get("pass"), raw.get("pass")),
            },
        }
    return regimes


def _agent_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "pass": payload.get("pass"),
        "collision": payload.get("collision"),
        "lane_violation_any": payload.get("lane_violation_any"),
        "min_clearance_m": payload.get("min_clearance_m"),
        "max_lane_error_m": payload.get("max_lane_error_m"),
    }


def _bridge_interpretation(
    *,
    first_impact_adapter: dict[str, Any],
    collision_actionable: dict[str, Any],
    noncollision_terminal: dict[str, Any],
    perturbation_map: dict[str, Any],
) -> dict[str, Any]:
    impact_exact = (
        first_impact_adapter.get("adapter_zero_structured_hazards") == first_impact_adapter.get("scene_count")
        and first_impact_adapter.get("oracle_positive_hazards_at_same_impact", 0) > 0
    )
    collision_median_hazards = (collision_actionable.get("hazard_count") or {}).get("median")
    control_median_hazards = (noncollision_terminal.get("hazard_count") or {}).get("median")
    collision_median_clearance = (collision_actionable.get("closest_clearance_m") or {}).get("median")
    control_median_clearance = (noncollision_terminal.get("closest_clearance_m") or {}).get("median")
    latency = perturbation_map.get("latency_3", {})
    return {
        "supported_bridge_claim": (
            "The committed audits support a collision-window hazard-dropout bridge: baseline adapter logs "
            "have zero structured hazards at first impact while the world-frame oracle proxy has positive "
            "actor hazards at the same impact frames."
        ),
        "not_supported_without_rerun": (
            "The committed audits do not support full per-frame route, heading, lane-scale, or feature-noise "
            "residual distributions; those require regenerating raw adapter selection logs."
        ),
        "impact_actor_omission_exact": impact_exact,
        "collision_vs_control_actor_load": {
            "collision_actionable_median_hazard_count": collision_median_hazards,
            "terminal_control_median_hazard_count": control_median_hazards,
            "collision_actionable_median_closest_clearance_m": collision_median_clearance,
            "terminal_control_median_closest_clearance_m": control_median_clearance,
        },
        "controlled_regime_alignment": {
            "closest_supported_regime": "latency_3 / actor-state corruption",
            "regime_raw_collision": (latency.get("raw_iter2") or {}).get("collision"),
            "regime_hybrid_collision": (latency.get("hybrid_veto_iter2") or {}).get("collision"),
            "reason": (
                "AlpaSim impact residual is a hazard-dropout signal: the baseline adapter records no structured "
                "actors at impact while the oracle proxy records actors. That aligns qualitatively with the "
                "controlled actor-latency/dropout failure mode, but not yet with route-offset or lane-scale "
                "residuals unless raw adapter logs are regenerated."
            ),
        },
    }


def _markdown(report: dict[str, Any]) -> str:
    impact = report["first_impact_adapter_residual"]
    action = report["collision_actionable_oracle_load"]
    control = report["noncollision_terminal_oracle_load"]
    interp = report["bridge_interpretation"]
    regimes = report["internal_perturbation_regimes"]
    latency = regimes.get("latency_3", {})
    lines = [
        "# AlpaSim partial bridge: pre-impact adapter residuals",
        "",
        "This analysis uses only committed artifacts. It does not require rerunning AlpaSim.",
        "",
        "## What is exact",
        "",
        (
            f"- First-impact collision scenes: `{impact['scene_count']}`; baseline adapter "
            f"structured hazards are zero in `{impact['adapter_zero_structured_hazards']}/"
            f"{impact['scene_count']}`."
        ),
        (
            f"- At the same baseline impact frames, the world-frame oracle proxy has positive "
            f"hazards in `{impact['oracle_positive_hazards_at_same_impact']}/"
            f"{impact['scene_count']}` scenes."
        ),
        "- The baseline count is the logged `alpasim_signal.structured_hazards` field, not a downstream score threshold.",
        (
            f"- Median oracle hazard count at impact: "
            f"`{_fmt((impact['oracle_hazard_count_at_same_impact'] or {}).get('median'))}`; "
            f"median closest clearance: "
            f"`{_fmt((impact['oracle_closest_clearance_m'] or {}).get('median'))} m`."
        ),
        "",
        "## Pre-impact window",
        "",
        (
            f"- Collision actionable frames: `{action['count']}` over `{action['scene_count']}` scenes "
            f"(lead >= `{ACTIONABLE_LEAD_FRAMES}` frames)."
        ),
        (
            f"- Median oracle hazard count: `{_fmt((action['hazard_count'] or {}).get('median'))}`; "
            f"median closest clearance: `{_fmt((action['closest_clearance_m'] or {}).get('median'))} m`; "
            f"closest actor is rear-lane in `{_fmt(action.get('closest_rear_rate'), digits=3)}` of frames."
        ),
        (
            f"- Direct-grid selected proxy-collision-free rate: "
            f"`{_fmt(action.get('direct_selected_collision_free_rate'), digits=3)}`; "
            f"low-margin (<0.55 m) selected rate: "
            f"`{_fmt(action.get('direct_selected_low_margin_rate_lt_0p55'), digits=3)}`."
        ),
        "",
        "## Terminal-matched non-collision control",
        "",
        (
            f"- Control frames: `{control['count']}` from `{control['scene_count']}` non-collision scenes, "
            "matched by lead-to-end frame offsets only."
        ),
        (
            f"- Median oracle hazard count: `{_fmt((control['hazard_count'] or {}).get('median'))}`; "
            f"median closest clearance: `{_fmt((control['closest_clearance_m'] or {}).get('median'))} m`; "
            f"closest actor is rear-lane in `{_fmt(control.get('closest_rear_rate'), digits=3)}` of frames."
        ),
        "",
        "## Controlled-perturbation alignment",
        "",
        (
            f"- Internal `latency_3` raw collision is "
            f"`{_fmt((latency.get('raw_iter2') or {}).get('collision'), digits=3)}`; "
            f"hybrid collision is "
            f"`{_fmt((latency.get('hybrid_veto_iter2') or {}).get('collision'), digits=3)}`."
        ),
        f"- Supported bridge claim: {interp['supported_bridge_claim']}",
        f"- Not supported without rerun: {interp['not_supported_without_rerun']}",
        "",
        "## Paper-safe interpretation",
        "",
        (
            "The committed audits support a partial bridge for collision-window hazard dropout in the "
            "structured-hazard interface. They do not yet support a full residual-density bridge for route "
            "offset, heading bias, lane scale, or feature noise. A full regenerate should therefore be "
            "targeted at raw adapter logs for those axes, not at redesigning the method."
        ),
    ]
    return "\n".join(lines)


def _svg(report: dict[str, Any]) -> str:
    action = report["collision_actionable_oracle_load"]
    control = report["noncollision_terminal_oracle_load"]
    latency = report["internal_perturbation_regimes"].get("latency_3", {})
    values = [
        ("AlpaSim collision\npre-impact", (action["hazard_count"] or {}).get("median"), "#9b2226"),
        ("Non-collision\nterminal control", (control["hazard_count"] or {}).get("median"), "#005f73"),
    ]
    max_value = max([float(v or 0.0) for _, v, _ in values] + [1.0])
    width, height = 720, 360
    chart_x, chart_y = 70, 70
    chart_w, chart_h = 560, 210
    bar_w = 150
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbf8ef"/>',
        '<text x="70" y="34" font-family="Georgia,serif" font-size="20" fill="#1f2933">Partial bridge: oracle actor load in collision-critical windows</text>',
        '<line x1="70" y1="280" x2="630" y2="280" stroke="#243b53" stroke-width="1"/>',
        '<line x1="70" y1="70" x2="70" y2="280" stroke="#243b53" stroke-width="1"/>',
        '<text x="22" y="78" font-family="Arial,sans-serif" font-size="12" fill="#52606d">median</text>',
        '<text x="14" y="94" font-family="Arial,sans-serif" font-size="12" fill="#52606d">hazards</text>',
    ]
    for i, (label, value, color) in enumerate(values):
        numeric = float(value or 0.0)
        x = chart_x + 90 + i * 250
        h = (numeric / max_value) * chart_h
        y = chart_y + chart_h - h
        svg.append(f'<rect x="{x}" y="{y:.1f}" width="{bar_w}" height="{h:.1f}" fill="{color}" opacity="0.88"/>')
        svg.append(f'<text x="{x + bar_w / 2}" y="{y - 8:.1f}" text-anchor="middle" font-family="Arial,sans-serif" font-size="18" fill="#102a43">{numeric:.1f}</text>')
        for j, part in enumerate(label.split("\\n")):
            svg.append(f'<text x="{x + bar_w / 2}" y="{306 + j * 15}" text-anchor="middle" font-family="Arial,sans-serif" font-size="13" fill="#334e68">{part}</text>')
    raw_collision = (latency.get("raw_iter2") or {}).get("collision")
    hybrid_collision = (latency.get("hybrid_veto_iter2") or {}).get("collision")
    svg.append(
        f'<text x="70" y="342" font-family="Arial,sans-serif" font-size="13" fill="#52606d">'
        f'Controlled actor-latency regime: raw collision {float(raw_collision or 0):.3f}, '
        f'hybrid collision {float(hybrid_collision or 0):.3f}. Full route/lane residuals require rerun.'
        '</text>'
    )
    svg.append("</svg>")
    return "\n".join(svg)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _clipgt_id_from_scene(scene: str) -> str:
    parts = str(scene).split("_", 1)
    return parts[1] if len(parts) == 2 and parts[1].startswith("clipgt-") else str(scene)


def _closest_clearance(closest: dict[str, Any]) -> float | None:
    distance = _to_float(closest.get("distance_m"))
    radius = _to_float(closest.get("radius"), default=0.0)
    if distance is None:
        return None
    return distance - float(radius or 0.0) - EGO_RADIUS_M


def _is_rear(closest: dict[str, Any]) -> bool:
    x = _to_float(closest.get("x"))
    y = _to_float(closest.get("y"))
    return bool(x is not None and y is not None and x < 0.0 and abs(y) <= 3.0)


def _distribution(values: list[Any]) -> dict[str, Any]:
    numeric = [float(value) for value in values if _is_number(value)]
    if not numeric:
        return {"count": 0}
    numeric.sort()
    return {
        "count": len(numeric),
        "mean": statistics.fmean(numeric),
        "median": statistics.median(numeric),
        "min": numeric[0],
        "p10": _percentile(numeric, 0.10),
        "p25": _percentile(numeric, 0.25),
        "p75": _percentile(numeric, 0.75),
        "p90": _percentile(numeric, 0.90),
        "max": numeric[-1],
    }


def _percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return math.nan
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = p * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _mean_bool(values: list[Any]) -> float | None:
    clean = [value for value in values if isinstance(value, bool)]
    if not clean:
        return None
    return sum(clean) / len(clean)


def _delta(candidate: Any, baseline: Any) -> float | None:
    if not _is_number(candidate) or not _is_number(baseline):
        return None
    return float(candidate) - float(baseline)


def _to_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(result):
        return default
    return result


def _to_int(value: Any, default: int | None = 0) -> int | None:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _is_number(value: Any) -> bool:
    if value is None:
        return False
    try:
        result = float(value)
    except (TypeError, ValueError):
        return False
    return not math.isnan(result)


def _fmt(value: Any, *, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if not _is_number(value):
        return str(value)
    return f"{float(value):.{digits}f}"


if __name__ == "__main__":
    raise SystemExit(main())
