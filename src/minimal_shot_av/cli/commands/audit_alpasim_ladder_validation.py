#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ACTION_SPACE = ROOT / "artifacts" / "alpasim_action_space_upper_bound_30scene.json"
DEFAULT_COUNTERFACTUAL = ROOT / "artifacts" / "corl2027" / "alpasim_candidate_counterfactual_30scene.json"
DEFAULT_OUTPUT_JSON = ROOT / "artifacts" / "corl2027" / "alpasim_ladder_validation.json"
DEFAULT_OUTPUT_MARKDOWN = ROOT / "artifacts" / "corl2027" / "alpasim_ladder_validation.md"

SELECTOR_BUCKETS = {
    "actor_axis_safe_candidate_missed_by_baseline_selector",
    "mixed_actor_axis_safe_candidate_intermittently_missed",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate AlpaSim diagnostic-ladder localization from committed per-frame artifacts. "
            "Reports clearance-divergence sensitivity and scene subsets for selector-ranking vs "
            "controller/proxy residual modes."
        )
    )
    parser.add_argument("--action-space-upper-bound", type=Path, default=DEFAULT_ACTION_SPACE)
    parser.add_argument("--candidate-counterfactual", type=Path, default=DEFAULT_COUNTERFACTUAL)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_OUTPUT_MARKDOWN)
    parser.add_argument("--actionable-lead-frames", type=int, default=5)
    parser.add_argument("--realized-near-thresholds", default="1,2,3,4,5,6")
    parser.add_argument("--proxy-clearance-thresholds", default="0,0.55,1.0")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = build_validation_report(
        action_space=_load_json(args.action_space_upper_bound),
        counterfactual=_load_json(args.candidate_counterfactual),
        actionable_lead_frames=args.actionable_lead_frames,
        realized_near_thresholds=_parse_float_list(args.realized_near_thresholds),
        proxy_clearance_thresholds=_parse_float_list(args.proxy_clearance_thresholds),
        source_paths={
            "action_space_upper_bound": args.action_space_upper_bound,
            "candidate_counterfactual": args.candidate_counterfactual,
        },
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = markdown_report(report)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text(markdown + "\n", encoding="utf-8")
    print(markdown)


def build_validation_report(
    *,
    action_space: dict[str, Any],
    counterfactual: dict[str, Any],
    actionable_lead_frames: int,
    realized_near_thresholds: tuple[float, ...],
    proxy_clearance_thresholds: tuple[float, ...],
    source_paths: dict[str, Path] | None = None,
) -> dict[str, Any]:
    action_scenes = {str(scene["scene"]): scene for scene in action_space.get("scenes", [])}
    counter_scenes = {str(scene["scene"]): scene for scene in counterfactual.get("scenes", [])}
    scene_rows = []
    for scene_name, action_scene in sorted(action_scenes.items()):
        counter_scene = counter_scenes.get(scene_name, {})
        scene_rows.append(
            _scene_row(
                action_scene,
                counter_scene,
                actionable_lead_frames=actionable_lead_frames,
                proxy_clearance_thresholds=proxy_clearance_thresholds,
            )
        )

    return {
        "schema": "alpasim_ladder_validation_v1",
        "source_paths": {key: str(path) for key, path in (source_paths or {}).items()},
        "method_note": (
            "Uses committed per-frame audit artifacts. Realized current clearance is the transformed "
            "ego-frame closest actor-proxy clearance at the rollout frame; proxy clearance is the "
            "selected direct-grid future minimum clearance stored by the open-loop audit."
        ),
        "config": {
            "actionable_lead_frames": actionable_lead_frames,
            "realized_near_thresholds_m": list(realized_near_thresholds),
            "proxy_clearance_thresholds_m": list(proxy_clearance_thresholds),
        },
        "summary": {
            "scene_count": len(scene_rows),
            "mode_counts": dict(Counter(row["assigned_mode"] for row in scene_rows)),
            "clearance_divergence": _clearance_divergence_summary(
                action_space.get("scenes", []),
                actionable_lead_frames=actionable_lead_frames,
                realized_near_thresholds=realized_near_thresholds,
            ),
            "proxy_clearance_erosion": _proxy_clearance_erosion_summary(
                action_space.get("scenes", []),
                proxy_clearance_thresholds=proxy_clearance_thresholds,
            ),
            "subset_characterization": _subset_characterization(scene_rows),
        },
        "scenes": scene_rows,
    }


def _scene_row(
    action_scene: dict[str, Any],
    counter_scene: dict[str, Any],
    *,
    actionable_lead_frames: int,
    proxy_clearance_thresholds: tuple[float, ...],
) -> dict[str, Any]:
    frames = [frame for frame in action_scene.get("frames", []) if frame.get("status") == "ok"]
    actionable = [frame for frame in frames if int(frame.get("lead_frames", 0)) >= actionable_lead_frames]
    counter_bucket = (counter_scene.get("bucket") or {}).get("label")
    assigned_mode = _assigned_mode(str(counter_bucket or ""), action_scene)
    realized_clearances = [_realized_current_clearance(frame) for frame in actionable]
    realized_clearances = [value for value in realized_clearances if value is not None]
    selected_proxy = [_direct_selected_clearance(frame) for frame in actionable]
    selected_proxy = [value for value in selected_proxy if value is not None]
    threshold_erosion = {
        str(threshold): _scene_erosion(action_scene, threshold=threshold)
        for threshold in proxy_clearance_thresholds
    }
    return {
        "scene": action_scene.get("scene"),
        "clipgt_id": action_scene.get("clipgt_id"),
        "assigned_mode": assigned_mode,
        "counterfactual_bucket": counter_bucket,
        "first_collision": action_scene.get("baseline_first_collision"),
        "actionable_frame_count": len(actionable),
        "median_speed_mps": _distribution([frame.get("speed_mps") for frame in actionable]).get("median"),
        "median_hazard_count": _distribution([frame.get("hazard_count") for frame in actionable]).get("median"),
        "median_realized_current_clearance_m": _distribution(realized_clearances).get("median"),
        "median_direct_selected_proxy_clearance_m": _distribution(selected_proxy).get("median"),
        "proxy_clearance_erosion": threshold_erosion,
    }


def _assigned_mode(counter_bucket: str, action_scene: dict[str, Any]) -> str:
    if counter_bucket in SELECTOR_BUCKETS:
        return "selector_ranking_secondary"
    if counter_bucket == "no_actor_axis_safe_candidate_actionable":
        return "candidate_feasibility_or_actor_axis_definition"
    direct = action_scene.get("bucket", {})
    if direct.get("direct_selected_proxy_safe_actionable_frame_count", 0) > 0:
        return "controller_proxy_residual"
    return "mixed_or_insufficient"


def _clearance_divergence_summary(
    scenes: list[dict[str, Any]],
    *,
    actionable_lead_frames: int,
    realized_near_thresholds: tuple[float, ...],
) -> dict[str, Any]:
    output = {}
    for threshold in realized_near_thresholds:
        hit_scenes = []
        hit_frames = 0
        onset_leads = []
        for scene in scenes:
            frames = [
                frame
                for frame in scene.get("frames", [])
                if frame.get("status") == "ok" and int(frame.get("lead_frames", 0)) >= actionable_lead_frames
            ]
            hits = [
                frame
                for frame in frames
                if _direct_selected_safe(frame)
                and (_realized_current_clearance(frame) is not None)
                and (_realized_current_clearance(frame) <= threshold)
            ]
            if hits:
                hit_scenes.append(scene.get("scene"))
                hit_frames += len(hits)
                onset_leads.append(max(int(frame["lead_frames"]) for frame in hits))
        output[str(threshold)] = {
            "scene_count": len(hit_scenes),
            "frame_count": hit_frames,
            "scene_rate": _safe_rate(len(hit_scenes), len(scenes)),
            "median_onset_lead_frames": _median(onset_leads),
            "median_onset_lead_seconds": _median([lead / 10.0 for lead in onset_leads]),
            "scenes": hit_scenes,
        }
    return output


def _proxy_clearance_erosion_summary(
    scenes: list[dict[str, Any]],
    *,
    proxy_clearance_thresholds: tuple[float, ...],
) -> dict[str, Any]:
    output = {}
    for threshold in proxy_clearance_thresholds:
        rows = [_scene_erosion(scene, threshold=threshold) for scene in scenes]
        eroded = [row for row in rows if row["eroded_before_impact"]]
        output[str(threshold)] = {
            "scene_count": len(eroded),
            "scene_rate": _safe_rate(len(eroded), len(scenes)),
            "median_first_unsafe_lead_frames": _median(
                [row["first_unsafe_after_safe_lead_frames"] for row in eroded]
            ),
            "median_first_unsafe_lead_seconds": _median(
                [row["first_unsafe_after_safe_lead_frames"] / 10.0 for row in eroded]
            ),
        }
    return output


def _scene_erosion(scene: dict[str, Any], *, threshold: float) -> dict[str, Any]:
    frames = [frame for frame in scene.get("frames", []) if frame.get("status") == "ok"]
    safe_leads = [
        int(frame["lead_frames"])
        for frame in frames
        if (_direct_selected_clearance(frame) is not None) and (_direct_selected_clearance(frame) >= threshold)
    ]
    if not safe_leads:
        return {"eroded_before_impact": False, "first_unsafe_after_safe_lead_frames": None}
    first_safe_lead = max(safe_leads)
    unsafe_after_safe = [
        int(frame["lead_frames"])
        for frame in frames
        if int(frame.get("lead_frames", -1)) < first_safe_lead
        and (_direct_selected_clearance(frame) is not None)
        and (_direct_selected_clearance(frame) < threshold)
    ]
    return {
        "eroded_before_impact": bool(unsafe_after_safe),
        "first_safe_lead_frames": first_safe_lead,
        "first_unsafe_after_safe_lead_frames": max(unsafe_after_safe) if unsafe_after_safe else None,
    }


def _subset_characterization(scene_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_mode: dict[str, list[dict[str, Any]]] = {}
    for row in scene_rows:
        by_mode.setdefault(str(row["assigned_mode"]), []).append(row)
    output = {}
    for mode, rows in sorted(by_mode.items()):
        output[mode] = {
            "scene_count": len(rows),
            "median_speed_mps": _distribution([row.get("median_speed_mps") for row in rows]).get("median"),
            "median_hazard_count": _distribution([row.get("median_hazard_count") for row in rows]).get("median"),
            "median_realized_current_clearance_m": _distribution(
                [row.get("median_realized_current_clearance_m") for row in rows]
            ).get("median"),
            "median_direct_selected_proxy_clearance_m": _distribution(
                [row.get("median_direct_selected_proxy_clearance_m") for row in rows]
            ).get("median"),
            "scenes": [row["scene"] for row in rows],
        }
    return output


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# AlpaSim ladder validation",
        "",
        report["method_note"],
        "",
        "## Clearance Divergence",
        "",
        "| Realized near threshold (m) | Scenes | Rate | Median onset (s before impact) |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for threshold, row in report["summary"]["clearance_divergence"].items():
        lines.append(
            f"| {threshold} | {row['scene_count']} | {row['scene_rate']:.3f} | "
            f"{_fmt(row['median_onset_lead_seconds'])} |"
        )
    lines.extend(
        [
            "",
            "## Proxy Clearance Erosion",
            "",
            "| Proxy threshold (m) | Scenes | Rate | Median first unsafe (s before impact) |",
            "| ---: | ---: | ---: | ---: |",
        ]
    )
    for threshold, row in report["summary"]["proxy_clearance_erosion"].items():
        lines.append(
            f"| {threshold} | {row['scene_count']} | {row['scene_rate']:.3f} | "
            f"{_fmt(row['median_first_unsafe_lead_seconds'])} |"
        )
    lines.extend(["", "## Failure Subsets", ""])
    for mode, row in report["summary"]["subset_characterization"].items():
        lines.append(
            f"- `{mode}`: {row['scene_count']} scenes; median hazards={_fmt(row['median_hazard_count'])}; "
            f"median realized current clearance={_fmt(row['median_realized_current_clearance_m'])} m; "
            f"median selected proxy clearance={_fmt(row['median_direct_selected_proxy_clearance_m'])} m."
        )
    return "\n".join(lines)


def _direct_selected_safe(frame: dict[str, Any]) -> bool:
    return bool((frame.get("direct_grid") or {}).get("selected_collision_free_proxy"))


def _direct_selected_clearance(frame: dict[str, Any]) -> float | None:
    return _as_float((frame.get("direct_grid") or {}).get("selected_min_clearance_m"))


def _realized_current_clearance(frame: dict[str, Any]) -> float | None:
    closest = frame.get("closest_hazard") or {}
    distance = _as_float(closest.get("distance_m"))
    radius = _as_float(closest.get("radius"))
    if distance is None or radius is None:
        return None
    return distance - radius


def _distribution(values: list[Any]) -> dict[str, Any]:
    parsed = [value for value in (_as_float(value) for value in values) if value is not None]
    if not parsed:
        return {"count": 0}
    return {
        "count": len(parsed),
        "mean": statistics.fmean(parsed),
        "median": statistics.median(parsed),
        "min": min(parsed),
        "max": max(parsed),
    }


def _median(values: list[Any]) -> float | None:
    parsed = [value for value in (_as_float(value) for value in values) if value is not None]
    return None if not parsed else statistics.median(parsed)


def _as_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _safe_rate(count: int, total: int) -> float:
    return 0.0 if total <= 0 else count / total


def _fmt(value: Any) -> str:
    parsed = _as_float(value)
    return "NA" if parsed is None else f"{parsed:.3f}"


def _parse_float_list(raw: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in raw.split(",") if item.strip())


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
