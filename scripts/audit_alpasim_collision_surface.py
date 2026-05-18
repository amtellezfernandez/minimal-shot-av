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


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = ROOT / "runs" / "alpasim_axis_constrained_30scene_eval"
DEFAULT_CANDIDATE = (
    ROOT
    / "runs"
    / "alpasim_actor_blindness_30scene_raw"
    / "token_dagger_iter2_axis_constrained_oracle_actor_clamped__front_camera_30scene_merged"
)
DEFAULT_JSON = ROOT / "artifacts" / "alpasim_collision_surface_30scene_audit.json"
DEFAULT_MD = ROOT / "artifacts" / "alpasim_collision_surface_30scene_audit.md"

COLLISION_COMPONENTS = ("collision_front", "collision_lateral", "collision_rear")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether raw AlpaSim collisions move when actor-complete oracle signals "
            "change the selector state, using existing metrics, selection logs, and controller traces."
        )
    )
    parser.add_argument("--baseline-run-dir", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--candidate-run-dir", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--baseline-label", default="axis_constrained_clamped")
    parser.add_argument("--candidate-label", default="world_frame_oracle_actor_completion")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_MD)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = audit_collision_surface(
        args.baseline_run_dir,
        args.candidate_run_dir,
        baseline_label=args.baseline_label,
        candidate_label=args.candidate_label,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = _markdown(report)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text(markdown + "\n", encoding="utf-8")
    print(markdown)
    return 0


def audit_collision_surface(
    baseline_run_dir: Path,
    candidate_run_dir: Path,
    *,
    baseline_label: str,
    candidate_label: str,
) -> dict[str, Any]:
    baseline_run_dir = baseline_run_dir.resolve()
    candidate_run_dir = candidate_run_dir.resolve()
    baseline = _load_batch(baseline_run_dir)
    candidate = _load_batch(candidate_run_dir)
    common = sorted(set(baseline) & set(candidate))
    if not common:
        raise SystemExit("No common scene directories found.")

    rows: list[dict[str, Any]] = []
    for scene in common:
        brow = baseline[scene]
        crow = candidate[scene]
        bfirst = brow["first_collision"]
        cfirst = crow["first_collision"]
        bsel = _nearest_log_row(brow["selection_log_rows"], bfirst["frame"] if bfirst else None)
        csel = _nearest_log_row(crow["selection_log_rows"], cfirst["frame"] if cfirst else None)
        bctrl = _nearest_controller_row(brow["controller_rows"], bfirst["timestamp_us"] if bfirst else None)
        cctrl = _nearest_controller_row(crow["controller_rows"], cfirst["timestamp_us"] if cfirst else None)
        rows.append(
            {
                "scene": scene,
                "baseline_collision": bfirst is not None,
                "candidate_collision": cfirst is not None,
                "baseline_first_collision": bfirst,
                "candidate_first_collision": cfirst,
                "baseline_selection_at_first_collision": _selection_summary(bsel),
                "candidate_selection_at_first_collision": _selection_summary(csel),
                "baseline_controller_at_first_collision": bctrl,
                "candidate_controller_at_first_collision": cctrl,
                "controller_delta_at_first_collision": _controller_delta(bctrl, cctrl),
                "candidate_lead_frames": _candidate_lead_frames(crow, cfirst),
                "baseline_selection_log": brow["selection_log_summary"],
                "candidate_selection_log": crow["selection_log_summary"],
            }
        )

    baseline_collision_rows = [r for r in rows if r["baseline_collision"]]
    candidate_collision_rows = [r for r in rows if r["candidate_collision"]]
    logged_candidate_collision_rows = [
        r
        for r in baseline_collision_rows
        if r["candidate_selection_at_first_collision"]["frame_index"] is not None
    ]

    paired = {
        "same_collision_binary": sum(
            r["baseline_collision"] == r["candidate_collision"] for r in rows
        ),
        "baseline_only_collision": sum(r["baseline_collision"] and not r["candidate_collision"] for r in rows),
        "candidate_only_collision": sum(r["candidate_collision"] and not r["baseline_collision"] for r in rows),
        "common_scene_count": len(rows),
    }
    first_component_same = sum(
        (r["baseline_first_collision"] or {}).get("components")
        == (r["candidate_first_collision"] or {}).get("components")
        for r in baseline_collision_rows
        if r["candidate_collision"]
    )
    timing_deltas = [
        (r["candidate_first_collision"]["timestamp_us"] - r["baseline_first_collision"]["timestamp_us"])
        / 1_000_000.0
        for r in baseline_collision_rows
        if r["candidate_collision"]
    ]
    lead_nonmaint = _nonnull(
        r["candidate_lead_frames"]["nonmaintain_selection"] for r in logged_candidate_collision_rows
    )
    lead_rear = _nonnull(
        r["candidate_lead_frames"]["rear_lane_hazard"] for r in logged_candidate_collision_rows
    )
    lead_close = _nonnull(
        r["candidate_lead_frames"]["top_clearance_lt_1m"] for r in logged_candidate_collision_rows
    )

    summary = {
        "baseline_label": baseline_label,
        "candidate_label": candidate_label,
        "baseline_run_dir": str(baseline_run_dir),
        "candidate_run_dir": str(candidate_run_dir),
        "common_scene_count": len(rows),
        "baseline_raw_collision_count": len(baseline_collision_rows),
        "candidate_raw_collision_count": len(candidate_collision_rows),
        "paired_collision": paired,
        "baseline_collision_logs": {
            "available": sum(r["baseline_selection_log"]["exists"] for r in baseline_collision_rows),
            "total": len(baseline_collision_rows),
            "good_json_lines": sum(r["baseline_selection_log"]["good_json_lines"] for r in baseline_collision_rows),
            "bad_json_lines": sum(r["baseline_selection_log"]["bad_json_lines"] for r in baseline_collision_rows),
        },
        "baseline_first_collision": {
            "component_counts": _component_counts(r["baseline_first_collision"] for r in baseline_collision_rows),
            "top_candidate_maintain": sum(
                r["baseline_selection_at_first_collision"]["top_candidate"] == "maintain"
                for r in baseline_collision_rows
            ),
            "structured_hazards_zero": sum(
                r["baseline_selection_at_first_collision"]["structured_hazard_count"] == 0
                for r in baseline_collision_rows
            ),
        },
        "candidate_first_collision": {
            "component_counts": _component_counts(r["candidate_first_collision"] for r in candidate_collision_rows),
            "same_first_component_as_baseline": first_component_same,
            "timing_delta_seconds": _distribution(timing_deltas),
            "logged_at_impact": len(logged_candidate_collision_rows),
            "proxy_hits": sum(
                r["candidate_selection_at_first_collision"]["oracle_actor_proxy_hit"] is True
                for r in logged_candidate_collision_rows
            ),
            "structured_hazards_positive": sum(
                r["candidate_selection_at_first_collision"]["structured_hazard_count"] > 0
                for r in logged_candidate_collision_rows
            ),
            "top_candidate_not_maintain": sum(
                r["candidate_selection_at_first_collision"]["top_candidate"] != "maintain"
                for r in logged_candidate_collision_rows
            ),
            "selected_token_not_maintain": sum(
                r["candidate_selection_at_first_collision"]["selected_token"] != "maintain"
                for r in logged_candidate_collision_rows
            ),
            "rear_lane_hazard_positive_for_logged_rear_collisions": _rear_hazard_count(
                logged_candidate_collision_rows
            ),
            "top_clearance": {
                "negative": sum(
                    _is_number(r["candidate_selection_at_first_collision"]["top_horizon_clearance_m"])
                    and r["candidate_selection_at_first_collision"]["top_horizon_clearance_m"] < 0
                    for r in logged_candidate_collision_rows
                ),
                "lt_1m": sum(
                    _is_number(r["candidate_selection_at_first_collision"]["top_horizon_clearance_m"])
                    and r["candidate_selection_at_first_collision"]["top_horizon_clearance_m"] < 1.0
                    for r in logged_candidate_collision_rows
                ),
            },
            "lead_frames": {
                "nonmaintain_selection": _distribution(lead_nonmaint),
                "rear_lane_hazard": _distribution(lead_rear),
                "top_clearance_lt_1m": _distribution(lead_close),
            },
        },
        "controller_at_first_collision": _controller_summary(baseline_collision_rows),
    }
    return {"schema": "alpasim_collision_surface_audit_v1", "summary": summary, "scenes": rows}


def _load_batch(run_dir: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    for scene_dir in sorted(run_dir.glob("[0-9][0-9][0-9]_*")):
        if not scene_dir.is_dir():
            continue
        rows[scene_dir.name] = {
            "first_collision": _first_collision(scene_dir),
            "selection_log_rows": _load_selection_log(scene_dir / "driver" / "selection-log.jsonl"),
            "selection_log_summary": _selection_log_summary(scene_dir / "driver" / "selection-log.jsonl"),
            "controller_rows": _load_controller_rows(scene_dir / "controller"),
        }
    return rows


def _first_collision(scene_dir: Path) -> dict[str, Any] | None:
    metrics_path = scene_dir / "aggregate" / "metrics_unprocessed.parquet"
    table = pq.read_table(metrics_path, columns=["name", "timestamps_us", "values", "valid"])
    payload = table.to_pydict()
    by_timestamp: dict[int, dict[str, float]] = {}
    collision_frame = 0
    collision_frame_by_timestamp: dict[int, int] = {}
    for name, timestamp, value, valid in zip(
        payload["name"], payload["timestamps_us"], payload["values"], payload["valid"]
    ):
        if not valid or value is None:
            continue
        timestamp_us = int(timestamp)
        by_timestamp.setdefault(timestamp_us, {})[name] = float(value)
        if name == "collision_any":
            collision_frame += 1
            collision_frame_by_timestamp[timestamp_us] = collision_frame
    collision_timestamps = sorted(
        timestamp
        for timestamp, values in by_timestamp.items()
        if values.get("collision_any", 0.0) > 0.0
    )
    if not collision_timestamps:
        return None
    timestamp_us = collision_timestamps[0]
    values = by_timestamp[timestamp_us]
    components = [
        component
        for component in COLLISION_COMPONENTS
        if values.get(component, 0.0) > 0.0
    ]
    return {
        "timestamp_us": timestamp_us,
        "frame": collision_frame_by_timestamp.get(timestamp_us),
        "components": components,
    }


def _load_selection_log(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _selection_log_summary(path: Path) -> dict[str, Any]:
    good = 0
    bad = 0
    exists = path.is_file()
    if exists:
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    json.loads(line)
                    good += 1
                except json.JSONDecodeError:
                    bad += 1
    return {"exists": exists, "good_json_lines": good, "bad_json_lines": bad}


def _load_controller_rows(controller_dir: Path) -> list[dict[str, float]]:
    csv_paths = sorted(controller_dir.glob("*.csv"))
    if not csv_paths:
        return []
    rows = []
    with csv_paths[0].open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            parsed = {}
            for key, value in row.items():
                try:
                    parsed[key] = float(value)
                except (TypeError, ValueError):
                    continue
            rows.append(parsed)
    return rows


def _nearest_log_row(rows: list[dict[str, Any]], frame: int | None) -> dict[str, Any] | None:
    if frame is None or not rows:
        return None
    return min(rows, key=lambda row: abs(int(row.get("frame_index", -1)) - frame))


def _nearest_controller_row(rows: list[dict[str, float]], timestamp_us: int | None) -> dict[str, float] | None:
    if timestamp_us is None or not rows:
        return None
    row = min(rows, key=lambda item: abs(int(item.get("timestamp_us", -1)) - timestamp_us))
    keys = (
        "timestamp_us",
        "x",
        "y",
        "vx",
        "vy",
        "u_longitudinal_actuation",
        "acceleration",
        "front_steering_angle",
    )
    return {key: row[key] for key in keys if key in row}


def _selection_summary(row: dict[str, Any] | None) -> dict[str, Any]:
    if row is None:
        return {
            "frame_index": None,
            "selected_token": None,
            "dagger_argmax_token": None,
            "spotlight_token": None,
            "top_candidate": None,
            "top_horizon_clearance_m": None,
            "top_action_clearance_m": None,
            "structured_hazard_count": None,
            "rear_lane_hazard_count": None,
            "oracle_actor_proxy_hit": None,
            "oracle_actor_proxy_count": None,
        }
    top_candidates = row.get("spotlight_top_candidates") or []
    top = top_candidates[0] if top_candidates else {}
    signal = row.get("alpasim_signal") or {}
    hazards = signal.get("structured_hazards") or []
    return {
        "frame_index": row.get("frame_index"),
        "selected_token": row.get("hybrid_token"),
        "dagger_argmax_token": row.get("dagger_argmax_token"),
        "spotlight_token": row.get("spotlight_token"),
        "top_candidate": top.get("candidate"),
        "top_horizon_clearance_m": top.get("horizon_clearance_m"),
        "top_action_clearance_m": top.get("action_clearance_m"),
        "structured_hazard_count": len(hazards),
        "rear_lane_hazard_count": _rear_lane_hazard_count(hazards),
        "oracle_actor_proxy_hit": signal.get("oracle_actor_proxy_hit"),
        "oracle_actor_proxy_count": signal.get("oracle_actor_proxy_count"),
    }


def _controller_delta(
    baseline: dict[str, float] | None,
    candidate: dict[str, float] | None,
) -> dict[str, float] | None:
    if baseline is None or candidate is None:
        return None
    out = {}
    for key in ("x", "y", "vx", "vy", "u_longitudinal_actuation", "acceleration", "front_steering_angle"):
        if key in baseline and key in candidate:
            out[key] = candidate[key] - baseline[key]
    out["speed"] = _speed(candidate) - _speed(baseline)
    return out


def _candidate_lead_frames(candidate: dict[str, Any], first_collision: dict[str, Any] | None) -> dict[str, int | None]:
    if first_collision is None:
        return {"nonmaintain_selection": None, "rear_lane_hazard": None, "top_clearance_lt_1m": None}
    collision_frame = first_collision["frame"]
    rows = candidate["selection_log_rows"]
    first_nonmaintain = next(
        (row.get("frame_index") for row in rows if row.get("hybrid_token") != "maintain"),
        None,
    )
    first_rear_hazard = next(
        (
            row.get("frame_index")
            for row in rows
            if _rear_lane_hazard_count((row.get("alpasim_signal") or {}).get("structured_hazards") or []) > 0
        ),
        None,
    )
    first_close_top = next(
        (
            row.get("frame_index")
            for row in rows
            if _top_horizon_clearance(row) is not None and _top_horizon_clearance(row) < 1.0
        ),
        None,
    )
    return {
        "nonmaintain_selection": _lead(collision_frame, first_nonmaintain),
        "rear_lane_hazard": _lead(collision_frame, first_rear_hazard),
        "top_clearance_lt_1m": _lead(collision_frame, first_close_top),
    }


def _top_horizon_clearance(row: dict[str, Any]) -> float | None:
    candidates = row.get("spotlight_top_candidates") or []
    if not candidates:
        return None
    value = candidates[0].get("horizon_clearance_m")
    return float(value) if isinstance(value, int | float) else None


def _lead(collision_frame: int | None, event_frame: int | None) -> int | None:
    if collision_frame is None or event_frame is None:
        return None
    return int(collision_frame) - int(event_frame)


def _rear_lane_hazard_count(hazards: list[dict[str, Any]]) -> int:
    return sum(
        1
        for hazard in hazards
        if _is_number(hazard.get("x"))
        and _is_number(hazard.get("y"))
        and hazard["x"] < 0
        and abs(hazard["y"]) < 3.5
    )


def _component_counts(rows: Any) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        if not row:
            continue
        components = row.get("components") or []
        if not components:
            counter["none"] += 1
        else:
            for component in components:
                counter[component] += 1
    return dict(sorted(counter.items()))


def _rear_hazard_count(rows: list[dict[str, Any]]) -> dict[str, int]:
    rear_rows = [
        row
        for row in rows
        if "collision_rear" in ((row["candidate_first_collision"] or {}).get("components") or [])
    ]
    return {
        "positive": sum(
            row["candidate_selection_at_first_collision"]["rear_lane_hazard_count"] > 0
            for row in rear_rows
        ),
        "total": len(rear_rows),
    }


def _controller_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    deltas = [row["controller_delta_at_first_collision"] for row in rows if row["controller_delta_at_first_collision"]]
    return {
        "speed_delta_mps": _distribution([row["speed"] for row in deltas]),
        "lateral_delta_m": _distribution([row["y"] for row in deltas]),
        "candidate_slower_count": sum(row["speed"] < 0 for row in deltas),
        "paired_controller_rows": len(deltas),
    }


def _distribution(values: list[float | int]) -> dict[str, float | int | None]:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not clean:
        return {"n": 0, "median": None, "mean": None, "min": None, "max": None}
    return {
        "n": len(clean),
        "median": statistics.median(clean),
        "mean": statistics.mean(clean),
        "min": min(clean),
        "max": max(clean),
    }


def _nonnull(values: Any) -> list[int]:
    return [int(value) for value in values if value is not None]


def _speed(row: dict[str, float] | None) -> float:
    if row is None:
        return float("nan")
    return math.hypot(float(row.get("vx", 0.0)), float(row.get("vy", 0.0)))


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and math.isfinite(float(value))


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        if math.isinf(value):
            return "inf"
        return f"{value:.{digits}f}"
    return str(value)


def _markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    baseline = summary["baseline_first_collision"]
    candidate = summary["candidate_first_collision"]
    timing = candidate["timing_delta_seconds"]
    speed = summary["controller_at_first_collision"]["speed_delta_mps"]
    lines = [
        "# AlpaSim 30-scene collision-surface audit",
        "",
        f"Baseline: `{summary['baseline_label']}`",
        f"Candidate: `{summary['candidate_label']}`",
        "",
        "## Summary",
        "",
        f"- Common scenes: `{summary['common_scene_count']}`.",
        f"- Raw collision scenes: baseline `{summary['baseline_raw_collision_count']}`, candidate `{summary['candidate_raw_collision_count']}`.",
        (
            "- Paired collision change: "
            f"`baseline_only={summary['paired_collision']['baseline_only_collision']}`, "
            f"`candidate_only={summary['paired_collision']['candidate_only_collision']}`, "
            f"`same={summary['paired_collision']['same_collision_binary']}/{summary['paired_collision']['common_scene_count']}`."
        ),
        (
            "- Baseline first-impact selection: "
            f"`top_candidate_maintain={baseline['top_candidate_maintain']}/{summary['baseline_raw_collision_count']}`, "
            f"`structured_hazards_zero={baseline['structured_hazards_zero']}/{summary['baseline_raw_collision_count']}`."
        ),
        (
            "- Baseline first-impact components: "
            + ", ".join(f"`{key}={value}`" for key, value in baseline["component_counts"].items())
            + "."
        ),
        (
            "- Candidate first-impact selection where logs exist: "
            f"`logged={candidate['logged_at_impact']}/{summary['baseline_raw_collision_count']}`, "
            f"`proxy_hits={candidate['proxy_hits']}/{candidate['logged_at_impact']}`, "
            f"`top_not_maintain={candidate['top_candidate_not_maintain']}/{candidate['logged_at_impact']}`, "
            f"`selected_not_maintain={candidate['selected_token_not_maintain']}/{candidate['logged_at_impact']}`."
        ),
        (
            "- Candidate rear-lane hazards at logged rear impacts: "
            f"`{candidate['rear_lane_hazard_positive_for_logged_rear_collisions']['positive']}/"
            f"{candidate['rear_lane_hazard_positive_for_logged_rear_collisions']['total']}`."
        ),
        (
            "- Collision timing delta candidate-baseline: "
            f"median `{_fmt(timing['median'], 3)}s`, mean `{_fmt(timing['mean'], 3)}s`, "
            f"range `[{_fmt(timing['min'], 3)}, {_fmt(timing['max'], 3)}]s`."
        ),
        (
            "- Controller speed delta at first impact: "
            f"median `{_fmt(speed['median'], 3)} m/s`, mean `{_fmt(speed['mean'], 3)} m/s`; "
            f"candidate slower on `{summary['controller_at_first_collision']['candidate_slower_count']}/"
            f"{summary['controller_at_first_collision']['paired_controller_rows']}` paired impacts."
        ),
        "",
        "## Interpretation",
        "",
        (
            "The matched oracle actor run changes the selector state and selected action at impact, "
            "but it does not change the collision scene set. In the baseline, first-impact frames "
            "look like clear corridor frames to the scorer. In the oracle run, actors are visible, "
            "rear-lane hazards are present for logged rear impacts, and the selector usually moves "
            "to nudge/evasive/stop actions. The remaining collision surface is therefore not "
            "explained by actor visibility or action ranking alone. The unresolved split is between "
            "candidate-set coverage and controller/traffic execution; resolving it requires "
            "counterfactual candidate/controller replay."
        ),
        "",
        "## First-impact scene table",
        "",
        "| Scene | Baseline impact | Candidate impact | Baseline sel/top/haz | Candidate sel/top/haz/proxy | dt (s) |",
        "| --- | --- | --- | --- | --- | ---: |",
    ]
    for row in report["scenes"]:
        if not row["baseline_collision"] and not row["candidate_collision"]:
            continue
        bfirst = row["baseline_first_collision"] or {}
        cfirst = row["candidate_first_collision"] or {}
        bsel = row["baseline_selection_at_first_collision"]
        csel = row["candidate_selection_at_first_collision"]
        dt = None
        if bfirst and cfirst:
            dt = (cfirst["timestamp_us"] - bfirst["timestamp_us"]) / 1_000_000.0
        lines.append(
            "| "
            + " | ".join(
                [
                    row["scene"],
                    f"f{bfirst.get('frame', 'NA')} {','.join(bfirst.get('components') or []) or 'none'}",
                    f"f{cfirst.get('frame', 'NA')} {','.join(cfirst.get('components') or []) or 'none'}",
                    (
                        f"{bsel['selected_token']}/{bsel['top_candidate']}/"
                        f"{bsel['structured_hazard_count']}"
                    ),
                    (
                        f"{csel['selected_token']}/{csel['top_candidate']}/"
                        f"{csel['structured_hazard_count']}/"
                        f"{csel['oracle_actor_proxy_hit']}"
                    ),
                    _fmt(dt, 3),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
