#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from minimal_shot_av.model.nuplan_devkit_integration import NuPlanSceneFilter
from minimal_shot_av.model.nuplan_devkit_integration import load_nuplan_scenes
from minimal_shot_av.model.nuplan_maneuver_token_adapter import build_scene_diagnostic_record
from minimal_shot_av.model.nuplan_maneuver_token_selector import load_selector
from minimal_shot_av.model.nuplan_maneuver_token_selector import selector_feature_row


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT_JSON = ROOT / "artifacts" / "corl2027" / "nuplan_maneuvertoken_rollout.json"
DEFAULT_OUTPUT_MARKDOWN = ROOT / "artifacts" / "corl2027" / "nuplan_maneuvertoken_rollout.md"
NEAR_MISS_THRESHOLD_M = 0.5
FAILURE_RUNGS = (
    "no actor/state visibility",
    "no safe token existed",
    "safe token existed but selector missed it",
    "proxy predicted safe but realized failed",
    "metric/spec ambiguity",
)
RESOLVED_SAFE = "resolved safe"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the ManeuverToken nuPlan adapter on real nuPlan DB scenarios or scene-summary JSON "
            "and export selected-token rollouts with per-frame proxy diagnostics."
        )
    )
    parser.add_argument("--input", type=Path, help="JSON list of pre-extracted nuPlan scene summaries.")
    parser.add_argument(
        "--nuplan-data-root",
        type=Path,
        help="nuPlan dataset root or split directory containing DB files.",
    )
    parser.add_argument(
        "--nuplan-db-file",
        action="append",
        default=[],
        help="Specific nuPlan DB file or DB directory. Pass multiple times if needed.",
    )
    parser.add_argument("--scenario-type", action="append", default=[], help="Optional nuPlan scenario type filter.")
    parser.add_argument("--scenario-token", action="append", default=[], help="Optional nuPlan token filter.")
    parser.add_argument("--log-name", action="append", default=[], help="Optional nuPlan log filename filter.")
    parser.add_argument("--map-name", action="append", default=[], help="Optional nuPlan map-name filter.")
    parser.add_argument("--limit", type=int, help="Limit the number of loaded nuPlan scenes.")
    parser.add_argument("--selector-model", type=Path, help="Optional trained selector artifact from Experiment 2.")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_OUTPUT_MARKDOWN)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    scenes = _load_scenes(args)
    selector = None if args.selector_model is None else load_selector(args.selector_model)
    report = run_rollout(scenes, selector=selector)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = markdown_report(report)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text(markdown + "\n", encoding="utf-8")
    print(markdown)


def _load_scenes(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.input is not None:
        return json.loads(args.input.read_text(encoding="utf-8"))
    if args.nuplan_data_root is None and not args.nuplan_db_file:
        raise ValueError("pass either --input or --nuplan-data-root/--nuplan-db-file")
    data_root = args.nuplan_data_root or Path(".")
    return load_nuplan_scenes(
        data_root=data_root,
        scene_filter=NuPlanSceneFilter(
            db_files=tuple(str(value) for value in args.nuplan_db_file),
            log_names=tuple(str(value) for value in args.log_name),
            map_names=tuple(str(value) for value in args.map_name),
            scenario_tokens=tuple(str(value) for value in args.scenario_token),
            scenario_types=tuple(str(value) for value in args.scenario_type),
            limit=args.limit,
        ),
    )


def run_rollout(scenes: list[dict[str, Any]], *, selector=None) -> dict[str, Any]:
    records = [build_scene_diagnostic_record(scene) for scene in scenes]
    selected_token_histogram: dict[str, int] = {}
    safe_count = 0
    rung_counts = {rung: 0 for rung in FAILURE_RUNGS}
    for scene, record in zip(scenes, records):
        if "expert_trajectory" in scene:
            record["expert_trajectory"] = list(scene.get("expert_trajectory", []))
        if selector is not None:
            selected = max(
                record["candidates"],
                key=lambda candidate: (
                    selector.predict_score(selector_feature_row(record, candidate)),
                    float(candidate["min_proxy_clearance_m"]),
                    float(candidate["final_progress_m"]),
                ),
            )
            record["selected_token"] = str(selected["token"])
            record["selected_token_proxy_safe"] = bool(selected["proxy_safe"])
            record["selected_token_min_proxy_clearance_m"] = round(float(selected["min_proxy_clearance_m"]), 6)
            record["selected_token_score"] = round(
                float(selector.predict_score(selector_feature_row(record, selected))),
                6,
            )
            record["selected_token_rollout"] = dict(selected)
            record["selection_source"] = "learned_selector"
        else:
            record["selection_source"] = "heuristic_selector"
        token = str(record["selected_token"])
        selected_token_histogram[token] = selected_token_histogram.get(token, 0) + 1
        if bool(record["selected_token_proxy_safe"]):
            safe_count += 1
        enrich_scene_failure_diagnostics(record)
        if str(record["failure_rung"]) in rung_counts:
            rung_counts[str(record["failure_rung"])] += 1
    rung_table = [
        {
            "failure_rung": rung,
            "count": rung_counts[rung],
            "rate": round(rung_counts[rung] / len(records), 6) if records else 0.0,
        }
        for rung in FAILURE_RUNGS
    ]
    proxy_safe_selected_count = sum(1 for record in records if bool(record["proxy_safe_selected"]))
    proxy_safe_realized_safe_count = sum(
        1
        for record in records
        if bool(record["proxy_safe_selected"]) and record["failure_rung"] == RESOLVED_SAFE
    )
    proxy_safe_realized_near_or_collision_count = sum(
        1
        for record in records
        if bool(record["proxy_safe_selected"]) and record["failure_rung"] == "proxy predicted safe but realized failed"
    )
    proxy_safe_realized_missing_count = sum(
        1
        for record in records
        if bool(record["proxy_safe_selected"]) and record["selected_token_realized_min_clearance_m"] is None
    )
    return {
        "schema": "nuplan_maneuvertoken_rollout_v2",
        "selection_mode": "learned_selector" if selector is not None else "heuristic_selector",
        "scene_count": len(records),
        "proxy_safe_rate": round(safe_count / len(records), 6) if records else 0.0,
        "proxy_safe_selected_count": proxy_safe_selected_count,
        "proxy_safe_realized_safe_count": proxy_safe_realized_safe_count,
        "proxy_safe_realized_near_or_collision_count": proxy_safe_realized_near_or_collision_count,
        "proxy_safe_realized_missing_count": proxy_safe_realized_missing_count,
        "selected_token_histogram": dict(sorted(selected_token_histogram.items())),
        "failure_rung_table": rung_table,
        "scenes": records,
        "limitations": [
            "This adapter consumes nuPlan-like scene summaries, not full nuPlan closed-loop logs.",
            "Realized controller-execution diagnostics still require a reactive nuPlan rollout surface.",
            "In this report, `realized_min_clearance` is null when no executed closed-loop rollout is available.",
        ],
    }


def enrich_scene_failure_diagnostics(record: dict[str, Any]) -> None:
    actor_summary = record["actor_summary"]
    safe_candidates = [candidate for candidate in record["candidates"] if bool(candidate["proxy_safe"])]
    safe_token_existed = bool(safe_candidates)
    oracle_safe_token = None
    if safe_candidates:
        oracle_safe_token = max(
            safe_candidates,
            key=lambda candidate: (
                float(candidate["score"]),
                float(candidate["min_proxy_clearance_m"]),
                float(candidate["final_progress_m"]),
            ),
        )["token"]
    realized_min_clearance = _realized_min_clearance(record)
    collision_or_near_miss = (
        None if realized_min_clearance is None else realized_min_clearance < NEAR_MISS_THRESHOLD_M
    )
    selector_chose_safe_token = bool(record["selected_token_proxy_safe"])
    failure_rung = classify_failure_rung(
        actor_visible_count=int(actor_summary["visible_actor_count"]),
        actor_count=int(actor_summary["actor_count"]),
        safe_token_existed=safe_token_existed,
        selector_chose_safe_token=selector_chose_safe_token,
        selected_token_realized_min_clearance_m=realized_min_clearance,
        near_miss_threshold_m=NEAR_MISS_THRESHOLD_M,
    )
    record["proxy_safe_selected"] = selector_chose_safe_token
    record["selected_token_realized_min_clearance_m"] = realized_min_clearance
    record["realized_min_clearance"] = realized_min_clearance
    record["collision_or_near_miss"] = collision_or_near_miss
    record["safe_token_existed"] = safe_token_existed
    record["oracle_safe_token"] = oracle_safe_token
    record["selector_chose_safe_token"] = selector_chose_safe_token
    record["failure_rung"] = failure_rung


def classify_failure_rung(
    *,
    actor_visible_count: int,
    actor_count: int,
    safe_token_existed: bool,
    selector_chose_safe_token: bool,
    selected_token_realized_min_clearance_m: float | None,
    near_miss_threshold_m: float,
) -> str:
    if actor_visible_count < actor_count:
        return "no actor/state visibility"
    if not safe_token_existed:
        return "no safe token existed"
    if not selector_chose_safe_token:
        return "safe token existed but selector missed it"
    if selected_token_realized_min_clearance_m is None:
        return "metric/spec ambiguity"
    if selected_token_realized_min_clearance_m < near_miss_threshold_m:
        return "proxy predicted safe but realized failed"
    return RESOLVED_SAFE


def _realized_min_clearance(record: dict[str, Any]) -> float | None:
    value = record.get("selected_token_realized_min_clearance_m")
    if value is None:
        return None
    value = float(value)
    return None if not math.isfinite(value) else round(value, 6)


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# nuPlan ManeuverToken Adapter Rollout",
        "",
        f"- Scene count: `{report['scene_count']}`",
        f"- Proxy-safe rate: `{report['proxy_safe_rate']:.3f}`",
        f"- Proxy-safe selected count: `{report['proxy_safe_selected_count']}`",
        f"- Proxy-safe + realized safe count: `{report['proxy_safe_realized_safe_count']}`",
        f"- Proxy-safe + realized near/collision count: `{report['proxy_safe_realized_near_or_collision_count']}`",
        f"- Proxy-safe + realized missing count: `{report['proxy_safe_realized_missing_count']}`",
        f"- Selected token histogram: `{json.dumps(report['selected_token_histogram'], sort_keys=True)}`",
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
        "## Selected scenes",
        "",
        ]
    )
    for scene in report["scenes"][:10]:
        lines.append(
            f"- `{scene['scene_id']}`: selected `{scene['selected_token']}` "
            f"(proxy_safe={scene['proxy_safe_selected']}, "
            f"safe_token_existed={scene['safe_token_existed']}, "
            f"failure_rung={scene['failure_rung']}, "
            f"proxy_min_clearance={scene['selected_token_min_proxy_clearance_m']:.3f} m, "
            f"realized_min_clearance={scene['realized_min_clearance']})"
        )
    return "\n".join(lines)
