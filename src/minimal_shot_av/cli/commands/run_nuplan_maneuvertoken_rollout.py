#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
    for record in records:
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
    return {
        "schema": "nuplan_maneuvertoken_rollout_v1",
        "selection_mode": "learned_selector" if selector is not None else "heuristic_selector",
        "scene_count": len(records),
        "proxy_safe_rate": round(safe_count / len(records), 6) if records else 0.0,
        "selected_token_histogram": dict(sorted(selected_token_histogram.items())),
        "scenes": records,
        "limitations": [
            "This adapter consumes nuPlan-like scene summaries, not full nuPlan closed-loop logs.",
            "Realized controller-execution diagnostics still require a reactive nuPlan rollout surface.",
        ],
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# nuPlan ManeuverToken Adapter Rollout",
        "",
        f"- Scene count: `{report['scene_count']}`",
        f"- Proxy-safe rate: `{report['proxy_safe_rate']:.3f}`",
        f"- Selected token histogram: `{json.dumps(report['selected_token_histogram'], sort_keys=True)}`",
        "",
        "## Selected scenes",
        "",
    ]
    for scene in report["scenes"][:10]:
        lines.append(
            f"- `{scene['scene_id']}`: selected `{scene['selected_token']}` "
            f"(proxy_safe={scene['selected_token_proxy_safe']}, "
            f"min_clearance={scene['selected_token_min_proxy_clearance_m']:.3f} m)"
        )
    return "\n".join(lines)
