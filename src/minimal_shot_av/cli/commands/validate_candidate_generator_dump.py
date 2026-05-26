#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any
from typing import Mapping

from minimal_shot_av.cli.commands.analyze_candidate_recoverable_regret import normalize_candidate_sets
from minimal_shot_av.cli.commands.analyze_candidate_recoverable_regret import read_json_or_jsonl


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT_JSON = ROOT / "artifacts" / "corl2027" / "candidate_generator_dump_validation.json"
DEFAULT_OUTPUT_MD = ROOT / "artifacts" / "corl2027" / "candidate_generator_dump_validation.md"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate candidate-generator dumps before recoverable-regret analysis."
    )
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--near-miss-threshold-m", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = read_json_or_jsonl(args.input_json)
    candidate_sets = normalize_candidate_sets(payload, near_miss_threshold_m=float(args.near_miss_threshold_m))
    report = validate_candidate_generator_dump(candidate_sets, input_json=str(args.input_json))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text(markdown_validation(report) + "\n", encoding="utf-8")
    print(markdown_validation(report))
    return 0


def validate_candidate_generator_dump(candidate_sets: list[Mapping[str, Any]], *, input_json: str) -> dict[str, Any]:
    scene_count = len(candidate_sets)
    candidate_counts = []
    top1_missing = 0
    trajectory_missing = 0
    replay_metric_missing = 0
    utility_metric_missing = 0
    valid_analysis_scenes = 0
    valid_generation_scenes = 0
    generator_histogram: Counter[str] = Counter()
    issue_rows = []
    for candidate_set in candidate_sets:
        generator_histogram.update([str(candidate_set.get("generator", ""))])
        candidates = list(candidate_set.get("candidates", []))
        candidate_counts.append(len(candidates))
        top1_id = str(candidate_set.get("top1_candidate_id", ""))
        candidate_ids = {str(candidate.get("candidate_id", "")) for candidate in candidates}
        scene_issues = []
        top1_valid = True
        if not top1_id or top1_id not in candidate_ids:
            top1_missing += 1
            top1_valid = False
            scene_issues.append("top1_candidate_missing")
        missing_traj = sum(1 for candidate in candidates if not candidate.get("trajectory"))
        trajectory_missing += missing_traj
        if missing_traj:
            scene_issues.append("candidate_trajectory_missing")
        missing_replay = sum(
            1
            for candidate in candidates
            if candidate.get("metrics", {}).get("replay_fail") is None
            or candidate.get("metrics", {}).get("clearance_m") is None
        )
        replay_metric_missing += missing_replay
        if missing_replay:
            scene_issues.append("candidate_replay_metric_missing")
        missing_utility = sum(
            1
            for candidate in candidates
            if candidate.get("metrics", {}).get("progress_m") is None
        )
        utility_metric_missing += missing_utility
        if missing_utility:
            scene_issues.append("candidate_utility_metric_missing")
        if candidates and missing_traj == 0 and top1_valid:
            valid_generation_scenes += 1
        if candidates and missing_replay == 0 and missing_utility == 0 and missing_traj == 0 and top1_valid:
            valid_analysis_scenes += 1
        if scene_issues and len(issue_rows) < 20:
            issue_rows.append(
                {
                    "scene_id": str(candidate_set.get("scene_id", "")),
                    "generator": str(candidate_set.get("generator", "")),
                    "issues": sorted(set(scene_issues)),
                }
            )
    total_candidates = sum(candidate_counts)
    readiness = (
        "analysis_ready"
        if valid_analysis_scenes == scene_count and scene_count > 0
        else "generation_ready"
        if valid_generation_scenes == scene_count and scene_count > 0
        else "not_ready"
    )
    return {
        "schema": "candidate_generator_dump_validation_v1",
        "input_json": input_json,
        "readiness": readiness,
        "scene_count": scene_count,
        "candidate_count": total_candidates,
        "mean_candidate_count": mean([float(value) for value in candidate_counts]),
        "valid_generation_scene_count": valid_generation_scenes,
        "valid_analysis_scene_count": valid_analysis_scenes,
        "top1_missing_scene_count": top1_missing,
        "trajectory_missing_candidate_count": trajectory_missing,
        "replay_metric_missing_candidate_count": replay_metric_missing,
        "utility_metric_missing_candidate_count": utility_metric_missing,
        "generator_histogram": dict(sorted(generator_histogram.items())),
        "issue_examples": issue_rows,
        "required_for_recoverable_regret": [
            "scene_id",
            "top1_candidate_id",
            "candidates[].candidate_id",
            "candidates[].trajectory",
            "candidates[].metrics.replay_fail or clearance_m",
            "candidates[].metrics.progress_m",
        ],
    }


def markdown_validation(report: Mapping[str, Any]) -> str:
    lines = [
        "# Candidate Generator Dump Validation",
        "",
        f"- Input: `{report['input_json']}`",
        f"- Readiness: `{report['readiness']}`",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| scenes | {report['scene_count']} |",
        f"| candidates | {report['candidate_count']} |",
        f"| mean candidates | {report['mean_candidate_count']:.3f} |",
        f"| generation-ready scenes | {report['valid_generation_scene_count']} |",
        f"| analysis-ready scenes | {report['valid_analysis_scene_count']} |",
        f"| missing replay metrics | {report['replay_metric_missing_candidate_count']} |",
        f"| missing utility metrics | {report['utility_metric_missing_candidate_count']} |",
        "",
    ]
    if report["issue_examples"]:
        lines.extend(["## Issue Examples", "", "| Scene | Generator | Issues |", "|---|---|---|"])
        for row in report["issue_examples"]:
            lines.append(f"| {row['scene_id']} | {row['generator']} | `{','.join(row['issues'])}` |")
        lines.append("")
    return "\n".join(lines)


def mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
