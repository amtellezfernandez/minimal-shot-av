#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from typing import Mapping


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT_JSON = ROOT / "artifacts" / "corl2027" / "candidate_regret_lifecycle.json"
DEFAULT_OUTPUT_MD = ROOT / "artifacts" / "corl2027" / "candidate_regret_lifecycle.md"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize candidate recoverable-regret reports across bootstrap stages."
    )
    parser.add_argument(
        "--report",
        action="append",
        required=True,
        help="Stage report in NAME=PATH form. Can be repeated.",
    )
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args()
    args.report = list(args.report or [])
    return args


def main() -> int:
    args = _parse_args()
    lifecycle = summarize_lifecycle(load_named_reports(args.report))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(lifecycle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text(markdown_lifecycle(lifecycle) + "\n", encoding="utf-8")
    print(markdown_lifecycle(lifecycle))
    return 0


def load_named_reports(report_args: list[str]) -> list[tuple[str, dict[str, Any]]]:
    reports = []
    for raw in report_args:
        if "=" not in raw:
            raise ValueError(f"--report must be NAME=PATH, got {raw!r}")
        name, path = raw.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"--report has empty stage name: {raw!r}")
        reports.append((name, json.loads(Path(path).read_text(encoding="utf-8"))))
    return reports


def summarize_lifecycle(named_reports: list[tuple[str, Mapping[str, Any]]]) -> dict[str, Any]:
    stages = [stage_summary(name, report) for name, report in named_reports]
    deltas = []
    for previous, current in zip(stages, stages[1:]):
        deltas.append(
            {
                "from": previous["stage"],
                "to": current["stage"],
                "generation_gap_delta": current["generation_gap_count"] - previous["generation_gap_count"],
                "safety_regret_delta": current["recoverable_safety_regret_count"]
                - previous["recoverable_safety_regret_count"],
                "utility_regret_delta": current["recoverable_utility_regret_count"]
                - previous["recoverable_utility_regret_count"],
                "mean_regret_delta": round(
                    current["mean_recoverable_regret"] - previous["mean_recoverable_regret"],
                    6,
                ),
                "top1_fail_rate_delta": round(current["top1_replay_fail_rate"] - previous["top1_replay_fail_rate"], 6),
            }
        )
    return {
        "schema": "candidate_regret_lifecycle_v1",
        "stage_count": len(stages),
        "stages": stages,
        "deltas": deltas,
        "interpretation": lifecycle_interpretation(stages),
    }


def stage_summary(name: str, report: Mapping[str, Any]) -> dict[str, Any]:
    gap = report["gap_decomposition"]
    score = report["score_summary"]
    return {
        "stage": name,
        "scene_count": int(report["valid_scene_count"]),
        "generation_gap_count": int(gap["generation_gap_count"]),
        "generation_gap_rate": float(gap["generation_gap_rate"]),
        "recoverable_safety_regret_count": int(gap["recoverable_safety_regret_count"]),
        "recoverable_safety_regret_rate": float(gap["recoverable_safety_regret_rate"]),
        "recoverable_utility_regret_count": int(gap["recoverable_utility_regret_count"]),
        "recoverable_utility_regret_rate": float(gap["recoverable_utility_regret_rate"]),
        "mean_recoverable_regret": float(score["mean_recoverable_regret"]),
        "top1_replay_fail_rate": float(score["top1_replay_fail_rate"]),
        "oracle_at_k_replay_fail_rate": float(score["oracle_at_k_replay_fail_rate"]),
    }


def lifecycle_interpretation(stages: list[Mapping[str, Any]]) -> str:
    if len(stages) < 2:
        return "single-stage report; no lifecycle interpretation"
    first = stages[0]
    last = stages[-1]
    safety_drop = int(first["recoverable_safety_regret_count"]) - int(last["recoverable_safety_regret_count"])
    utility_gain = int(last["recoverable_utility_regret_count"]) - int(first["recoverable_utility_regret_count"])
    generation_delta = int(last["generation_gap_count"]) - int(first["generation_gap_count"])
    if safety_drop > 0 and utility_gain > 0 and generation_delta == 0:
        return (
            "bootstrap closes much of the recoverable safety regret, leaves generation gap unchanged, "
            "and exposes utility regret as the next optimization branch"
        )
    return "bootstrap lifecycle does not match the safety-to-utility residual shift pattern"


def markdown_lifecycle(report: Mapping[str, Any]) -> str:
    lines = [
        "# Candidate Regret Lifecycle",
        "",
        f"- Stages: `{report['stage_count']}`",
        f"- Interpretation: {report['interpretation']}",
        "",
        "## Stage Table",
        "",
        "| Stage | Gen gap | Safety regret | Utility regret | Mean regret | Top-1 fail | Oracle fail |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for stage in report["stages"]:
        lines.append(
            f"| {stage['stage']} | {stage['generation_gap_count']} | "
            f"{stage['recoverable_safety_regret_count']} | "
            f"{stage['recoverable_utility_regret_count']} | "
            f"{stage['mean_recoverable_regret']:.3f} | "
            f"{stage['top1_replay_fail_rate']:.3f} | "
            f"{stage['oracle_at_k_replay_fail_rate']:.3f} |"
        )
    lines.extend(["", "## Deltas", "", "| From | To | Gen gap Δ | Safety Δ | Utility Δ | Mean regret Δ |"])
    lines.append("|---|---|---:|---:|---:|---:|")
    for delta in report["deltas"]:
        lines.append(
            f"| {delta['from']} | {delta['to']} | {delta['generation_gap_delta']} | "
            f"{delta['safety_regret_delta']} | {delta['utility_regret_delta']} | "
            f"{delta['mean_regret_delta']:.3f} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
