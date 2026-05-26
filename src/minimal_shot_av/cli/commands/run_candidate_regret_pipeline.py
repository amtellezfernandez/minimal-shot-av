#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from typing import Mapping

from minimal_shot_av.cli.commands.analyze_candidate_recoverable_regret import analyze_candidate_recoverable_regret
from minimal_shot_av.cli.commands.analyze_candidate_recoverable_regret import normalize_candidate_sets
from minimal_shot_av.cli.commands.analyze_candidate_recoverable_regret import read_json_or_jsonl
from minimal_shot_av.cli.commands.attach_candidate_metrics import attach_candidate_metrics
from minimal_shot_av.cli.commands.build_candidate_regret_curriculum import build_candidate_regret_curriculum
from minimal_shot_av.cli.commands.build_candidate_regret_curriculum import markdown_curriculum
from minimal_shot_av.cli.commands.summarize_candidate_regret_lifecycle import markdown_lifecycle
from minimal_shot_av.cli.commands.summarize_candidate_regret_lifecycle import summarize_lifecycle
from minimal_shot_av.cli.commands.validate_candidate_generator_dump import markdown_validation
from minimal_shot_av.cli.commands.validate_candidate_generator_dump import validate_candidate_generator_dump


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "corl2027" / "candidate_regret_pipeline"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the candidate recoverable-regret pipeline for G0 and optional G1 generator dumps."
    )
    parser.add_argument("--g0-candidates", type=Path, required=True)
    parser.add_argument("--g0-metrics", type=Path)
    parser.add_argument("--g1-candidates", type=Path)
    parser.add_argument("--g1-metrics", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--near-miss-threshold-m", type=float, default=1.0)
    parser.add_argument("--failure-penalty", type=float, default=250.0)
    parser.add_argument("--progress-weight", type=float, default=1.0)
    parser.add_argument("--ade-weight", type=float, default=2.0)
    parser.add_argument("--fde-weight", type=float, default=0.0)
    parser.add_argument("--clearance-weight", type=float, default=0.0)
    parser.add_argument("--utility-regret-threshold", type=float, default=0.001)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    score_config = {
        "near_miss_threshold_m": float(args.near_miss_threshold_m),
        "failure_penalty": float(args.failure_penalty),
        "progress_weight": float(args.progress_weight),
        "ade_weight": float(args.ade_weight),
        "fde_weight": float(args.fde_weight),
        "clearance_weight": float(args.clearance_weight),
    }
    pipeline = run_candidate_regret_pipeline(
        g0_candidates=args.g0_candidates,
        g0_metrics=args.g0_metrics,
        g1_candidates=args.g1_candidates,
        g1_metrics=args.g1_metrics,
        output_dir=args.output_dir,
        score_config=score_config,
        utility_regret_threshold=float(args.utility_regret_threshold),
    )
    print(markdown_pipeline_summary(pipeline))
    return 0


def run_candidate_regret_pipeline(
    *,
    g0_candidates: Path,
    g0_metrics: Path | None,
    g1_candidates: Path | None,
    g1_metrics: Path | None,
    output_dir: Path,
    score_config: Mapping[str, float],
    utility_regret_threshold: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stages = []
    g0 = process_stage(
        name="g0",
        candidate_path=g0_candidates,
        metrics_path=g0_metrics,
        output_dir=output_dir,
        score_config=score_config,
        utility_regret_threshold=utility_regret_threshold,
    )
    stages.append(g0)
    g1 = None
    if g1_candidates is not None:
        g1 = process_stage(
            name="g1",
            candidate_path=g1_candidates,
            metrics_path=g1_metrics,
            output_dir=output_dir,
            score_config=score_config,
            utility_regret_threshold=utility_regret_threshold,
        )
        stages.append(g1)
    lifecycle = None
    curriculum = None
    if all(stage["analysis_ready"] for stage in stages):
        lifecycle = summarize_lifecycle([(stage["name"], stage["analysis_report"]) for stage in stages])
        (output_dir / "lifecycle.json").write_text(
            json.dumps(lifecycle, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (output_dir / "lifecycle.md").write_text(markdown_lifecycle(lifecycle) + "\n", encoding="utf-8")
        curriculum = build_candidate_regret_curriculum(
            g0["analysis_report"],
            g1_report=None if g1 is None else g1["analysis_report"],
            g0_report_path=str(output_dir / "g0_analysis.json"),
            g1_report_path=None if g1 is None else str(output_dir / "g1_analysis.json"),
            safety_base_weight=2.0,
            utility_base_weight=1.0,
            max_targets_per_phase=None,
        )
        (output_dir / "curriculum.json").write_text(
            json.dumps(curriculum, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (output_dir / "curriculum.md").write_text(markdown_curriculum(curriculum) + "\n", encoding="utf-8")
        (output_dir / "curriculum_targets.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in curriculum["training_targets"]),
            encoding="utf-8",
        )
    summary = {
        "schema": "candidate_regret_pipeline_v1",
        "output_dir": str(output_dir),
        "score_config": dict(score_config),
        "utility_regret_threshold": float(utility_regret_threshold),
        "stages": [stage_summary(stage) for stage in stages],
        "lifecycle_available": lifecycle is not None,
        "curriculum_available": curriculum is not None,
        "claim_boundary": (
            "Pipeline is analysis-complete only when every candidate stage has evaluator metrics. "
            "Generation-ready-only dumps are preserved but not used for regret claims."
        ),
    }
    (output_dir / "pipeline_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "pipeline_summary.md").write_text(markdown_pipeline_summary(summary) + "\n", encoding="utf-8")
    return summary


def process_stage(
    *,
    name: str,
    candidate_path: Path,
    metrics_path: Path | None,
    output_dir: Path,
    score_config: Mapping[str, float],
    utility_regret_threshold: float,
) -> dict[str, Any]:
    candidate_payload = read_json_or_jsonl(candidate_path)
    if metrics_path is not None:
        metric_payload = read_json_or_jsonl(metrics_path)
        candidate_payload, join_report = attach_candidate_metrics(
            candidate_payload,
            metric_payload,
            candidate_json=str(candidate_path),
            metrics_json=str(metrics_path),
            near_miss_threshold_m=float(score_config["near_miss_threshold_m"]),
        )
        (output_dir / f"{name}_metric_join.json").write_text(
            json.dumps(join_report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (output_dir / f"{name}_candidates_with_metrics.json").write_text(
            json.dumps(candidate_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    candidate_sets = normalize_candidate_sets(
        candidate_payload,
        near_miss_threshold_m=float(score_config["near_miss_threshold_m"]),
    )
    validation = validate_candidate_generator_dump(candidate_sets, input_json=str(candidate_path))
    (output_dir / f"{name}_validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / f"{name}_validation.md").write_text(markdown_validation(validation) + "\n", encoding="utf-8")
    analysis_report = None
    if validation["readiness"] == "analysis_ready":
        analysis_report = analyze_candidate_recoverable_regret(
            candidate_sets,
            input_json=str(candidate_path),
            score_config=score_config,
            utility_regret_threshold=utility_regret_threshold,
            target_limit=None,
        )
        (output_dir / f"{name}_analysis.json").write_text(
            json.dumps(analysis_report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (output_dir / f"{name}_targets.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in analysis_report["bootstrap_targets"]),
            encoding="utf-8",
        )
    return {
        "name": name,
        "candidate_path": str(candidate_path),
        "metrics_path": None if metrics_path is None else str(metrics_path),
        "validation": validation,
        "analysis_ready": validation["readiness"] == "analysis_ready",
        "analysis_report": analysis_report,
    }


def stage_summary(stage: Mapping[str, Any]) -> dict[str, Any]:
    validation = stage["validation"]
    analysis = stage.get("analysis_report")
    result = {
        "name": stage["name"],
        "candidate_path": stage["candidate_path"],
        "metrics_path": stage["metrics_path"],
        "readiness": validation["readiness"],
        "scene_count": validation["scene_count"],
        "candidate_count": validation["candidate_count"],
    }
    if analysis is not None:
        result.update(
            {
                "generation_gap_count": analysis["gap_decomposition"]["generation_gap_count"],
                "recoverable_safety_regret_count": analysis["gap_decomposition"][
                    "recoverable_safety_regret_count"
                ],
                "recoverable_utility_regret_count": analysis["gap_decomposition"][
                    "recoverable_utility_regret_count"
                ],
                "mean_recoverable_regret": analysis["score_summary"]["mean_recoverable_regret"],
                "top1_replay_fail_rate": analysis["score_summary"]["top1_replay_fail_rate"],
            }
        )
    return result


def markdown_pipeline_summary(report: Mapping[str, Any]) -> str:
    lines = [
        "# Candidate Regret Pipeline",
        "",
        f"- Output dir: `{report['output_dir']}`",
        f"- Lifecycle available: `{report['lifecycle_available']}`",
        f"- Curriculum available: `{report['curriculum_available']}`",
        f"- Claim boundary: {report['claim_boundary']}",
        "",
        "| Stage | Readiness | Scenes | Candidates | Gen gap | Safety regret | Utility regret | Top-1 fail |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for stage in report["stages"]:
        lines.append(
            f"| {stage['name']} | {stage['readiness']} | {stage['scene_count']} | {stage['candidate_count']} | "
            f"{optional(stage.get('generation_gap_count'))} | "
            f"{optional(stage.get('recoverable_safety_regret_count'))} | "
            f"{optional(stage.get('recoverable_utility_regret_count'))} | "
            f"{optional(stage.get('top1_replay_fail_rate'))} |"
        )
    lines.append("")
    return "\n".join(lines)


def optional(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
