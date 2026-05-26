#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any
from typing import Mapping


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT_JSON = ROOT / "artifacts" / "corl2027" / "candidate_regret_curriculum.json"
DEFAULT_OUTPUT_MD = ROOT / "artifacts" / "corl2027" / "candidate_regret_curriculum.md"
DEFAULT_TARGETS_JSONL = ROOT / "artifacts" / "corl2027" / "candidate_regret_curriculum_targets.jsonl"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a staged generator-training curriculum from recoverable-regret reports: "
            "safety recovery, utility recovery, and candidate-expansion requests."
        )
    )
    parser.add_argument("--g0-report", type=Path, required=True)
    parser.add_argument("--g1-report", type=Path)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--targets-jsonl", type=Path, default=DEFAULT_TARGETS_JSONL)
    parser.add_argument("--safety-base-weight", type=float, default=2.0)
    parser.add_argument("--utility-base-weight", type=float, default=1.0)
    parser.add_argument("--max-targets-per-phase", type=int)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    g0_report = json.loads(args.g0_report.read_text(encoding="utf-8"))
    g1_report = json.loads(args.g1_report.read_text(encoding="utf-8")) if args.g1_report else None
    curriculum = build_candidate_regret_curriculum(
        g0_report,
        g1_report=g1_report,
        g0_report_path=str(args.g0_report),
        g1_report_path=None if args.g1_report is None else str(args.g1_report),
        safety_base_weight=float(args.safety_base_weight),
        utility_base_weight=float(args.utility_base_weight),
        max_targets_per_phase=args.max_targets_per_phase,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(curriculum, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text(markdown_curriculum(curriculum) + "\n", encoding="utf-8")
    args.targets_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.targets_jsonl.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in curriculum["training_targets"]),
        encoding="utf-8",
    )
    print(markdown_curriculum(curriculum))
    return 0


def build_candidate_regret_curriculum(
    g0_report: Mapping[str, Any],
    *,
    g1_report: Mapping[str, Any] | None,
    g0_report_path: str,
    g1_report_path: str | None,
    safety_base_weight: float,
    utility_base_weight: float,
    max_targets_per_phase: int | None,
) -> dict[str, Any]:
    safety_targets = curriculum_targets(
        g0_report,
        phase="safety_recovery",
        source_stage="g0",
        regret_type="recoverable_safety_regret",
        base_weight=safety_base_weight,
        max_targets=max_targets_per_phase,
    )
    utility_source = g1_report if g1_report is not None else g0_report
    utility_stage = "g1" if g1_report is not None else "g0"
    utility_targets = curriculum_targets(
        utility_source,
        phase="utility_recovery",
        source_stage=utility_stage,
        regret_type="recoverable_utility_regret",
        base_weight=utility_base_weight,
        max_targets=max_targets_per_phase,
    )
    expansion_source = g1_report if g1_report is not None else g0_report
    expansion_stage = "g1" if g1_report is not None else "g0"
    expansion_requests = candidate_expansion_requests(expansion_source, source_stage=expansion_stage)
    training_targets = safety_targets + utility_targets
    return {
        "schema": "candidate_regret_curriculum_v1",
        "g0_report": g0_report_path,
        "g1_report": g1_report_path,
        "claim_boundary": (
            "Curriculum for generator bootstrapping. Safety and utility phases have in-bank teacher "
            "trajectories; generation-gap scenes are expansion requests because no candidate in K is safe."
        ),
        "phase_order": ["safety_recovery", "utility_recovery", "candidate_expansion"],
        "phase_summary": {
            "safety_recovery": phase_summary(safety_targets),
            "utility_recovery": phase_summary(utility_targets),
            "candidate_expansion": expansion_summary(expansion_requests),
        },
        "training_target_count": len(training_targets),
        "training_targets": training_targets,
        "candidate_expansion_request_count": len(expansion_requests),
        "candidate_expansion_requests": expansion_requests,
    }


def curriculum_targets(
    report: Mapping[str, Any],
    *,
    phase: str,
    source_stage: str,
    regret_type: str,
    base_weight: float,
    max_targets: int | None,
) -> list[dict[str, Any]]:
    targets = [
        target
        for target in report.get("bootstrap_targets", [])
        if str(target.get("regret_type", "")) == regret_type
    ]
    targets = sorted(targets, key=lambda row: float(row.get("recoverable_regret", 0.0)), reverse=True)
    if max_targets is not None:
        targets = targets[: int(max_targets)]
    max_regret = max([float(target.get("recoverable_regret", 0.0)) for target in targets] or [1.0])
    max_regret = max(max_regret, 1.0e-6)
    rows = []
    for target in targets:
        regret = float(target.get("recoverable_regret", 0.0))
        normalized = regret / max_regret
        rows.append(
            {
                "phase": phase,
                "source_stage": source_stage,
                "scene_id": str(target.get("scene_id", "")),
                "source": str(target.get("source", "")),
                "scenario_type": str(target.get("scenario_type", "")),
                "generator": str(target.get("generator", "")),
                "top1_candidate_id": str(target.get("top1_candidate_id", "")),
                "teacher_candidate_id": str(target.get("teacher_candidate_id", "")),
                "regret_type": regret_type,
                "recoverable_regret": regret,
                "loss_weight": round(float(base_weight) * (0.5 + normalized), 6),
                "top1_metrics": target.get("top1_metrics", {}),
                "teacher_metrics": target.get("teacher_metrics", {}),
                "top1_trajectory": target.get("top1_trajectory"),
                "teacher_trajectory": target.get("teacher_trajectory"),
            }
        )
    return rows


def candidate_expansion_requests(report: Mapping[str, Any], *, source_stage: str) -> list[dict[str, Any]]:
    records = [
        record
        for record in report.get("regret_records", [])
        if bool(record.get("valid")) and str(record.get("regret_type", "")) == "generation_gap"
    ]
    rows = []
    for record in records:
        rows.append(
            {
                "phase": "candidate_expansion",
                "source_stage": source_stage,
                "scene_id": str(record.get("scene_id", "")),
                "source": str(record.get("source", "")),
                "scenario_type": str(record.get("scenario_type", "")),
                "generator": str(record.get("generator", "")),
                "candidate_count": int(record.get("candidate_count", 0)),
                "top1_candidate_id": str(record.get("top1_candidate_id", "")),
                "oracle_candidate_id": str(record.get("oracle_candidate_id", "")),
                "top1_metrics": record.get("top1_metrics", {}),
                "oracle_metrics": record.get("oracle_metrics", {}),
                "request": "expand_candidate_distribution",
                "reason": "oracle_at_k_replay_fails_no_in_bank_safe_teacher",
            }
        )
    return rows


def phase_summary(targets: list[Mapping[str, Any]]) -> dict[str, Any]:
    weights = [float(target.get("loss_weight", 0.0)) for target in targets]
    regrets = [float(target.get("recoverable_regret", 0.0)) for target in targets]
    return {
        "target_count": len(targets),
        "mean_loss_weight": mean(weights),
        "mean_recoverable_regret": mean(regrets),
        "teacher_histogram": dict(
            sorted(Counter(str(target.get("teacher_candidate_id", "")) for target in targets).items())
        ),
    }


def expansion_summary(requests: list[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "request_count": len(requests),
        "top1_histogram": dict(sorted(Counter(str(row.get("top1_candidate_id", "")) for row in requests).items())),
        "generator_histogram": dict(sorted(Counter(str(row.get("generator", "")) for row in requests).items())),
    }


def markdown_curriculum(report: Mapping[str, Any]) -> str:
    phases = report["phase_summary"]
    lines = [
        "# Candidate Regret Curriculum",
        "",
        f"- G0 report: `{report['g0_report']}`",
        f"- G1 report: `{report['g1_report']}`",
        f"- Claim boundary: {report['claim_boundary']}",
        "",
        "## Phase Summary",
        "",
        "| Phase | Count | Mean regret | Mean weight | Main histogram |",
        "|---|---:|---:|---:|---|",
    ]
    safety = phases["safety_recovery"]
    utility = phases["utility_recovery"]
    expansion = phases["candidate_expansion"]
    lines.append(phase_row("safety_recovery", safety, "teacher_histogram"))
    lines.append(phase_row("utility_recovery", utility, "teacher_histogram"))
    lines.append(
        f"| candidate_expansion | {expansion['request_count']} | n/a | n/a | "
        f"`{json.dumps(expansion['top1_histogram'], sort_keys=True)}` |"
    )
    lines.extend(
        [
            "",
            "## Training Use",
            "",
            "- Phase 1 trains the generator toward replay-safe teachers where top-1 failed.",
            "- Phase 2 trains utility recovery after safety regret is mostly removed.",
            "- Candidate-expansion requests require new samples or a richer generator; no in-bank teacher exists.",
            "",
        ]
    )
    return "\n".join(lines)


def phase_row(name: str, summary: Mapping[str, Any], histogram_key: str) -> str:
    return (
        f"| {name} | {summary['target_count']} | {summary['mean_recoverable_regret']:.3f} | "
        f"{summary['mean_loss_weight']:.3f} | `{json.dumps(summary[histogram_key], sort_keys=True)}` |"
    )


def mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
