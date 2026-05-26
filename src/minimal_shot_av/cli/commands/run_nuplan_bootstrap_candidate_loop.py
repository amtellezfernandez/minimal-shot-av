#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any
from typing import Mapping

from minimal_shot_av.cli.commands.analyze_nuplan_bootstrap_candidate_calibration import (
    candidate_distribution,
)
from minimal_shot_av.cli.commands.analyze_nuplan_bootstrap_candidate_calibration import (
    generation_selection_gap_report,
)
from minimal_shot_av.cli.commands.analyze_nuplan_bootstrap_candidate_calibration import _rate
from minimal_shot_av.cli.commands.train_nuplan_replay_calibrated_selector import _candidate_by_token
from minimal_shot_av.cli.commands.train_nuplan_replay_calibrated_selector import _replay_by_token
from minimal_shot_av.cli.commands.train_nuplan_replay_calibrated_selector import _select_replay_oracle_token
from minimal_shot_av.cli.commands.train_nuplan_replay_calibrated_selector import candidate_logged_ego_utility
from minimal_shot_av.cli.commands.train_nuplan_replay_calibrated_selector import evaluate_selector_policy


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_INPUT = ROOT / "artifacts" / "corl2027" / "nuplan_public_replay_study_interaction250" / "replay.json"
DEFAULT_TARGETS = ROOT / "artifacts" / "corl2027" / "nuplan_bootstrap_teacher_targets.jsonl"
DEFAULT_OUTPUT_JSON = ROOT / "artifacts" / "corl2027" / "nuplan_bootstrap_candidate_loop.json"
DEFAULT_OUTPUT_MARKDOWN = ROOT / "artifacts" / "corl2027" / "nuplan_bootstrap_candidate_loop.md"
DEFAULT_OUTPUT_G1 = ROOT / "artifacts" / "corl2027" / "nuplan_bootstrap_candidate_loop_g1_replay.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a G0-to-G1 teacher-promotion bootstrap loop over nuPlan replay candidates."
    )
    parser.add_argument("--input-replay-json", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--teacher-targets-jsonl", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_OUTPUT_MARKDOWN)
    parser.add_argument("--output-g1-replay-json", type=Path, default=DEFAULT_OUTPUT_G1)
    parser.add_argument("--near-miss-threshold-m", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    replay_report = json.loads(args.input_replay_json.read_text(encoding="utf-8"))
    targets = load_teacher_targets(args.teacher_targets_jsonl)
    g1_report, application = build_teacher_promoted_g1_replay_report(replay_report, targets)
    report = analyze_bootstrap_candidate_loop(
        replay_report,
        g1_report,
        targets,
        application=application,
        input_replay_json=str(args.input_replay_json),
        teacher_targets_jsonl=str(args.teacher_targets_jsonl),
        near_miss_threshold_m=float(args.near_miss_threshold_m),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text(markdown_loop_report(report) + "\n", encoding="utf-8")
    args.output_g1_replay_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_g1_replay_json.write_text(json.dumps(g1_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(markdown_loop_report(report))
    return 0


def load_teacher_targets(path: Path) -> list[dict[str, Any]]:
    targets = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        targets.append(json.loads(line))
    return targets


def build_teacher_promoted_g1_replay_report(
    replay_report: Mapping[str, Any],
    targets: list[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    copied = json.loads(json.dumps(replay_report))
    target_by_key = {_target_key(target): target for target in targets}
    target_by_scene_id = {str(target.get("scene_id", "")): target for target in targets}
    token_histogram: Counter[str] = Counter()
    stats = {
        "strategy": "promote_teacher_top1",
        "target_count": len(targets),
        "scene_count": len(list(copied.get("scenes", []))),
        "applied_count": 0,
        "missing_target_count": 0,
        "missing_candidate_count": 0,
        "changed_token_count": 0,
        "unchanged_token_count": 0,
    }
    for scene in copied.get("scenes", []):
        target = target_by_key.get(_scene_key(scene)) or target_by_scene_id.get(str(scene.get("scene_id", "")))
        if target is None:
            stats["missing_target_count"] += 1
            continue
        teacher_token = str(target.get("teacher_token", ""))
        candidate = _candidate_by_token(scene).get(teacher_token)
        replay = _replay_by_token(scene).get(teacher_token)
        if candidate is None or replay is None:
            stats["missing_candidate_count"] += 1
            continue
        previous_token = str(scene.get("selected_token", ""))
        if teacher_token == previous_token:
            stats["unchanged_token_count"] += 1
        else:
            stats["changed_token_count"] += 1
        stats["applied_count"] += 1
        token_histogram[teacher_token] += 1
        scene["selected_token"] = teacher_token
        scene["selected_token_source"] = "bootstrap_teacher_promoted"
        scene["selected_token_realized_min_clearance_m"] = replay.get("realized_min_clearance_m")
        scene["selected_token_realized_near_miss"] = replay.get("realized_near_miss")
        scene["selected_token_proxy_safe"] = bool(candidate.get("proxy_safe", False))
        scene["selected_token_min_proxy_clearance_m"] = candidate.get("min_proxy_clearance_m")
        scene["selected_token_score"] = candidate.get("score")
        scene["bootstrap_loop"] = {
            "strategy": "promote_teacher_top1",
            "teacher": target.get("teacher"),
            "teacher_token": teacher_token,
            "previous_selected_token": previous_token,
            "target_scene_id": target.get("scene_id"),
        }
    stats["applied_rate"] = _rate(int(stats["applied_count"]), int(stats["scene_count"]))
    stats["changed_token_rate"] = _rate(int(stats["changed_token_count"]), int(stats["scene_count"]))
    stats["teacher_token_histogram"] = dict(sorted(token_histogram.items()))
    copied["scene_count"] = len(list(copied.get("scenes", [])))
    copied["bootstrap_loop"] = stats
    return copied, stats


def analyze_bootstrap_candidate_loop(
    g0_replay_report: Mapping[str, Any],
    g1_replay_report: Mapping[str, Any],
    targets: list[Mapping[str, Any]],
    *,
    application: Mapping[str, Any],
    input_replay_json: str,
    teacher_targets_jsonl: str,
    near_miss_threshold_m: float,
) -> dict[str, Any]:
    g1_target_fn = _teacher_target_selector(targets)
    selector_fns = {
        "proxy_selector": lambda scene: str(scene.get("selected_token", "")),
        "replay_oracle": _select_replay_oracle_token,
    }
    g0_gap = generation_selection_gap_report(
        g0_replay_report,
        selector_fns=selector_fns,
        near_miss_threshold_m=near_miss_threshold_m,
    )
    g1_gap = generation_selection_gap_report(
        g1_replay_report,
        selector_fns=selector_fns,
        near_miss_threshold_m=near_miss_threshold_m,
    )
    g0_metrics = evaluate_selector_policy(
        g0_replay_report,
        selector_name="g0_proxy_top1",
        selected_token_fn=lambda scene: str(scene.get("selected_token", "")),
        near_miss_threshold_m=near_miss_threshold_m,
    )
    g1_metrics = evaluate_selector_policy(
        g0_replay_report,
        selector_name="g1_teacher_promoted_top1",
        selected_token_fn=g1_target_fn,
        near_miss_threshold_m=near_miss_threshold_m,
    )
    oracle_metrics = evaluate_selector_policy(
        g0_replay_report,
        selector_name="replay_oracle",
        selected_token_fn=_select_replay_oracle_token,
        near_miss_threshold_m=near_miss_threshold_m,
    )
    g0_selection_gap = int(g0_gap["proxy_selection_gap_count"])
    g1_selection_gap = int(g1_gap["proxy_selection_gap_count"])
    generation_gap_delta = int(g1_gap["generation_gap_count"]) - int(g0_gap["generation_gap_count"])
    return {
        "schema": "nuplan_bootstrap_candidate_loop_v1",
        "input_replay_json": input_replay_json,
        "teacher_targets_jsonl": teacher_targets_jsonl,
        "near_miss_threshold_m": float(near_miss_threshold_m),
        "method": {
            "name": "teacher_promoted_g1_surrogate",
            "claim_boundary": (
                "Promotes an existing teacher-selected candidate to G1 top-1. "
                "It measures selection-gap closure, not new-candidate generation."
            ),
        },
        "g1_application": dict(application),
        "g0_candidate_distribution": candidate_distribution(g0_replay_report),
        "g1_candidate_distribution": candidate_distribution(g1_replay_report),
        "gap_report": {
            "g0": g0_gap,
            "g1": g1_gap,
            "selection_gap_closed_count": g0_selection_gap - g1_selection_gap,
            "selection_gap_closure_rate": _rate(g0_selection_gap - g1_selection_gap, g0_selection_gap),
            "generation_gap_delta_count": generation_gap_delta,
        },
        "selector_metrics": {
            "g0_proxy_top1": g0_metrics,
            "g1_teacher_promoted_top1": g1_metrics,
            "replay_oracle": oracle_metrics,
        },
        "teacher_supervision": teacher_supervision_summary(g0_replay_report, targets),
    }


def teacher_supervision_summary(
    replay_report: Mapping[str, Any],
    targets: list[Mapping[str, Any]],
) -> dict[str, Any]:
    target_fn = _teacher_target_selector(targets)
    progress_deltas = []
    ade_deltas = []
    changed = 0
    pose_shift_values = []
    for scene in replay_report.get("scenes", []):
        proxy_token = str(scene.get("selected_token", ""))
        teacher_token = str(target_fn(scene))
        if proxy_token == teacher_token:
            continue
        candidate_by_token = _candidate_by_token(scene)
        proxy_candidate = candidate_by_token.get(proxy_token)
        teacher_candidate = candidate_by_token.get(teacher_token)
        if proxy_candidate is None or teacher_candidate is None:
            continue
        changed += 1
        progress_deltas.append(
            float(teacher_candidate.get("final_progress_m", 0.0)) - float(proxy_candidate.get("final_progress_m", 0.0))
        )
        proxy_utility = candidate_logged_ego_utility(proxy_candidate, scene)
        teacher_utility = candidate_logged_ego_utility(teacher_candidate, scene)
        if proxy_utility is not None and teacher_utility is not None:
            ade_deltas.append(float(teacher_utility["ade_3s_m"]) - float(proxy_utility["ade_3s_m"]))
        pose_shift = _final_pose_shift(proxy_candidate, teacher_candidate)
        if pose_shift is not None:
            pose_shift_values.append(pose_shift)
    return {
        "changed_token_count": changed,
        "mean_teacher_minus_proxy_progress_m": _mean(progress_deltas),
        "mean_teacher_minus_proxy_ade_3s_m": _mean(ade_deltas),
        "mean_final_pose_shift_m": _mean(pose_shift_values),
    }


def markdown_loop_report(report: Mapping[str, Any]) -> str:
    g0_gap = report["gap_report"]["g0"]
    g1_gap = report["gap_report"]["g1"]
    lines = [
        "# nuPlan Bootstrap Candidate Loop",
        "",
        f"- Input replay JSON: `{report['input_replay_json']}`",
        f"- Teacher targets: `{report['teacher_targets_jsonl']}`",
        f"- Method: `{report['method']['name']}`",
        f"- Claim boundary: {report['method']['claim_boundary']}",
        "",
        "## G1 Application",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| scenes | {report['g1_application']['scene_count']} |",
        f"| targets | {report['g1_application']['target_count']} |",
        f"| applied | {report['g1_application']['applied_count']} |",
        f"| changed top-1 token | {report['g1_application']['changed_token_count']} |",
        f"| missing target | {report['g1_application']['missing_target_count']} |",
        f"| missing candidate | {report['g1_application']['missing_candidate_count']} |",
        "",
        "## Gap Movement",
        "",
        "| Gap | G0 | G1 | Delta |",
        "|---|---:|---:|---:|",
        _gap_row("oracle@K safe", g0_gap, g1_gap, "oracle_at_k_safe_count"),
        _gap_row("generation gap", g0_gap, g1_gap, "generation_gap_count"),
        _gap_row("top-1 safe", g0_gap, g1_gap, "proxy_top1_safe_count"),
        _gap_row("selection gap", g0_gap, g1_gap, "proxy_selection_gap_count"),
        "",
        "## Selector Metrics",
        "",
        "| Selector | Replay-infeasible | Rate | Recovery | Progress | ADE3 | Entropy |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in report["selector_metrics"].items():
        lines.append(
            f"| {name} | {metrics['selected_replay_infeasible_count']} | "
            f"{metrics['selected_replay_infeasible_rate']:.3f} | "
            f"{metrics['recoverable_proxy_optimism_fixed_count']}/"
            f"{metrics['recoverable_proxy_optimism_count']} | "
            f"{metrics['mean_selected_progress_m']:.3f} | "
            f"{_optional(metrics['mean_ade_3s_m'])} | "
            f"{metrics['selected_token_entropy']:.3f} |"
        )
    teacher = report["teacher_supervision"]
    lines.extend(
        [
            "",
            "## Teacher Supervision Burden",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| changed top-1 tokens | {teacher['changed_token_count']} |",
            f"| mean progress delta | {teacher['mean_teacher_minus_proxy_progress_m']:.3f} |",
            f"| mean ADE3 delta | {teacher['mean_teacher_minus_proxy_ade_3s_m']:.3f} |",
            f"| mean final-pose shift | {teacher['mean_final_pose_shift_m']:.3f} |",
            "",
        ]
    )
    return "\n".join(lines)


def _teacher_target_selector(targets: list[Mapping[str, Any]]):
    target_by_key = {_target_key(target): target for target in targets}
    target_by_scene_id = {str(target.get("scene_id", "")): target for target in targets}

    def select(scene: Mapping[str, Any]) -> str:
        target = target_by_key.get(_scene_key(scene)) or target_by_scene_id.get(str(scene.get("scene_id", "")))
        if target is None:
            return str(scene.get("selected_token", ""))
        return str(target.get("teacher_token", scene.get("selected_token", "")))

    return select


def _scene_key(scene: Mapping[str, Any]) -> str:
    return f"{scene.get('source_db_file', '')}::{scene.get('scene_id', '')}"


def _target_key(target: Mapping[str, Any]) -> str:
    return f"{target.get('source_db_file', '')}::{target.get('scene_id', '')}"


def _gap_row(label: str, g0_gap: Mapping[str, Any], g1_gap: Mapping[str, Any], key: str) -> str:
    left = int(g0_gap[key])
    right = int(g1_gap[key])
    return f"| {label} | {left} | {right} | {right - left:+d} |"


def _final_pose_shift(left_candidate: Mapping[str, Any], right_candidate: Mapping[str, Any]) -> float | None:
    left_poses = list(left_candidate.get("poses", []))
    right_poses = list(right_candidate.get("poses", []))
    if not left_poses or not right_poses:
        return None
    left = left_poses[-1]
    right = right_poses[-1]
    dx = float(right[0]) - float(left[0])
    dy = float(right[1]) - float(left[1])
    return (dx * dx + dy * dy) ** 0.5


def _mean(values: list[float]) -> float:
    return round(sum(float(value) for value in values) / len(values), 6) if values else 0.0


def _optional(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
