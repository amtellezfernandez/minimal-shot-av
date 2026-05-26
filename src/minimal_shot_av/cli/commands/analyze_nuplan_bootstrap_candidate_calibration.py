#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Mapping

from minimal_shot_av.cli.commands.train_nuplan_boundary_intervention_selector import MODEL_TYPE as BOUNDARY_MODEL_TYPE
from minimal_shot_av.cli.commands.train_nuplan_boundary_intervention_selector import (
    load_boundary_intervention_policy,
)
from minimal_shot_av.cli.commands.train_nuplan_boundary_intervention_selector import (
    select_with_boundary_intervention,
)
from minimal_shot_av.cli.commands.train_nuplan_replay_calibrated_selector import _candidate_by_token
from minimal_shot_av.cli.commands.train_nuplan_replay_calibrated_selector import _replay_by_token
from minimal_shot_av.cli.commands.train_nuplan_replay_calibrated_selector import _select_replay_oracle_token
from minimal_shot_av.cli.commands.train_nuplan_replay_calibrated_selector import _select_with_score_model
from minimal_shot_av.cli.commands.train_nuplan_replay_calibrated_selector import _token_is_replay_safe
from minimal_shot_av.cli.commands.train_nuplan_replay_calibrated_selector import candidate_logged_ego_utility
from minimal_shot_av.cli.commands.train_nuplan_replay_calibrated_selector import evaluate_selector_policy
from minimal_shot_av.model.nuplan_maneuver_token_selector import load_selector


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_INPUT = ROOT / "artifacts" / "corl2027" / "nuplan_public_replay_study_interaction250" / "replay.json"
DEFAULT_OUTPUT_JSON = ROOT / "artifacts" / "corl2027" / "nuplan_bootstrap_candidate_calibration.json"
DEFAULT_OUTPUT_MARKDOWN = ROOT / "artifacts" / "corl2027" / "nuplan_bootstrap_candidate_calibration.md"
DEFAULT_TARGETS_JSONL = ROOT / "artifacts" / "corl2027" / "nuplan_bootstrap_teacher_targets.jsonl"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit candidate generation and replay-calibrated selection gaps for bootstrapping experiments."
    )
    parser.add_argument("--input-replay-json", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_OUTPUT_MARKDOWN)
    parser.add_argument("--targets-jsonl", type=Path, default=DEFAULT_TARGETS_JSONL)
    parser.add_argument("--near-miss-threshold-m", type=float, default=1.0)
    parser.add_argument(
        "--selector",
        action="append",
        default=[],
        help="Named selector artifact in NAME=PATH form. Can be repeated.",
    )
    parser.add_argument(
        "--teacher",
        default="replay_oracle",
        help="Teacher used for bootstrap targets: replay_oracle, proxy_selector, or a named --selector.",
    )
    parser.add_argument("--target-limit", type=int, help="Optional maximum number of bootstrap targets to emit.")
    args = parser.parse_args()
    args.selector = list(args.selector or [])
    return args


def main() -> int:
    args = _parse_args()
    replay_report = json.loads(args.input_replay_json.read_text(encoding="utf-8"))
    selector_fns = load_named_selector_fns(args.selector)
    report = analyze_bootstrap_candidate_calibration(
        replay_report,
        input_replay_json=str(args.input_replay_json),
        selector_fns=selector_fns,
        teacher_name=str(args.teacher),
        near_miss_threshold_m=float(args.near_miss_threshold_m),
        target_limit=args.target_limit,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text(markdown_report(report) + "\n", encoding="utf-8")
    args.targets_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.targets_jsonl.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in report["bootstrap_targets"]),
        encoding="utf-8",
    )
    print(markdown_report(report))
    return 0


def load_named_selector_fns(selector_args: list[str]) -> dict[str, Callable[[Mapping[str, Any]], str]]:
    selectors: dict[str, Callable[[Mapping[str, Any]], str]] = {}
    for raw in selector_args:
        if "=" not in raw:
            raise ValueError(f"--selector must be NAME=PATH, got {raw!r}")
        name, path = raw.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"--selector has empty name: {raw!r}")
        selector_path = Path(path)
        payload = json.loads(selector_path.read_text(encoding="utf-8"))
        if str(payload.get("model_type", "")) == BOUNDARY_MODEL_TYPE:
            policy = load_boundary_intervention_policy(selector_path)
            baseline_path = Path(str(payload["training"]["baseline_selector_model"]))
            baseline_selector = load_selector(baseline_path)
            selectors[name] = lambda scene, baseline=baseline_selector, boundary_policy=policy: (
                select_with_boundary_intervention(
                    scene,
                    baseline_selector=baseline,
                    policy=boundary_policy,
                )
            )
        else:
            selector = load_selector(selector_path)
            selectors[name] = lambda scene, model=selector: _select_with_score_model(scene, model)
    return selectors


def analyze_bootstrap_candidate_calibration(
    replay_report: Mapping[str, Any],
    *,
    input_replay_json: str,
    selector_fns: Mapping[str, Callable[[Mapping[str, Any]], str]],
    teacher_name: str,
    near_miss_threshold_m: float,
    target_limit: int | None,
) -> dict[str, Any]:
    selector_fns: dict[str, Callable[[Mapping[str, Any]], str]] = {
        "proxy_selector": lambda scene: str(scene.get("selected_token", "")),
        **dict(selector_fns),
        "replay_oracle": _select_replay_oracle_token,
    }
    if teacher_name not in selector_fns:
        raise ValueError(f"unknown teacher {teacher_name!r}; available: {sorted(selector_fns)}")
    scene_count = len(list(replay_report.get("scenes", [])))
    selector_metrics = {
        name: evaluate_selector_policy(
            replay_report,
            selector_name=name,
            selected_token_fn=fn,
            near_miss_threshold_m=near_miss_threshold_m,
        )
        for name, fn in selector_fns.items()
    }
    gap_report = generation_selection_gap_report(
        replay_report,
        selector_fns=selector_fns,
        near_miss_threshold_m=near_miss_threshold_m,
    )
    targets = bootstrap_targets(
        replay_report,
        teacher_fn=selector_fns[teacher_name],
        teacher_name=teacher_name,
        near_miss_threshold_m=near_miss_threshold_m,
        target_limit=target_limit,
    )
    bootstrap_summary = bootstrap_iteration_summary(
        replay_report,
        teacher_fn=selector_fns[teacher_name],
        near_miss_threshold_m=near_miss_threshold_m,
    )
    return {
        "schema": "nuplan_bootstrap_candidate_calibration_v1",
        "input_replay_json": input_replay_json,
        "near_miss_threshold_m": float(near_miss_threshold_m),
        "scene_count": scene_count,
        "candidate_distribution": candidate_distribution(replay_report),
        "selector_metrics": selector_metrics,
        "gap_report": gap_report,
        "teacher": teacher_name,
        "bootstrap_iteration_summary": bootstrap_summary,
        "bootstrap_target_count": len(targets),
        "bootstrap_target_token_histogram": dict(sorted(Counter(row["teacher_token"] for row in targets).items())),
        "bootstrap_targets": targets,
    }


def generation_selection_gap_report(
    replay_report: Mapping[str, Any],
    *,
    selector_fns: Mapping[str, Callable[[Mapping[str, Any]], str]],
    near_miss_threshold_m: float,
) -> dict[str, Any]:
    scenes = list(replay_report.get("scenes", []))
    generation_failures = 0
    proxy_selection_failures = 0
    proxy_safe_count = 0
    per_selector = {
        name: {
            "resolved_selection_gap_count": 0,
            "regressed_generation_gap_count": 0,
        }
        for name in selector_fns
        if name != "proxy_selector"
    }
    for scene in scenes:
        oracle_token = scene.get("oracle_log_replay_safe_token")
        oracle_safe = oracle_token is not None and _token_is_replay_safe(
            scene, str(oracle_token), near_miss_threshold_m
        )
        if not oracle_safe:
            generation_failures += 1
        proxy_token = str(scene.get("selected_token", ""))
        proxy_safe = _token_is_replay_safe(scene, proxy_token, near_miss_threshold_m)
        if proxy_safe:
            proxy_safe_count += 1
        selection_gap = oracle_safe and not proxy_safe
        if selection_gap:
            proxy_selection_failures += 1
        for name, fn in selector_fns.items():
            if name == "proxy_selector":
                continue
            token = str(fn(scene))
            token_safe = _token_is_replay_safe(scene, token, near_miss_threshold_m)
            if selection_gap and token_safe:
                per_selector[name]["resolved_selection_gap_count"] += 1
            if not oracle_safe and token_safe:
                per_selector[name]["regressed_generation_gap_count"] += 1
    scene_count = len(scenes)
    for metrics in per_selector.values():
        resolved = int(metrics["resolved_selection_gap_count"])
        metrics["resolved_selection_gap_rate"] = (
            round(resolved / proxy_selection_failures, 6) if proxy_selection_failures else 0.0
        )
    return {
        "oracle_at_k_safe_count": scene_count - generation_failures,
        "oracle_at_k_safe_rate": _rate(scene_count - generation_failures, scene_count),
        "generation_gap_count": generation_failures,
        "generation_gap_rate": _rate(generation_failures, scene_count),
        "proxy_top1_safe_count": proxy_safe_count,
        "proxy_top1_safe_rate": _rate(proxy_safe_count, scene_count),
        "proxy_selection_gap_count": proxy_selection_failures,
        "proxy_selection_gap_rate": _rate(proxy_selection_failures, scene_count),
        "per_selector": per_selector,
    }


def bootstrap_iteration_summary(
    replay_report: Mapping[str, Any],
    *,
    teacher_fn: Callable[[Mapping[str, Any]], str],
    near_miss_threshold_m: float,
) -> dict[str, Any]:
    token_changes = 0
    replay_safe_improvements = 0
    replay_safe_regressions = 0
    clearance_deltas = []
    progress_deltas = []
    ade_deltas = []
    fde_deltas = []
    pose_shift_values = []
    scene_count = 0
    for scene in replay_report.get("scenes", []):
        proxy_token = str(scene.get("selected_token", ""))
        teacher_token = str(teacher_fn(scene))
        candidate_by_token = _candidate_by_token(scene)
        replay_by_token = _replay_by_token(scene)
        proxy_candidate = candidate_by_token.get(proxy_token)
        teacher_candidate = candidate_by_token.get(teacher_token)
        proxy_replay = replay_by_token.get(proxy_token)
        teacher_replay = replay_by_token.get(teacher_token)
        if (
            proxy_candidate is None
            or teacher_candidate is None
            or proxy_replay is None
            or teacher_replay is None
            or proxy_replay.get("realized_min_clearance_m") is None
            or teacher_replay.get("realized_min_clearance_m") is None
        ):
            continue
        scene_count += 1
        if teacher_token != proxy_token:
            token_changes += 1
        proxy_safe = float(proxy_replay["realized_min_clearance_m"]) >= near_miss_threshold_m
        teacher_safe = float(teacher_replay["realized_min_clearance_m"]) >= near_miss_threshold_m
        if teacher_safe and not proxy_safe:
            replay_safe_improvements += 1
        if proxy_safe and not teacher_safe:
            replay_safe_regressions += 1
        clearance_deltas.append(
            float(teacher_replay["realized_min_clearance_m"]) - float(proxy_replay["realized_min_clearance_m"])
        )
        progress_deltas.append(
            float(teacher_candidate.get("final_progress_m", 0.0)) - float(proxy_candidate.get("final_progress_m", 0.0))
        )
        proxy_utility = candidate_logged_ego_utility(proxy_candidate, scene)
        teacher_utility = candidate_logged_ego_utility(teacher_candidate, scene)
        if proxy_utility is not None and teacher_utility is not None:
            ade_deltas.append(float(teacher_utility["ade_3s_m"]) - float(proxy_utility["ade_3s_m"]))
            fde_deltas.append(float(teacher_utility["fde_3s_m"]) - float(proxy_utility["fde_3s_m"]))
        pose_shift = final_pose_shift(proxy_candidate, teacher_candidate)
        if pose_shift is not None:
            pose_shift_values.append(pose_shift)
    return {
        "scene_count": scene_count,
        "teacher_changes_proxy_token_count": token_changes,
        "teacher_changes_proxy_token_rate": _rate(token_changes, scene_count),
        "replay_safe_improvement_count": replay_safe_improvements,
        "replay_safe_regression_count": replay_safe_regressions,
        "mean_teacher_minus_proxy_clearance_m": _mean(clearance_deltas),
        "mean_teacher_minus_proxy_progress_m": _mean(progress_deltas),
        "mean_teacher_minus_proxy_ade_3s_m": _mean(ade_deltas),
        "mean_teacher_minus_proxy_fde_3s_m": _mean(fde_deltas),
        "mean_final_pose_shift_m": _mean(pose_shift_values),
    }


def candidate_distribution(replay_report: Mapping[str, Any]) -> dict[str, Any]:
    candidate_counts = []
    token_histogram: Counter[str] = Counter()
    diversity_values = []
    for scene in replay_report.get("scenes", []):
        candidates = list(scene.get("candidates", []))
        candidate_counts.append(len(candidates))
        token_histogram.update(str(candidate.get("token", "")) for candidate in candidates)
        diversity = mean_pairwise_final_pose_distance(candidates)
        if diversity is not None:
            diversity_values.append(diversity)
    return {
        "mean_candidate_count": _mean(candidate_counts),
        "min_candidate_count": min(candidate_counts) if candidate_counts else 0,
        "max_candidate_count": max(candidate_counts) if candidate_counts else 0,
        "candidate_token_entropy": token_entropy(token_histogram),
        "candidate_token_histogram": dict(sorted(token_histogram.items())),
        "mean_pairwise_final_pose_distance_m": _mean(diversity_values),
    }


def bootstrap_targets(
    replay_report: Mapping[str, Any],
    *,
    teacher_fn: Callable[[Mapping[str, Any]], str],
    teacher_name: str,
    near_miss_threshold_m: float,
    target_limit: int | None,
) -> list[dict[str, Any]]:
    targets = []
    for scene in replay_report.get("scenes", []):
        teacher_token = str(teacher_fn(scene))
        candidate = _candidate_by_token(scene).get(teacher_token)
        replay = _replay_by_token(scene).get(teacher_token)
        if candidate is None or replay is None or replay.get("realized_min_clearance_m") is None:
            continue
        utility = candidate_logged_ego_utility(candidate, scene) or {}
        target = {
            "scene_id": str(scene.get("scene_id", "")),
            "source_db_file": str(scene.get("source_db_file", "")),
            "scenario_type": str(scene.get("scenario_type", "")),
            "teacher": teacher_name,
            "teacher_token": teacher_token,
            "proxy_token": str(scene.get("selected_token", "")),
            "oracle_token": (
                None
                if scene.get("oracle_log_replay_safe_token") is None
                else str(scene["oracle_log_replay_safe_token"])
            ),
            "teacher_replay_safe": bool(float(replay["realized_min_clearance_m"]) >= near_miss_threshold_m),
            "teacher_replay_min_clearance_m": float(replay["realized_min_clearance_m"]),
            "teacher_progress_m": float(candidate.get("final_progress_m", 0.0)),
            "teacher_ade_3s_m": utility.get("ade_3s_m"),
            "teacher_fde_3s_m": utility.get("fde_3s_m"),
            "poses": list(candidate.get("poses", [])),
        }
        targets.append(target)
        if target_limit is not None and len(targets) >= int(target_limit):
            break
    return targets


def mean_pairwise_final_pose_distance(candidates: list[Mapping[str, Any]]) -> float | None:
    final_points = []
    for candidate in candidates:
        poses = list(candidate.get("poses", []))
        if not poses:
            continue
        final_pose = poses[-1]
        final_points.append((float(final_pose[0]), float(final_pose[1])))
    if len(final_points) < 2:
        return None
    distances = []
    for left_index, left in enumerate(final_points):
        for right in final_points[left_index + 1 :]:
            distances.append(math.hypot(left[0] - right[0], left[1] - right[1]))
    return round(sum(distances) / len(distances), 6)


def final_pose_shift(left_candidate: Mapping[str, Any], right_candidate: Mapping[str, Any]) -> float | None:
    left_poses = list(left_candidate.get("poses", []))
    right_poses = list(right_candidate.get("poses", []))
    if not left_poses or not right_poses:
        return None
    left_pose = left_poses[-1]
    right_pose = right_poses[-1]
    return round(
        math.hypot(float(right_pose[0]) - float(left_pose[0]), float(right_pose[1]) - float(left_pose[1])),
        6,
    )


def markdown_report(report: Mapping[str, Any]) -> str:
    gap = report["gap_report"]
    bootstrap = report["bootstrap_iteration_summary"]
    lines = [
        "# nuPlan Bootstrapped Candidate-Calibration Audit",
        "",
        f"- Input replay JSON: `{report['input_replay_json']}`",
        f"- Scenes: `{report['scene_count']}`",
        f"- Teacher: `{report['teacher']}`",
        "",
        "## Candidate Surface",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| mean candidate count | {report['candidate_distribution']['mean_candidate_count']:.3f} |",
        f"| candidate token entropy | {report['candidate_distribution']['candidate_token_entropy']:.3f} |",
        "| mean pairwise final-pose distance | "
        f"{report['candidate_distribution']['mean_pairwise_final_pose_distance_m']:.3f} |",
        "",
        "## Gap Decomposition",
        "",
        "| Gap | Count | Rate |",
        "|---|---:|---:|",
        f"| oracle@K safe | {gap['oracle_at_k_safe_count']} | {gap['oracle_at_k_safe_rate']:.3f} |",
        f"| generation gap | {gap['generation_gap_count']} | {gap['generation_gap_rate']:.3f} |",
        f"| proxy top-1 safe | {gap['proxy_top1_safe_count']} | {gap['proxy_top1_safe_rate']:.3f} |",
        f"| proxy selection gap | {gap['proxy_selection_gap_count']} | {gap['proxy_selection_gap_rate']:.3f} |",
        "",
        "## Bootstrap Iteration Target",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| teacher changes proxy token | {bootstrap['teacher_changes_proxy_token_count']} |",
        f"| teacher change rate | {bootstrap['teacher_changes_proxy_token_rate']:.3f} |",
        f"| replay-safe improvements | {bootstrap['replay_safe_improvement_count']} |",
        f"| replay-safe regressions | {bootstrap['replay_safe_regression_count']} |",
        f"| mean clearance delta | {bootstrap['mean_teacher_minus_proxy_clearance_m']:.3f} |",
        f"| mean progress delta | {bootstrap['mean_teacher_minus_proxy_progress_m']:.3f} |",
        f"| mean ADE3 delta | {bootstrap['mean_teacher_minus_proxy_ade_3s_m']:.3f} |",
        f"| mean final-pose shift | {bootstrap['mean_final_pose_shift_m']:.3f} |",
        "",
        "## Selector Table",
        "",
        "| Selector | Replay-infeasible | Rate | Progress | ADE3 | Entropy | Selection Gap Fixed |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    per_selector = gap["per_selector"]
    for name, metrics in report["selector_metrics"].items():
        fixed = per_selector.get(name, {}).get("resolved_selection_gap_count", 0)
        lines.append(
            f"| {name} | {metrics['selected_replay_infeasible_count']} | "
            f"{metrics['selected_replay_infeasible_rate']:.3f} | "
            f"{metrics['mean_selected_progress_m']:.3f} | "
            f"{_optional(metrics['mean_ade_3s_m'])} | "
            f"{metrics['selected_token_entropy']:.3f} | {fixed} |"
        )
    lines.extend(
        [
            "",
            "## Bootstrap Targets",
            "",
            f"- Target count: `{report['bootstrap_target_count']}`",
            f"- Target token histogram: `{json.dumps(report['bootstrap_target_token_histogram'], sort_keys=True)}`",
            "",
        ]
    )
    return "\n".join(lines)


def token_entropy(histogram: Mapping[str, int]) -> float:
    total = sum(int(value) for value in histogram.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in histogram.values():
        probability = float(count) / float(total)
        entropy -= probability * math.log(probability, 2)
    return round(entropy, 6)


def _rate(numerator: int, denominator: int) -> float:
    return round(float(numerator) / float(denominator), 6) if denominator else 0.0


def _mean(values: list[float] | list[int]) -> float:
    return round(sum(float(value) for value in values) / len(values), 6) if values else 0.0


def _optional(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
