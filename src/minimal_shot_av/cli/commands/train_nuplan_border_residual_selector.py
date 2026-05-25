#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any
from typing import Mapping

from minimal_shot_av.cli.commands.train_nuplan_replay_calibrated_selector import candidate_logged_ego_utility
from minimal_shot_av.cli.commands.train_nuplan_replay_calibrated_selector import evaluate_selector_policy
from minimal_shot_av.cli.commands.train_nuplan_replay_calibrated_selector import split_replay_report_by_db
from minimal_shot_av.cli.commands.train_nuplan_replay_calibrated_selector import _candidate_by_token
from minimal_shot_av.cli.commands.train_nuplan_replay_calibrated_selector import _features
from minimal_shot_av.cli.commands.train_nuplan_replay_calibrated_selector import _replay_by_token
from minimal_shot_av.cli.commands.train_nuplan_replay_calibrated_selector import _select_replay_oracle_token
from minimal_shot_av.cli.commands.train_nuplan_replay_calibrated_selector import _select_with_score_model
from minimal_shot_av.model.nuplan_maneuver_token_selector import fit_replay_value_selector
from minimal_shot_av.model.nuplan_maneuver_token_selector import load_selector


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_INPUT = ROOT / "artifacts" / "corl2027" / "nuplan_public_replay_study_interaction250" / "replay.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "corl2027" / "nuplan_border_residual_selector.json"
DEFAULT_REPORT_JSON = ROOT / "artifacts" / "corl2027" / "nuplan_border_residual_selector_eval.json"
DEFAULT_REPORT_MARKDOWN = ROOT / "artifacts" / "corl2027" / "nuplan_border_residual_selector_eval.md"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a token-level border residual selector on top of a fixed baseline selector."
    )
    parser.add_argument("--input-replay-json", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--baseline-selector-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-markdown", type=Path, default=DEFAULT_REPORT_MARKDOWN)
    parser.add_argument("--holdout-fraction", type=float, default=0.3)
    parser.add_argument("--near-miss-threshold-m", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ridge-alpha", type=float, default=1.0e-3)
    parser.add_argument("--fail-penalty", type=float, default=250.0)
    parser.add_argument("--objective-progress-weight", type=float, default=1.0)
    parser.add_argument("--objective-ade-weight", type=float, default=2.0)
    parser.add_argument("--recoverable-bonus", type=float, default=4.0)
    parser.add_argument("--margin-temperature", type=float, default=0.5)
    parser.add_argument("--residual-scale", type=float, default=0.5)
    parser.add_argument("--margin-threshold", type=float, default=0.75)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    replay_report = json.loads(args.input_replay_json.read_text(encoding="utf-8"))
    baseline_selector = load_selector(args.baseline_selector_model)
    train_report, holdout_report, split = split_replay_report_by_db(
        replay_report,
        holdout_fraction=float(args.holdout_fraction),
        seed=int(args.seed),
    )
    train_examples = build_border_residual_examples(
        train_report,
        baseline_selector=baseline_selector,
        near_miss_threshold_m=float(args.near_miss_threshold_m),
        fail_penalty=float(args.fail_penalty),
        objective_progress_weight=float(args.objective_progress_weight),
        objective_ade_weight=float(args.objective_ade_weight),
        recoverable_bonus=float(args.recoverable_bonus),
        margin_temperature=float(args.margin_temperature),
    )
    selector, fit_metrics = fit_replay_value_selector(
        train_examples,
        ridge_alpha=float(args.ridge_alpha),
    )
    evaluation = {
        "schema": "nuplan_border_residual_selector_eval_v1",
        "input_replay_json": str(args.input_replay_json),
        "baseline_selector_model": str(args.baseline_selector_model),
        "split": split,
        "near_miss_threshold_m": float(args.near_miss_threshold_m),
        "objective": {
            "fail_penalty": float(args.fail_penalty),
            "progress_weight": float(args.objective_progress_weight),
            "ade_weight": float(args.objective_ade_weight),
            "recoverable_bonus": float(args.recoverable_bonus),
            "margin_temperature": float(args.margin_temperature),
            "residual_scale": float(args.residual_scale),
            "margin_threshold": float(args.margin_threshold),
        },
        "train_fit_metrics": fit_metrics,
        "train": evaluate_selector_family(
            train_report,
            baseline_selector=baseline_selector,
            residual_selector=selector,
            near_miss_threshold_m=float(args.near_miss_threshold_m),
            residual_scale=float(args.residual_scale),
            margin_threshold=float(args.margin_threshold),
        ),
        "holdout": evaluate_selector_family(
            holdout_report,
            baseline_selector=baseline_selector,
            residual_selector=selector,
            near_miss_threshold_m=float(args.near_miss_threshold_m),
            residual_scale=float(args.residual_scale),
            margin_threshold=float(args.margin_threshold),
        ),
        "train_example_count": len(train_examples),
    }
    payload = {
        **selector.to_payload(),
        "training": {
            "input_replay_json": str(args.input_replay_json),
            "baseline_selector_model": str(args.baseline_selector_model),
            "split": split,
            "objective": evaluation["objective"],
            "train_fit_metrics": fit_metrics,
            "train_example_count": len(train_examples),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(evaluation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = markdown_evaluation(evaluation)
    args.report_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.report_markdown.write_text(markdown + "\n", encoding="utf-8")
    print(markdown)
    return 0


def build_border_residual_examples(
    replay_report: Mapping[str, Any],
    *,
    baseline_selector: Any,
    near_miss_threshold_m: float,
    fail_penalty: float,
    objective_progress_weight: float,
    objective_ade_weight: float,
    recoverable_bonus: float,
    margin_temperature: float,
) -> list[dict[str, Any]]:
    examples = []
    for scene in replay_report.get("scenes", []):
        candidate_by_token = _candidate_by_token(scene)
        replay_by_token = _replay_by_token(scene)
        baseline_rows = _baseline_scored_candidates(scene, baseline_selector)
        if not baseline_rows:
            continue
        baseline_token = str(baseline_rows[0]["token"])
        baseline_objective = _candidate_objective(
            scene,
            candidate_by_token[baseline_token],
            replay_by_token.get(baseline_token),
            near_miss_threshold_m=near_miss_threshold_m,
            fail_penalty=fail_penalty,
            objective_progress_weight=objective_progress_weight,
            objective_ade_weight=objective_ade_weight,
        )
        if baseline_objective is None:
            continue
        oracle_token = str(scene.get("oracle_log_replay_safe_token") or "")
        recoverable = bool(oracle_token) and oracle_token != baseline_token
        margin = (
            math.inf
            if len(baseline_rows) < 2
            else float(baseline_rows[0]["score"]) - float(baseline_rows[1]["score"])
        )
        border_weight = (
            1.0
            if not math.isfinite(margin)
            else 1.0 + math.exp(-margin / max(margin_temperature, 1.0e-6))
        )
        for token, candidate in candidate_by_token.items():
            objective = _candidate_objective(
                scene,
                candidate,
                replay_by_token.get(str(token)),
                near_miss_threshold_m=near_miss_threshold_m,
                fail_penalty=fail_penalty,
                objective_progress_weight=objective_progress_weight,
                objective_ade_weight=objective_ade_weight,
            )
            if objective is None:
                continue
            example_weight = border_weight
            if recoverable and str(token) == oracle_token:
                example_weight += float(recoverable_bonus)
            examples.append(
                {
                    "scene_id": str(scene["scene_id"]),
                    "source_db_file": str(scene.get("source_db_file", "")),
                    "token": str(token),
                    "objective_value": float(objective - baseline_objective),
                    "example_weight": float(example_weight),
                    "features": candidate.get("features") if "features" in candidate else _features(scene, candidate),
                }
            )
    return examples


def evaluate_selector_family(
    replay_report: Mapping[str, Any],
    *,
    baseline_selector: Any,
    residual_selector: Any,
    near_miss_threshold_m: float,
    residual_scale: float,
    margin_threshold: float,
) -> dict[str, Any]:
    return {
        "proxy_selector": evaluate_selector_policy(
            replay_report,
            selector_name="proxy_selector",
            selected_token_fn=lambda scene: str(scene["selected_token"]),
            near_miss_threshold_m=near_miss_threshold_m,
        ),
        "baseline_selector": evaluate_selector_policy(
            replay_report,
            selector_name="baseline_selector",
            selected_token_fn=lambda scene: _select_with_score_model(scene, baseline_selector),
            near_miss_threshold_m=near_miss_threshold_m,
        ),
        "border_residual": evaluate_selector_policy(
            replay_report,
            selector_name="border_residual",
            selected_token_fn=lambda scene: select_with_border_residual(
                scene,
                baseline_selector=baseline_selector,
                residual_selector=residual_selector,
                residual_scale=residual_scale,
                margin_threshold=margin_threshold,
            ),
            near_miss_threshold_m=near_miss_threshold_m,
        ),
        "replay_oracle": evaluate_selector_policy(
            replay_report,
            selector_name="replay_oracle",
            selected_token_fn=_select_replay_oracle_token,
            near_miss_threshold_m=near_miss_threshold_m,
        ),
    }


def select_with_border_residual(
    scene: Mapping[str, Any],
    *,
    baseline_selector: Any,
    residual_selector: Any,
    residual_scale: float,
    margin_threshold: float,
) -> str:
    baseline_rows = _baseline_scored_candidates(scene, baseline_selector)
    if not baseline_rows:
        return str(scene.get("selected_token", ""))
    top_score = float(baseline_rows[0]["score"])
    adjusted = []
    for row in baseline_rows:
        delta = top_score - float(row["score"])
        residual = float(residual_selector.predict_score(row["features"]))
        if delta <= float(margin_threshold):
            adjusted_score = float(row["score"]) + float(residual_scale) * residual
        else:
            adjusted_score = float(row["score"])
        adjusted.append(
            {
                **row,
                "residual": residual,
                "adjusted_score": adjusted_score,
            }
        )
    best = max(
        adjusted,
        key=lambda row: (
            float(row["adjusted_score"]),
            float(row["clearance"]),
            float(row["progress"]),
        ),
    )
    return str(best["token"])


def _baseline_scored_candidates(scene: Mapping[str, Any], baseline_selector: Any) -> list[dict[str, Any]]:
    rows = []
    for candidate in list(scene.get("candidates", [])):
        features = candidate.get("features") if "features" in candidate else _features(scene, candidate)
        rows.append(
            {
                "token": str(candidate.get("token", "")),
                "score": float(baseline_selector.predict_score(features)),
                "clearance": float(candidate.get("min_proxy_clearance_m", 0.0)),
                "progress": float(candidate.get("final_progress_m", 0.0)),
                "features": features,
            }
        )
    rows.sort(key=lambda row: (float(row["score"]), float(row["clearance"]), float(row["progress"])), reverse=True)
    return rows


def _candidate_objective(
    scene: Mapping[str, Any],
    candidate: Mapping[str, Any],
    replay: Mapping[str, Any] | None,
    *,
    near_miss_threshold_m: float,
    fail_penalty: float,
    objective_progress_weight: float,
    objective_ade_weight: float,
) -> float | None:
    if replay is None or replay.get("realized_min_clearance_m") is None:
        return None
    utility = candidate_logged_ego_utility(candidate, scene)
    if utility is None:
        return None
    replay_fail = float(replay["realized_min_clearance_m"]) < float(near_miss_threshold_m)
    return (
        -float(fail_penalty) * (1.0 if replay_fail else 0.0)
        + float(objective_progress_weight) * float(candidate.get("final_progress_m", 0.0))
        - float(objective_ade_weight) * float(utility["ade_3s_m"])
    )


def markdown_evaluation(report: Mapping[str, Any]) -> str:
    lines = [
        "# nuPlan Border Residual Selector Evaluation",
        "",
        f"- Input replay JSON: `{report['input_replay_json']}`",
        f"- Baseline selector: `{report['baseline_selector_model']}`",
        f"- Split unit: `{report['split']['split_unit']}`",
        f"- Train scenes: `{report['split']['train_scene_count']}`",
        f"- Holdout scenes: `{report['split']['holdout_scene_count']}`",
        f"- Near-miss threshold: `{report['near_miss_threshold_m']:.2f} m`",
        (
            f"- Objective: `-{report['objective']['fail_penalty']:.1f} * replay_failure + "
            f"{report['objective']['progress_weight']:.1f} * progress - "
            f"{report['objective']['ade_weight']:.1f} * ADE3`"
        ),
        (
            f"- Border residual: `recoverable_bonus={report['objective']['recoverable_bonus']:.1f}, "
            f"margin_temperature={report['objective']['margin_temperature']:.2f}, "
            f"residual_scale={report['objective']['residual_scale']:.2f}, "
            f"margin_threshold={report['objective']['margin_threshold']:.2f}`"
        ),
        "",
    ]
    for split_name in ("train", "holdout"):
        lines.extend(
            [
                f"## {split_name.title()}",
                "",
                "| Selector | Replay-infeasible | Rate | Recovery | Progress | ADE3 | FDE3 | Entropy |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for selector_name, metrics in report[split_name].items():
            lines.append(
                f"| {selector_name} | {metrics['selected_replay_infeasible_count']} | "
                f"{metrics['selected_replay_infeasible_rate']:.3f} | "
                f"{metrics['recoverable_proxy_optimism_fixed_count']}/"
                f"{metrics['recoverable_proxy_optimism_count']} | "
                f"{metrics['mean_selected_progress_m']:.3f} | "
                f"{_optional_metric(metrics['mean_ade_3s_m'])} | "
                f"{_optional_metric(metrics['mean_fde_3s_m'])} | "
                f"{metrics['selected_token_entropy']:.3f} |"
            )
        lines.append("")
    return "\n".join(lines)


def _optional_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"
