#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
DEFAULT_OUTPUT = ROOT / "artifacts" / "corl2027" / "nuplan_replay_value_selector.json"
DEFAULT_REPORT_JSON = ROOT / "artifacts" / "corl2027" / "nuplan_replay_value_selector_eval.json"
DEFAULT_REPORT_MARKDOWN = ROOT / "artifacts" / "corl2027" / "nuplan_replay_value_selector_eval.md"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a token-level replay-value selector from all-token replay outcomes."
    )
    parser.add_argument("--input-replay-json", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-markdown", type=Path, default=DEFAULT_REPORT_MARKDOWN)
    parser.add_argument("--baseline-selector-model", type=Path)
    parser.add_argument("--holdout-fraction", type=float, default=0.3)
    parser.add_argument("--near-miss-threshold-m", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ridge-alpha", type=float, default=1.0e-3)
    parser.add_argument("--fail-penalty", type=float, default=250.0)
    parser.add_argument("--objective-progress-weight", type=float, default=1.0)
    parser.add_argument("--objective-ade-weight", type=float, default=2.0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    replay_report = json.loads(args.input_replay_json.read_text(encoding="utf-8"))
    train_report, holdout_report, split = split_replay_report_by_db(
        replay_report,
        holdout_fraction=float(args.holdout_fraction),
        seed=int(args.seed),
    )
    train_examples = build_replay_value_examples(
        train_report,
        near_miss_threshold_m=float(args.near_miss_threshold_m),
        fail_penalty=float(args.fail_penalty),
        objective_progress_weight=float(args.objective_progress_weight),
        objective_ade_weight=float(args.objective_ade_weight),
    )
    holdout_examples = build_replay_value_examples(
        holdout_report,
        near_miss_threshold_m=float(args.near_miss_threshold_m),
        fail_penalty=float(args.fail_penalty),
        objective_progress_weight=float(args.objective_progress_weight),
        objective_ade_weight=float(args.objective_ade_weight),
    )
    selector, train_metrics = fit_replay_value_selector(
        train_examples,
        ridge_alpha=float(args.ridge_alpha),
    )
    baseline_selector = None if args.baseline_selector_model is None else load_selector(args.baseline_selector_model)
    evaluation = {
        "schema": "nuplan_replay_value_selector_eval_v1",
        "input_replay_json": str(args.input_replay_json),
        "split": split,
        "near_miss_threshold_m": float(args.near_miss_threshold_m),
        "objective": {
            "fail_penalty": float(args.fail_penalty),
            "progress_weight": float(args.objective_progress_weight),
            "ade_weight": float(args.objective_ade_weight),
        },
        "train_fit_metrics": train_metrics,
        "train": evaluate_selector_family(
            train_report,
            replay_value_selector=selector,
            baseline_selector=baseline_selector,
            near_miss_threshold_m=float(args.near_miss_threshold_m),
        ),
        "holdout": evaluate_selector_family(
            holdout_report,
            replay_value_selector=selector,
            baseline_selector=baseline_selector,
            near_miss_threshold_m=float(args.near_miss_threshold_m),
        ),
        "train_example_count": len(train_examples),
        "holdout_example_count": len(holdout_examples),
    }
    payload = {
        **selector.to_payload(),
        "training": {
            "input_replay_json": str(args.input_replay_json),
            "split": split,
            "objective": evaluation["objective"],
            "train_fit_metrics": train_metrics,
            "train_example_count": len(train_examples),
            "holdout_example_count": len(holdout_examples),
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


def build_replay_value_examples(
    replay_report: Mapping[str, Any],
    *,
    near_miss_threshold_m: float,
    fail_penalty: float,
    objective_progress_weight: float,
    objective_ade_weight: float,
) -> list[dict[str, Any]]:
    examples = []
    for scene in replay_report.get("scenes", []):
        candidate_by_token = _candidate_by_token(scene)
        replay_by_token = _replay_by_token(scene)
        for token, candidate in candidate_by_token.items():
            replay = replay_by_token.get(str(token))
            if replay is None or replay.get("realized_min_clearance_m") is None:
                continue
            utility = candidate_logged_ego_utility(candidate, scene)
            if utility is None:
                continue
            replay_fail = float(replay["realized_min_clearance_m"]) < float(near_miss_threshold_m)
            objective_value = (
                -float(fail_penalty) * (1.0 if replay_fail else 0.0)
                + float(objective_progress_weight) * float(candidate.get("final_progress_m", 0.0))
                - float(objective_ade_weight) * float(utility["ade_3s_m"])
            )
            examples.append(
                {
                    "scene_id": str(scene["scene_id"]),
                    "source_db_file": str(scene.get("source_db_file", "")),
                    "token": str(token),
                    "objective_value": float(objective_value),
                    "features": candidate.get("features") if "features" in candidate else _features(scene, candidate),
                }
            )
    return examples


def evaluate_selector_family(
    replay_report: Mapping[str, Any],
    *,
    replay_value_selector: Any,
    baseline_selector: Any | None,
    near_miss_threshold_m: float,
) -> dict[str, Any]:
    rows = {
        "proxy_selector": evaluate_selector_policy(
            replay_report,
            selector_name="proxy_selector",
            selected_token_fn=lambda scene: str(scene["selected_token"]),
            near_miss_threshold_m=near_miss_threshold_m,
        ),
        "replay_value": evaluate_selector_policy(
            replay_report,
            selector_name="replay_value",
            selected_token_fn=lambda scene: _select_with_score_model(scene, replay_value_selector),
            near_miss_threshold_m=near_miss_threshold_m,
        ),
        "replay_oracle": evaluate_selector_policy(
            replay_report,
            selector_name="replay_oracle",
            selected_token_fn=_select_replay_oracle_token,
            near_miss_threshold_m=near_miss_threshold_m,
        ),
    }
    if baseline_selector is not None:
        rows["baseline_selector"] = evaluate_selector_policy(
            replay_report,
            selector_name="baseline_selector",
            selected_token_fn=lambda scene: _select_with_score_model(scene, baseline_selector),
            near_miss_threshold_m=near_miss_threshold_m,
        )
    return rows


def markdown_evaluation(report: Mapping[str, Any]) -> str:
    lines = [
        "# nuPlan Replay-Value Selector Evaluation",
        "",
        f"- Input replay JSON: `{report['input_replay_json']}`",
        f"- Split unit: `{report['split']['split_unit']}`",
        f"- Train scenes: `{report['split']['train_scene_count']}`",
        f"- Holdout scenes: `{report['split']['holdout_scene_count']}`",
        f"- Near-miss threshold: `{report['near_miss_threshold_m']:.2f} m`",
        (
            f"- Objective: `-{report['objective']['fail_penalty']:.1f} * replay_failure + "
            f"{report['objective']['progress_weight']:.1f} * progress - "
            f"{report['objective']['ade_weight']:.1f} * ADE3`"
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
        lines.extend(["", "### Token Histograms", ""])
        for selector_name, metrics in report[split_name].items():
            lines.append(f"- `{selector_name}`: `{json.dumps(metrics['selected_token_histogram'], sort_keys=True)}`")
        lines.append("")
    return "\n".join(lines)


def _optional_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"
