#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
from typing import Any
from typing import Mapping

from minimal_shot_av.cli.commands.run_nuplan_replay_calibration_experiments import (
    _lambda_objective,
)
from minimal_shot_av.cli.commands.run_nuplan_replay_calibration_experiments import (
    _select_with_lambda_risk,
)
from minimal_shot_av.cli.commands.run_nuplan_replay_calibration_experiments import (
    fit_meta_lambda_value_policy,
)
from minimal_shot_av.cli.commands.run_nuplan_replay_calibration_experiments import (
    predict_meta_lambda_value,
)
from minimal_shot_av.cli.commands.run_nuplan_replay_calibration_experiments import (
    select_with_meta_lambda_value_policy,
)
from minimal_shot_av.cli.commands.train_nuplan_replay_calibrated_selector import (
    build_replay_calibration_examples,
)
from minimal_shot_av.cli.commands.train_nuplan_replay_calibrated_selector import (
    evaluate_selector_policy,
)
from minimal_shot_av.cli.commands.train_nuplan_replay_calibrated_selector import (
    fit_replay_calibrated_selector,
)
from minimal_shot_av.cli.commands.train_nuplan_replay_calibrated_selector import (
    split_replay_report_by_db,
)
from minimal_shot_av.model.nuplan_maneuver_token_selector import load_selector


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_INPUT = ROOT / "artifacts" / "corl2027" / "nuplan_public_replay_study_interaction1000_expert" / "replay.json"
DEFAULT_SELECTOR = ROOT / "artifacts" / "corl2027" / "nuplan_replay_calibrated_selector_expert1000.json"
DEFAULT_OUTPUT_JSON = ROOT / "artifacts" / "corl2027" / "nuplan_meta_penalty_frontier.json"
DEFAULT_OUTPUT_MARKDOWN = ROOT / "artifacts" / "corl2027" / "nuplan_meta_penalty_frontier.md"
DEFAULT_POLICY_DIR = ROOT / "artifacts" / "corl2027"
DEFAULT_STRICT_JSON = ROOT / "artifacts" / "corl2027" / "nuplan_meta250_strict_adaptation_audit.json"
DEFAULT_STRICT_MARKDOWN = ROOT / "artifacts" / "corl2027" / "nuplan_meta250_strict_adaptation_audit.md"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze the nuPlan meta penalty frontier and write seed-0 meta-value policies."
    )
    parser.add_argument("--input-replay-json", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--selector-model", type=Path, default=DEFAULT_SELECTOR)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_OUTPUT_MARKDOWN)
    parser.add_argument("--policy-output-dir", type=Path, default=DEFAULT_POLICY_DIR)
    parser.add_argument("--strict-output-json", type=Path, default=DEFAULT_STRICT_JSON)
    parser.add_argument("--strict-output-markdown", type=Path, default=DEFAULT_STRICT_MARKDOWN)
    parser.add_argument("--penalties", default="30,60,100,150,250,500")
    parser.add_argument("--lambda-values", default="0,20,40,80")
    parser.add_argument("--split-count", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--holdout-fraction", type=float, default=0.3)
    parser.add_argument("--near-miss-threshold-m", type=float, default=1.0)
    parser.add_argument("--objective-progress-weight", type=float, default=1.0)
    parser.add_argument("--objective-ade-weight", type=float, default=2.0)
    parser.add_argument("--save-seed0-penalties", default="150,250,500")
    parser.add_argument("--strict-penalty", type=float, default=250.0)
    parser.add_argument("--risk-hidden-dim", type=int, default=24)
    parser.add_argument("--risk-epochs", type=int, default=800)
    parser.add_argument("--risk-learning-rate", type=float, default=0.04)
    parser.add_argument("--seed0-selector-output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    replay_report = json.loads(args.input_replay_json.read_text(encoding="utf-8"))
    selector = load_selector(args.selector_model)
    lambda_values = _parse_float_list(str(args.lambda_values))
    penalties = _parse_float_list(str(args.penalties))
    save_seed0_penalties = {int(value) for value in _parse_float_list(str(args.save_seed0_penalties))}
    report, strict_report = analyze_meta_penalty_frontier(
        replay_report,
        selector_model=selector,
        penalties=penalties,
        lambda_values=lambda_values,
        split_count=int(args.split_count),
        seed_start=int(args.seed_start),
        holdout_fraction=float(args.holdout_fraction),
        near_miss_threshold_m=float(args.near_miss_threshold_m),
        objective_progress_weight=float(args.objective_progress_weight),
        objective_ade_weight=float(args.objective_ade_weight),
        policy_output_dir=args.policy_output_dir,
        save_seed0_penalties=save_seed0_penalties,
        strict_penalty=float(args.strict_penalty),
        risk_hidden_dim=int(args.risk_hidden_dim),
        risk_epochs=int(args.risk_epochs),
        risk_learning_rate=float(args.risk_learning_rate),
        seed0_selector_output=args.seed0_selector_output,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text(markdown_report(report) + "\n", encoding="utf-8")
    if strict_report is not None:
        args.strict_output_json.parent.mkdir(parents=True, exist_ok=True)
        args.strict_output_json.write_text(json.dumps(strict_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        args.strict_output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.strict_output_markdown.write_text(markdown_strict_report(strict_report) + "\n", encoding="utf-8")
    print(markdown_report(report))
    return 0


def analyze_meta_penalty_frontier(
    replay_report: Mapping[str, Any],
    *,
    selector_model: Any,
    penalties: list[float],
    lambda_values: list[float],
    split_count: int,
    seed_start: int,
    holdout_fraction: float,
    near_miss_threshold_m: float,
    objective_progress_weight: float,
    objective_ade_weight: float,
    policy_output_dir: Path,
    save_seed0_penalties: set[int],
    strict_penalty: float,
    risk_hidden_dim: int,
    risk_epochs: int,
    risk_learning_rate: float,
    seed0_selector_output: Path | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    selection_kwargs_template = {
        "proxy_score_weight": float(getattr(selector_model, "proxy_score_weight", 1.0)),
        "progress_weight": float(getattr(selector_model, "progress_weight", 0.0)),
        "unsafe_penalty": float(getattr(selector_model, "unsafe_penalty", 2.0)),
    }
    strict_report = None
    penalty_rows = []
    seed0_policy_paths: dict[str, str] = {}
    seed0_selector_path: str | None = None
    for penalty in penalties:
        split_rows = []
        aggregate_choice = Counter()
        aggregate_target = Counter()
        for offset in range(split_count):
            seed = seed_start + offset
            train_report, holdout_report, split = split_replay_report_by_db(
                replay_report,
                holdout_fraction=holdout_fraction,
                seed=seed,
            )
            risk_examples = build_replay_calibration_examples(
                train_report,
                near_miss_threshold_m=near_miss_threshold_m,
            )
            split_selector_model, _risk_metrics = fit_replay_calibrated_selector(
                risk_examples,
                hidden_dim=risk_hidden_dim,
                epochs=risk_epochs,
                learning_rate=risk_learning_rate,
                seed=seed,
                risk_weight=0.0,
                near_miss_threshold_m=near_miss_threshold_m,
                **selection_kwargs_template,
            )
            if seed == seed_start and seed0_selector_output is not None and seed0_selector_path is None:
                seed0_selector_output.parent.mkdir(parents=True, exist_ok=True)
                seed0_selector_output.write_text(
                    json.dumps(split_selector_model.to_payload(), indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                seed0_selector_path = str(seed0_selector_output)
            selection_kwargs = {
                "proxy_score_weight": float(getattr(split_selector_model, "proxy_score_weight", 1.0)),
                "progress_weight": float(getattr(split_selector_model, "progress_weight", 0.0)),
                "unsafe_penalty": float(getattr(split_selector_model, "unsafe_penalty", 2.0)),
            }
            policy = fit_meta_lambda_value_policy(
                train_report=train_report,
                risk_selector=split_selector_model,
                lambda_values=lambda_values,
                near_miss_threshold_m=near_miss_threshold_m,
                fail_penalty=float(penalty),
                objective_progress_weight=objective_progress_weight,
                objective_ade_weight=objective_ade_weight,
                **selection_kwargs,
            )
            if seed == seed_start and int(round(penalty)) in save_seed0_penalties:
                path = policy_output_dir / f"nuplan_meta_value_policy_penalty{int(round(penalty))}_seed0.json"
                path.write_text(json.dumps(policy, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                seed0_policy_paths[str(int(round(penalty)))] = str(path)
            holdout = evaluate_selector_policy(
                holdout_report,
                selector_name=f"meta_value_penalty_{int(round(penalty))}",
                selected_token_fn=lambda scene, policy=policy: (
                    select_with_meta_lambda_value_policy(
                        scene,
                        split_selector_model,
                        policy,
                        **selection_kwargs,
                    )
                ),
                near_miss_threshold_m=near_miss_threshold_m,
            )
            choice_hist = Counter(
                str(predict_meta_lambda_value(scene, split_selector_model, policy))
                for scene in holdout_report.get("scenes", [])
            )
            target_hist = Counter()
            for scene in train_report.get("scenes", []):
                values = [
                    _lambda_objective(
                        scene,
                        split_selector_model,
                        lambda_value=lambda_value,
                        near_miss_threshold_m=near_miss_threshold_m,
                        fail_penalty=float(penalty),
                        objective_progress_weight=objective_progress_weight,
                        objective_ade_weight=objective_ade_weight,
                        **selection_kwargs,
                    )
                    for lambda_value in lambda_values
                ]
                best_index = max(range(len(values)), key=lambda index: values[index])
                target_hist[str(lambda_values[best_index])] += 1
            split_rows.append(
                {
                    "seed": seed,
                    "holdout_dbs": split["holdout_dbs"],
                    "scene_count": holdout["scene_count"],
                    "selected_replay_infeasible_rate": holdout["selected_replay_infeasible_rate"],
                    "mean_selected_progress_m": holdout["mean_selected_progress_m"],
                    "mean_ade_3s_m": holdout["mean_ade_3s_m"],
                    "mean_fde_3s_m": holdout["mean_fde_3s_m"],
                    "mean_heading_error_3s_rad": holdout["mean_heading_error_3s_rad"],
                    "recoverable_proxy_optimism_recovery_rate": holdout["recoverable_proxy_optimism_recovery_rate"],
                    "selected_token_entropy": holdout["selected_token_entropy"],
                    "chosen_lambda_histogram": dict(sorted(choice_hist.items())),
                    "train_target_lambda_histogram": dict(sorted(target_hist.items())),
                    "policy_train_choice_accuracy": policy["train_choice_accuracy"],
                    "policy_train_mse": policy["train_mse"],
                }
            )
            aggregate_choice.update(choice_hist)
            aggregate_target.update(target_hist)
            if seed == seed_start and abs(float(penalty) - float(strict_penalty)) < 1.0e-9:
                strict_report = _build_strict_adaptation_report(
                    holdout_report,
                    selector_model=split_selector_model,
                    policy=policy,
                    near_miss_threshold_m=near_miss_threshold_m,
                    lambda40=40.0,
                    lambda80=80.0,
                    selection_kwargs=selection_kwargs,
                    seed=seed,
                    holdout_dbs=split["holdout_dbs"],
                )
        summary = {}
        for key in (
            "selected_replay_infeasible_rate",
            "mean_selected_progress_m",
            "mean_ade_3s_m",
            "mean_fde_3s_m",
            "mean_heading_error_3s_rad",
            "recoverable_proxy_optimism_recovery_rate",
            "selected_token_entropy",
        ):
            values = [float(row[key]) for row in split_rows]
            summary[key] = {
                "mean": round(statistics.mean(values), 6),
                "std": round(statistics.pstdev(values), 6),
            }
        penalty_rows.append(
            {
                "fail_penalty": float(penalty),
                "summary": summary,
                "aggregate_chosen_lambda_histogram": dict(sorted(aggregate_choice.items())),
                "aggregate_train_target_lambda_histogram": dict(sorted(aggregate_target.items())),
                "split_rows": split_rows,
            }
        )
    return (
        {
            "schema": "nuplan_meta_penalty_frontier_v1",
            "penalties": penalty_rows,
            "seed0_policy_paths": seed0_policy_paths,
            "seed0_selector_path": seed0_selector_path,
        },
        strict_report,
    )


def _build_strict_adaptation_report(
    holdout_report: Mapping[str, Any],
    *,
    selector_model: Any,
    policy: Mapping[str, Any],
    near_miss_threshold_m: float,
    lambda40: float,
    lambda80: float,
    selection_kwargs: Mapping[str, float],
    seed: int,
    holdout_dbs: list[str],
) -> dict[str, Any]:
    groups = {
        "both_lambda40_and_lambda80_safe": [],
        "lambda40_fails_lambda80_safe": [],
        "both_lambda40_and_lambda80_fail": [],
        "lambda40_safe_lambda80_fail": [],
    }
    for scene in holdout_report.get("scenes", []):
        token40 = _select_with_lambda_risk(scene, selector_model, risk_weight=lambda40, **selection_kwargs)
        token80 = _select_with_lambda_risk(scene, selector_model, risk_weight=lambda80, **selection_kwargs)
        meta_token = select_with_meta_lambda_value_policy(scene, selector_model, policy, **selection_kwargs)
        fail40 = _token_fails(scene, token40, near_miss_threshold_m=near_miss_threshold_m)
        fail80 = _token_fails(scene, token80, near_miss_threshold_m=near_miss_threshold_m)
        fail_meta = _token_fails(scene, meta_token, near_miss_threshold_m=near_miss_threshold_m)
        row = {
            "scene_id": scene["scene_id"],
            "lambda40_token": token40,
            "lambda80_token": token80,
            "meta250_token": meta_token,
            "meta250_chosen_lambda": str(predict_meta_lambda_value(scene, selector_model, policy)),
            "lambda40_fails": fail40,
            "lambda80_fails": fail80,
            "meta250_fails": fail_meta,
        }
        if not fail40 and not fail80:
            groups["both_lambda40_and_lambda80_safe"].append(row)
        elif fail40 and not fail80:
            groups["lambda40_fails_lambda80_safe"].append(row)
        elif fail40 and fail80:
            groups["both_lambda40_and_lambda80_fail"].append(row)
        else:
            groups["lambda40_safe_lambda80_fail"].append(row)
    return {
        "schema": "nuplan_meta250_strict_adaptation_audit_v1",
        "seed": seed,
        "holdout_dbs": holdout_dbs,
        "groups": [
            {
                "group": name,
                "scene_count": len(rows),
                "meta250_failure_count": sum(1 for row in rows if row["meta250_fails"]),
                "meta250_failure_rate": (
                    round(sum(1 for row in rows if row["meta250_fails"]) / len(rows), 6) if rows else None
                ),
                "meta250_chosen_lambda_histogram": dict(
                    sorted(Counter(row["meta250_chosen_lambda"] for row in rows).items())
                ),
            }
            for name, rows in groups.items()
        ],
        "cases": groups,
    }


def _token_fails(scene: Mapping[str, Any], token: str, *, near_miss_threshold_m: float) -> bool:
    replay_by_token = {str(row["token"]): row for row in scene.get("candidate_replay_evaluations", [])}
    row = replay_by_token.get(str(token))
    return (
        row is None
        or row.get("realized_min_clearance_m") is None
        or float(row["realized_min_clearance_m"]) < near_miss_threshold_m
    )


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Meta Penalty Frontier",
        "",
        "| Penalty | Replay rate | Progress | ADE3 | FDE3 | Heading | Recovery | Token entropy | Lambda hist |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in report["penalties"]:
        summary = row["summary"]
        lines.append(
            f"| {int(round(row['fail_penalty']))} | "
            f"{summary['selected_replay_infeasible_rate']['mean']:.3f} +/- "
            f"{summary['selected_replay_infeasible_rate']['std']:.3f} | "
            f"{summary['mean_selected_progress_m']['mean']:.3f} +/- {summary['mean_selected_progress_m']['std']:.3f} | "
            f"{summary['mean_ade_3s_m']['mean']:.3f} +/- {summary['mean_ade_3s_m']['std']:.3f} | "
            f"{summary['mean_fde_3s_m']['mean']:.3f} +/- {summary['mean_fde_3s_m']['std']:.3f} | "
            f"{summary['mean_heading_error_3s_rad']['mean']:.3f} +/- "
            f"{summary['mean_heading_error_3s_rad']['std']:.3f} | "
            f"{summary['recoverable_proxy_optimism_recovery_rate']['mean']:.3f} +/- "
            f"{summary['recoverable_proxy_optimism_recovery_rate']['std']:.3f} | "
            f"{summary['selected_token_entropy']['mean']:.3f} +/- {summary['selected_token_entropy']['std']:.3f} | "
            f"`{json.dumps(row['aggregate_chosen_lambda_histogram'], sort_keys=True)}` |"
        )
    if report.get("seed0_policy_paths"):
        lines.extend(["", "## Seed-0 Policies", ""])
        for penalty, path in sorted(report["seed0_policy_paths"].items(), key=lambda item: int(item[0])):
            lines.append(f"- penalty {penalty}: `{path}`")
    if report.get("seed0_selector_path"):
        lines.append(f"- seed0 selector: `{report['seed0_selector_path']}`")
    return "\n".join(lines)


def markdown_strict_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Meta250 Strict Adaptation Audit",
        "",
        "Outcome-defined held-out groups on the seed-0 DB split.",
        "",
        "| Group | Scenes | Meta250 failures | Failure rate | Chosen lambda histogram |",
        "|---|---:|---:|---:|---|",
    ]
    for row in report["groups"]:
        rate = "n/a" if row["meta250_failure_rate"] is None else f"{row['meta250_failure_rate']:.3f}"
        lines.append(
            f"| {row['group']} | {row['scene_count']} | {row['meta250_failure_count']} | {rate} | "
            f"`{json.dumps(row['meta250_chosen_lambda_histogram'], sort_keys=True)}` |"
        )
    return "\n".join(lines)


def _parse_float_list(raw: str) -> list[float]:
    values = []
    for item in raw.split(","):
        stripped = item.strip()
        if stripped:
            values.append(float(stripped))
    return values


if __name__ == "__main__":
    raise SystemExit(main())
