#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from statistics import mean
from statistics import pstdev
from typing import Any
from typing import Mapping

import numpy as np

from minimal_shot_av.cli.commands.train_nuplan_replay_calibrated_selector import _select_replay_oracle_token
from minimal_shot_av.cli.commands.train_nuplan_replay_calibrated_selector import candidate_logged_ego_utility
from minimal_shot_av.cli.commands.train_nuplan_replay_calibrated_selector import evaluate_selector_policy
from minimal_shot_av.cli.commands.train_nuplan_replay_calibrated_selector import select_with_constrained_replay_risk
from minimal_shot_av.cli.commands.train_nuplan_replay_calibrated_selector import split_replay_report_by_db
from minimal_shot_av.model.nuplan_maneuver_token_selector import build_replay_calibration_examples
from minimal_shot_av.model.nuplan_maneuver_token_selector import fit_replay_calibrated_selector


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_INPUT = ROOT / "artifacts" / "corl2027" / "nuplan_public_replay_study_interaction250" / "replay.json"
DEFAULT_OUTPUT_JSON = ROOT / "artifacts" / "corl2027" / "nuplan_replay_calibration_experiments.json"
DEFAULT_OUTPUT_MARKDOWN = ROOT / "artifacts" / "corl2027" / "nuplan_replay_calibration_experiments.md"
DEFAULT_OUTPUT_PARETO_CSV = ROOT / "artifacts" / "corl2027" / "nuplan_replay_calibration_experiments_pareto.csv"
DEFAULT_OUTPUT_PARETO_SVG = ROOT / "artifacts" / "corl2027" / "nuplan_replay_calibration_experiments_pareto.svg"
META_LAMBDA_FEATURE_NAMES = (
    "obstacle_pressure",
    "route_blockage",
    "corridor_blocked",
    "heading_error_abs",
    "lane_offset_abs",
    "route_remaining_m",
    "nearest_actor_distance_m",
    "leading_actor_distance_m",
    "rear_closing_actor_count",
    "crossing_actor_count",
    "candidate_count",
    "proxy_safe_candidate_count",
    "risk_min",
    "risk_mean",
    "risk_max",
    "risk_std",
    "risk_entropy",
    "proxy_selected_risk",
    "proxy_to_min_risk_margin",
    "low_risk_count_0_2",
    "low_risk_count_0_5",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DB-split sweeps and ablations for replay-calibrated ManeuverToken selection."
    )
    parser.add_argument("--input-replay-json", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_OUTPUT_MARKDOWN)
    parser.add_argument("--output-pareto-csv", type=Path, default=DEFAULT_OUTPUT_PARETO_CSV)
    parser.add_argument("--output-pareto-svg", type=Path, default=DEFAULT_OUTPUT_PARETO_SVG)
    parser.add_argument("--split-count", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--holdout-fraction", type=float, default=0.3)
    parser.add_argument("--near-miss-threshold-m", type=float, default=1.0)
    parser.add_argument("--risk-weight-sweep", default="0,5,10,20,40,80")
    parser.add_argument("--risk-threshold-sweep", default="0.05,0.1,0.2,0.3,0.4,0.5")
    parser.add_argument("--meta-lambda-values", default="0,20,40,80")
    parser.add_argument("--meta-fail-penalty", type=float, default=250.0)
    parser.add_argument("--meta-progress-weight", type=float, default=1.0)
    parser.add_argument("--meta-ade-weight", type=float, default=2.0)
    parser.add_argument("--meta-epochs", type=int, default=400)
    parser.add_argument("--meta-learning-rate", type=float, default=0.08)
    parser.add_argument("--meta-class-balanced", action="store_true")
    parser.add_argument(
        "--enable-meta-calibration",
        action="store_true",
        help=(
            "Run exploratory meta-lambda policies. Disabled by default; "
            "the CoRL path uses fixed/frontier calibration."
        ),
    )
    parser.add_argument("--main-risk-weight", type=float, default=40.0)
    parser.add_argument("--hidden-dim", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--proxy-score-weight", type=float, default=1.0)
    parser.add_argument("--progress-weight", type=float, default=0.02)
    parser.add_argument("--unsafe-penalty", type=float, default=2.0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    replay_report = json.loads(args.input_replay_json.read_text(encoding="utf-8"))
    risk_weights = _parse_float_list(str(args.risk_weight_sweep))
    report = run_replay_calibration_experiments(
        replay_report,
        input_replay_json=str(args.input_replay_json),
        split_count=max(1, int(args.split_count)),
        seed_start=int(args.seed_start),
        holdout_fraction=float(args.holdout_fraction),
        near_miss_threshold_m=float(args.near_miss_threshold_m),
        risk_weights=risk_weights,
        risk_thresholds=_parse_float_list(str(args.risk_threshold_sweep)),
        meta_lambda_values=_parse_float_list(str(args.meta_lambda_values)),
        meta_fail_penalty=float(args.meta_fail_penalty),
        meta_progress_weight=float(args.meta_progress_weight),
        meta_ade_weight=float(args.meta_ade_weight),
        meta_epochs=int(args.meta_epochs),
        meta_learning_rate=float(args.meta_learning_rate),
        meta_class_balanced=bool(args.meta_class_balanced),
        enable_meta_calibration=bool(args.enable_meta_calibration),
        main_risk_weight=float(args.main_risk_weight),
        hidden_dim=int(args.hidden_dim),
        epochs=int(args.epochs),
        learning_rate=float(args.learning_rate),
        proxy_score_weight=float(args.proxy_score_weight),
        progress_weight=float(args.progress_weight),
        unsafe_penalty=float(args.unsafe_penalty),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = markdown_report(report)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text(markdown + "\n", encoding="utf-8")
    args.output_pareto_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_pareto_csv.write_text(pareto_csv(report), encoding="utf-8")
    args.output_pareto_svg.parent.mkdir(parents=True, exist_ok=True)
    args.output_pareto_svg.write_text(pareto_svg(report), encoding="utf-8")
    print(markdown)
    return 0


def run_replay_calibration_experiments(
    replay_report: Mapping[str, Any],
    *,
    input_replay_json: str,
    split_count: int,
    seed_start: int,
    holdout_fraction: float,
    near_miss_threshold_m: float,
    risk_weights: list[float],
    risk_thresholds: list[float],
    meta_lambda_values: list[float],
    meta_fail_penalty: float,
    meta_progress_weight: float,
    meta_ade_weight: float,
    meta_epochs: int,
    meta_learning_rate: float,
    meta_class_balanced: bool,
    main_risk_weight: float,
    hidden_dim: int,
    epochs: int,
    learning_rate: float,
    proxy_score_weight: float,
    progress_weight: float,
    unsafe_penalty: float,
    enable_meta_calibration: bool = False,
) -> dict[str, Any]:
    split_results = []
    for offset in range(split_count):
        seed = seed_start + offset
        train_report, holdout_report, split = split_replay_report_by_db(
            replay_report,
            holdout_fraction=holdout_fraction,
            seed=seed,
        )
        train_examples = build_replay_calibration_examples(
            train_report,
            near_miss_threshold_m=near_miss_threshold_m,
        )
        split_result = {
            "seed": seed,
            "split": split,
            "risk_weight_sweep": [],
            "constrained_threshold_sweep": [],
            "ablations": [],
        }
        for risk_weight in risk_weights:
            selector, risk_metrics = fit_replay_calibrated_selector(
                train_examples,
                hidden_dim=hidden_dim,
                epochs=epochs,
                learning_rate=learning_rate,
                seed=seed,
                risk_weight=risk_weight,
                proxy_score_weight=proxy_score_weight,
                progress_weight=progress_weight,
                unsafe_penalty=unsafe_penalty,
                near_miss_threshold_m=near_miss_threshold_m,
            )
            split_result["risk_weight_sweep"].append(
                {
                    "risk_weight": risk_weight,
                    "risk_head_train_metrics": risk_metrics,
                    "holdout": evaluate_selector_policy(
                        holdout_report,
                        selector_name=f"lambda_{risk_weight:g}",
                        selected_token_fn=lambda scene, selector=selector: _select_with_score_model(scene, selector),
                        near_miss_threshold_m=near_miss_threshold_m,
                    ),
                }
            )
        constrained_selector, constrained_risk_metrics = fit_replay_calibrated_selector(
            train_examples,
            hidden_dim=hidden_dim,
            epochs=epochs,
            learning_rate=learning_rate,
            seed=seed,
            risk_weight=0.0,
            proxy_score_weight=proxy_score_weight,
            progress_weight=progress_weight,
            unsafe_penalty=unsafe_penalty,
            near_miss_threshold_m=near_miss_threshold_m,
        )
        for threshold in risk_thresholds:
            split_result["constrained_threshold_sweep"].append(
                {
                    "risk_threshold": threshold,
                    "risk_head_train_metrics": constrained_risk_metrics,
                    "holdout": evaluate_selector_policy(
                        holdout_report,
                        selector_name=f"constrained_risk_{threshold:g}",
                        selected_token_fn=lambda scene, selector=constrained_selector, threshold=threshold: (
                            select_with_constrained_replay_risk(
                                scene,
                                selector,
                                risk_threshold=float(threshold),
                                proxy_score_weight=proxy_score_weight,
                                progress_weight=progress_weight,
                                unsafe_penalty=unsafe_penalty,
                            )
                        ),
                        near_miss_threshold_m=near_miss_threshold_m,
                    ),
                }
            )
        if enable_meta_calibration:
            meta_policy = fit_meta_lambda_policy(
                train_report=train_report,
                risk_selector=constrained_selector,
                lambda_values=meta_lambda_values,
                near_miss_threshold_m=near_miss_threshold_m,
                fail_penalty=meta_fail_penalty,
                objective_progress_weight=meta_progress_weight,
                objective_ade_weight=meta_ade_weight,
                proxy_score_weight=proxy_score_weight,
                progress_weight=progress_weight,
                unsafe_penalty=unsafe_penalty,
                epochs=meta_epochs,
                learning_rate=meta_learning_rate,
                class_balanced=meta_class_balanced,
                seed=seed,
            )
            split_result["meta_lambda"] = {
                "policy": meta_policy,
                "holdout": evaluate_selector_policy(
                    holdout_report,
                    selector_name="meta_lambda",
                    selected_token_fn=lambda scene, selector=constrained_selector, policy=meta_policy: (
                        select_with_meta_lambda_policy(
                            scene,
                            selector,
                            policy,
                            proxy_score_weight=proxy_score_weight,
                            progress_weight=progress_weight,
                            unsafe_penalty=unsafe_penalty,
                        )
                    ),
                    near_miss_threshold_m=near_miss_threshold_m,
                ),
            }
            value_policy = fit_meta_lambda_value_policy(
                train_report=train_report,
                risk_selector=constrained_selector,
                lambda_values=meta_lambda_values,
                near_miss_threshold_m=near_miss_threshold_m,
                fail_penalty=meta_fail_penalty,
                objective_progress_weight=meta_progress_weight,
                objective_ade_weight=meta_ade_weight,
                proxy_score_weight=proxy_score_weight,
                progress_weight=progress_weight,
                unsafe_penalty=unsafe_penalty,
            )
            split_result["meta_lambda_value"] = {
                "policy": value_policy,
                "holdout": evaluate_selector_policy(
                    holdout_report,
                    selector_name="meta_lambda_value",
                    selected_token_fn=lambda scene, selector=constrained_selector, policy=value_policy: (
                        select_with_meta_lambda_value_policy(
                            scene,
                            selector,
                            policy,
                            proxy_score_weight=proxy_score_weight,
                            progress_weight=progress_weight,
                            unsafe_penalty=unsafe_penalty,
                        )
                    ),
                    near_miss_threshold_m=near_miss_threshold_m,
                ),
            }
        for config in _ablation_configs(main_risk_weight, proxy_score_weight, progress_weight, unsafe_penalty):
            if config["name"] == "proxy_selector":
                metrics = evaluate_selector_policy(
                    holdout_report,
                    selector_name="proxy_selector",
                    selected_token_fn=lambda scene: str(scene["selected_token"]),
                    near_miss_threshold_m=near_miss_threshold_m,
                )
            elif config["name"] == "replay_oracle":
                metrics = evaluate_selector_policy(
                    holdout_report,
                    selector_name="replay_oracle",
                    selected_token_fn=_select_replay_oracle_token,
                    near_miss_threshold_m=near_miss_threshold_m,
                )
            else:
                selector, _risk_metrics = fit_replay_calibrated_selector(
                    train_examples,
                    hidden_dim=hidden_dim,
                    epochs=epochs,
                    learning_rate=learning_rate,
                    seed=seed,
                    risk_weight=float(config["risk_weight"]),
                    proxy_score_weight=float(config["proxy_score_weight"]),
                    progress_weight=float(config["progress_weight"]),
                    unsafe_penalty=float(config["unsafe_penalty"]),
                    near_miss_threshold_m=near_miss_threshold_m,
                )
                metrics = evaluate_selector_policy(
                    holdout_report,
                    selector_name=str(config["name"]),
                    selected_token_fn=lambda scene, selector=selector: _select_with_score_model(scene, selector),
                    near_miss_threshold_m=near_miss_threshold_m,
                )
            split_result["ablations"].append({"config": config, "holdout": metrics})
        split_results.append(split_result)
    return {
        "schema": "nuplan_replay_calibration_experiments_v1",
        "input_replay_json": input_replay_json,
        "scene_count": int(replay_report.get("scene_count", len(replay_report.get("scenes", [])))),
        "split_count": split_count,
        "holdout_fraction": holdout_fraction,
        "near_miss_threshold_m": near_miss_threshold_m,
        "meta_calibration_enabled": bool(enable_meta_calibration),
        "meta_objective": {
            "fail_penalty": float(meta_fail_penalty),
            "progress_weight": float(meta_progress_weight),
            "ade_weight": float(meta_ade_weight),
            "lambda_values": [float(value) for value in meta_lambda_values],
        },
        "expert_distance_available": _expert_distance_available(replay_report),
        "expert_distance_note": _expert_distance_note(replay_report),
        "risk_weight_sweep_summary": _summarize_sweep(split_results),
        "constrained_threshold_sweep_summary": _summarize_constrained_thresholds(split_results),
        "meta_lambda_summary": _summarize_meta_lambda(split_results),
        "meta_lambda_value_summary": _summarize_meta_lambda_value(split_results),
        "ablation_summary": _summarize_ablations(split_results),
        "token_shift_summary": _token_shift_summary(split_results),
        "split_results": split_results,
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# nuPlan Replay Calibration Experiments",
        "",
        f"- Input replay JSON: `{report['input_replay_json']}`",
        f"- Scene count: `{report['scene_count']}`",
        f"- DB-level split count: `{report['split_count']}`",
        f"- Holdout fraction: `{report['holdout_fraction']:.2f}`",
        f"- Near-miss threshold: `{report['near_miss_threshold_m']:.2f} m`",
        f"- Expert deviation: `{report['expert_distance_note']}`",
    ]
    if report.get("meta_calibration_enabled"):
        meta = report["meta_objective"]
        lines.extend(
            [
                f"- Meta objective: `-{meta['fail_penalty']:.1f} * replay_failure "
                f"+ {meta['progress_weight']:.1f} * progress "
                f"- {meta['ade_weight']:.1f} * ADE3`",
                f"- Meta lambda values: `{json.dumps(meta['lambda_values'])}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Risk Weight Sweep",
            "",
            "| lambda | Replay-infeasible rate | Progress | ADE3 | FDE3 | Recovery rate | Token entropy |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["risk_weight_sweep_summary"]:
        lines.append(
            f"| {row['risk_weight']:.3g} | "
            f"{_mean_std(row['selected_replay_infeasible_rate'])} | "
            f"{_mean_std(row['mean_selected_progress_m'])} | "
            f"{_mean_std_optional(row['mean_ade_3s_m'])} | "
            f"{_mean_std_optional(row['mean_fde_3s_m'])} | "
            f"{_mean_std(row['recoverable_proxy_optimism_recovery_rate'])} | "
            f"{_mean_std(row['selected_token_entropy'])} |"
        )
    lines.extend(
        [
            "",
            "## Constrained Risk Threshold Sweep",
            "",
            "| risk threshold | Replay-infeasible rate | Progress | ADE3 | FDE3 | Recovery rate | Token entropy |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["constrained_threshold_sweep_summary"]:
        lines.append(
            f"| {row['risk_threshold']:.3g} | "
            f"{_mean_std(row['selected_replay_infeasible_rate'])} | "
            f"{_mean_std(row['mean_selected_progress_m'])} | "
            f"{_mean_std_optional(row['mean_ade_3s_m'])} | "
            f"{_mean_std_optional(row['mean_fde_3s_m'])} | "
            f"{_mean_std(row['recoverable_proxy_optimism_recovery_rate'])} | "
            f"{_mean_std(row['selected_token_entropy'])} |"
        )
    meta_rows = list(report.get("meta_lambda_summary", [])) + list(report.get("meta_lambda_value_summary", []))
    if meta_rows:
        lines.extend(
            [
                "",
                "## Exploratory Meta-Calibrated Lambda Policy",
                "",
                "| Model | Replay-infeasible rate | Progress | ADE3 | FDE3 | Recovery rate | Token entropy |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in meta_rows:
            lines.append(
                f"| {row['name']} | "
                f"{_mean_std(row['selected_replay_infeasible_rate'])} | "
                f"{_mean_std(row['mean_selected_progress_m'])} | "
                f"{_mean_std_optional(row['mean_ade_3s_m'])} | "
                f"{_mean_std_optional(row['mean_fde_3s_m'])} | "
                f"{_mean_std(row['recoverable_proxy_optimism_recovery_rate'])} | "
                f"{_mean_std(row['selected_token_entropy'])} |"
            )
    lines.extend(
        [
            "",
            "## Ablations",
            "",
            "| Model | Replay-infeasible rate | Progress | ADE3 | FDE3 | Recovery rate | Token entropy |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["ablation_summary"]:
        lines.append(
            f"| {row['name']} | "
            f"{_mean_std(row['selected_replay_infeasible_rate'])} | "
            f"{_mean_std(row['mean_selected_progress_m'])} | "
            f"{_mean_std_optional(row['mean_ade_3s_m'])} | "
            f"{_mean_std_optional(row['mean_fde_3s_m'])} | "
            f"{_mean_std(row['recoverable_proxy_optimism_recovery_rate'])} | "
            f"{_mean_std(row['selected_token_entropy'])} |"
        )
    lines.extend(
        [
            "",
            "## Token Shift",
            "",
            "| Model | Token histogram across holdout splits |",
            "|---|---|",
        ]
    )
    for row in report["token_shift_summary"]:
        lines.append(f"| {row['name']} | `{json.dumps(row['token_histogram'], sort_keys=True)}` |")
    return "\n".join(lines)


def pareto_csv(report: Mapping[str, Any]) -> str:
    header = [
        "lambda",
        "replay_infeasible_rate_mean",
        "replay_infeasible_rate_std",
        "progress_mean_m",
        "progress_std_m",
        "ade3_mean_m",
        "ade3_std_m",
        "fde3_mean_m",
        "fde3_std_m",
        "recovery_rate_mean",
        "recovery_rate_std",
        "token_entropy_mean",
        "token_entropy_std",
    ]
    lines = [",".join(header)]
    for row in report["risk_weight_sweep_summary"]:
        values = [
            row["risk_weight"],
            row["selected_replay_infeasible_rate"]["mean"],
            row["selected_replay_infeasible_rate"]["std"],
            row["mean_selected_progress_m"]["mean"],
            row["mean_selected_progress_m"]["std"],
            row["mean_ade_3s_m"]["mean"],
            row["mean_ade_3s_m"]["std"],
            row["mean_fde_3s_m"]["mean"],
            row["mean_fde_3s_m"]["std"],
            row["recoverable_proxy_optimism_recovery_rate"]["mean"],
            row["recoverable_proxy_optimism_recovery_rate"]["std"],
            row["selected_token_entropy"]["mean"],
            row["selected_token_entropy"]["std"],
        ]
        lines.append(",".join(_csv_cell(value) for value in values))
    return "\n".join(lines) + "\n"


def pareto_svg(report: Mapping[str, Any]) -> str:
    rows = list(report["risk_weight_sweep_summary"])
    points = [
        (
            float(row["risk_weight"]),
            float(row["mean_selected_progress_m"]["mean"]),
            float(row["selected_replay_infeasible_rate"]["mean"]),
        )
        for row in rows
        if row["mean_selected_progress_m"].get("mean") is not None
        and row["selected_replay_infeasible_rate"].get("mean") is not None
    ]
    if not points:
        return _empty_svg("No Pareto points available")
    width = 760
    height = 480
    left = 86
    right = 36
    top = 52
    bottom = 78
    x_values = [point[1] for point in points]
    y_values = [point[2] for point in points]
    x_min, x_max = _expanded_range(min(x_values), max(x_values), pad_fraction=0.08)
    y_min, y_max = _expanded_range(min(y_values), max(y_values), pad_fraction=0.12, floor=0.0, ceiling=1.0)

    def sx(progress: float) -> float:
        return left + ((progress - x_min) / (x_max - x_min)) * (width - left - right)

    def sy(rate: float) -> float:
        return top + ((y_max - rate) / (y_max - y_min)) * (height - top - bottom)

    sorted_points = sorted(points, key=lambda point: point[1])
    polyline = " ".join(f"{sx(progress):.2f},{sy(rate):.2f}" for _lam, progress, rate in sorted_points)
    x_ticks = _tick_values(x_min, x_max, 5)
    y_ticks = _tick_values(y_min, y_max, 5)
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        ".title{font:700 20px sans-serif;fill:#171717}"
        ".label{font:13px sans-serif;fill:#333}"
        ".tick{font:12px sans-serif;fill:#555}"
        ".grid{stroke:#ddd;stroke-width:1}"
        ".axis{stroke:#222;stroke-width:1.5}"
        ".line{fill:none;stroke:#0b7285;stroke-width:3}"
        ".point{fill:#0b7285;stroke:white;stroke-width:2}"
        ".note{font:12px sans-serif;fill:#555}",
        "</style>",
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#fffdf7"/>',
        f'<text class="title" x="{left}" y="30">Replay-calibrated selector safety-progress frontier</text>',
    ]
    for tick in x_ticks:
        x = sx(tick)
        elements.append(f'<line class="grid" x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{height - bottom}"/>')
        elements.append(
            f'<text class="tick" x="{x:.2f}" y="{height - bottom + 24}" '
            f'text-anchor="middle">{tick:.1f}</text>'
        )
    for tick in y_ticks:
        y = sy(tick)
        elements.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}"/>')
        elements.append(f'<text class="tick" x="{left - 10}" y="{y + 4:.2f}" text-anchor="end">{tick:.2f}</text>')
    elements.extend(
        [
            f'<line class="axis" x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}"/>',
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}"/>',
            f'<polyline class="line" points="{polyline}"/>',
        ]
    )
    for lam, progress, rate in points:
        x = sx(progress)
        y = sy(rate)
        elements.append(f'<circle class="point" cx="{x:.2f}" cy="{y:.2f}" r="6"/>')
        elements.append(f'<text class="tick" x="{x + 9:.2f}" y="{y - 9:.2f}">lambda={lam:g}</text>')
    elements.extend(
        [
            f'<text class="label" x="{(left + width - right) / 2:.2f}" y="{height - 28}" '
            'text-anchor="middle">Mean selected progress (m), higher is better</text>',
            f'<text class="label" transform="translate(24 {(top + height - bottom) / 2:.2f}) '
            'rotate(-90)" text-anchor="middle">Replay-infeasible rate, lower is better</text>',
            f'<text class="note" x="{left}" y="{height - 8}">Each point is mean over DB-level splits; '
            "CSV sidecar contains mean/std and expert-distance metrics.</text>",
            "</svg>",
        ]
    )
    return "\n".join(elements) + "\n"


def _ablation_configs(
    main_risk_weight: float,
    proxy_score_weight: float,
    progress_weight: float,
    unsafe_penalty: float,
) -> list[dict[str, Any]]:
    return [
        {"name": "proxy_selector"},
        {
            "name": "risk_only",
            "risk_weight": main_risk_weight,
            "proxy_score_weight": 0.0,
            "progress_weight": 0.0,
            "unsafe_penalty": 0.0,
        },
        {
            "name": "proxy_plus_progress",
            "risk_weight": 0.0,
            "proxy_score_weight": proxy_score_weight,
            "progress_weight": progress_weight,
            "unsafe_penalty": unsafe_penalty,
        },
        {
            "name": "proxy_plus_risk",
            "risk_weight": main_risk_weight,
            "proxy_score_weight": proxy_score_weight,
            "progress_weight": 0.0,
            "unsafe_penalty": unsafe_penalty,
        },
        {
            "name": "proxy_plus_progress_plus_risk",
            "risk_weight": main_risk_weight,
            "proxy_score_weight": proxy_score_weight,
            "progress_weight": progress_weight,
            "unsafe_penalty": unsafe_penalty,
        },
        {"name": "replay_oracle"},
    ]


def fit_meta_lambda_policy(
    *,
    train_report: Mapping[str, Any],
    risk_selector: Any,
    lambda_values: list[float],
    near_miss_threshold_m: float,
    fail_penalty: float,
    objective_progress_weight: float,
    objective_ade_weight: float,
    proxy_score_weight: float,
    progress_weight: float,
    unsafe_penalty: float,
    epochs: int,
    learning_rate: float,
    class_balanced: bool,
    seed: int,
) -> dict[str, Any]:
    if not lambda_values:
        raise ValueError("meta lambda policy requires at least one lambda value")
    examples = []
    for scene in train_report.get("scenes", []):
        label = _best_lambda_index(
            scene,
            risk_selector,
            lambda_values=lambda_values,
            near_miss_threshold_m=near_miss_threshold_m,
            fail_penalty=fail_penalty,
            objective_progress_weight=objective_progress_weight,
            objective_ade_weight=objective_ade_weight,
            proxy_score_weight=proxy_score_weight,
            progress_weight=progress_weight,
            unsafe_penalty=unsafe_penalty,
        )
        examples.append(
            {
                "features": meta_lambda_feature_row(scene, risk_selector),
                "label": label,
            }
        )
    return _fit_softmax_policy(
        examples,
        lambda_values=lambda_values,
        epochs=epochs,
        learning_rate=learning_rate,
        class_balanced=class_balanced,
        seed=seed,
    )


def select_with_meta_lambda_policy(
    scene: Mapping[str, Any],
    risk_selector: Any,
    policy: Mapping[str, Any],
    *,
    proxy_score_weight: float,
    progress_weight: float,
    unsafe_penalty: float,
) -> str:
    selected_lambda = predict_meta_lambda(scene, risk_selector, policy)
    return _select_with_lambda_risk(
        scene,
        risk_selector,
        risk_weight=selected_lambda,
        proxy_score_weight=proxy_score_weight,
        progress_weight=progress_weight,
        unsafe_penalty=unsafe_penalty,
    )


def fit_meta_lambda_value_policy(
    *,
    train_report: Mapping[str, Any],
    risk_selector: Any,
    lambda_values: list[float],
    near_miss_threshold_m: float,
    fail_penalty: float,
    objective_progress_weight: float,
    objective_ade_weight: float,
    proxy_score_weight: float,
    progress_weight: float,
    unsafe_penalty: float,
    ridge_alpha: float = 1.0e-3,
) -> dict[str, Any]:
    feature_names = tuple(META_LAMBDA_FEATURE_NAMES)
    scenes = list(train_report.get("scenes", []))
    if not scenes:
        return _empty_value_policy(feature_names=feature_names, lambda_values=lambda_values)
    x = np.asarray(
        [[float(meta_lambda_feature_row(scene, risk_selector)[name]) for name in feature_names] for scene in scenes],
        dtype=np.float64,
    )
    y = np.asarray(
        [
            [
                _lambda_objective(
                    scene,
                    risk_selector,
                    lambda_value=lambda_value,
                    near_miss_threshold_m=near_miss_threshold_m,
                    fail_penalty=fail_penalty,
                    objective_progress_weight=objective_progress_weight,
                    objective_ade_weight=objective_ade_weight,
                    proxy_score_weight=proxy_score_weight,
                    progress_weight=progress_weight,
                    unsafe_penalty=unsafe_penalty,
                )
                for lambda_value in lambda_values
            ]
            for scene in scenes
        ],
        dtype=np.float64,
    )
    feature_mean = x.mean(axis=0)
    feature_scale = x.std(axis=0)
    feature_scale[feature_scale < 1.0e-8] = 1.0
    x_norm = (x - feature_mean) / feature_scale
    design = np.concatenate([x_norm, np.ones((x_norm.shape[0], 1), dtype=np.float64)], axis=1)
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(ridge_alpha)
    penalty[-1, -1] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    predictions = design @ coefficients
    target_choice = np.argmax(y, axis=1)
    predicted_choice = np.argmax(predictions, axis=1)
    return {
        "model_type": "meta_lambda_value_ridge_v1",
        "lambda_values": [float(value) for value in lambda_values],
        "feature_names": list(feature_names),
        "feature_mean": [float(value) for value in feature_mean],
        "feature_scale": [float(value) for value in feature_scale],
        "weights": [[float(value) for value in row] for row in coefficients[:-1]],
        "bias": [float(value) for value in coefficients[-1]],
        "train_label_histogram": dict(Counter(str(lambda_values[int(index)]) for index in target_choice)),
        "train_choice_accuracy": round(float(np.mean(predicted_choice == target_choice)), 6),
        "train_mse": round(float(np.mean((predictions - y) ** 2)), 6),
    }


def select_with_meta_lambda_value_policy(
    scene: Mapping[str, Any],
    risk_selector: Any,
    policy: Mapping[str, Any],
    *,
    proxy_score_weight: float,
    progress_weight: float,
    unsafe_penalty: float,
) -> str:
    selected_lambda = predict_meta_lambda_value(scene, risk_selector, policy)
    return _select_with_lambda_risk(
        scene,
        risk_selector,
        risk_weight=selected_lambda,
        proxy_score_weight=proxy_score_weight,
        progress_weight=progress_weight,
        unsafe_penalty=unsafe_penalty,
    )


def predict_meta_lambda_value(scene: Mapping[str, Any], risk_selector: Any, policy: Mapping[str, Any]) -> float:
    features = meta_lambda_feature_row(scene, risk_selector)
    vector = np.asarray([float(features[name]) for name in policy["feature_names"]], dtype=np.float64)
    mean_vector = np.asarray(policy["feature_mean"], dtype=np.float64)
    scale_vector = np.asarray(policy["feature_scale"], dtype=np.float64)
    normalized = (vector - mean_vector) / scale_vector
    values = normalized @ np.asarray(policy["weights"], dtype=np.float64) + np.asarray(policy["bias"], dtype=np.float64)
    index = int(np.argmax(values))
    return float(policy["lambda_values"][index])


def _lambda_objective(
    scene: Mapping[str, Any],
    risk_selector: Any,
    *,
    lambda_value: float,
    near_miss_threshold_m: float,
    fail_penalty: float,
    objective_progress_weight: float,
    objective_ade_weight: float,
    proxy_score_weight: float,
    progress_weight: float,
    unsafe_penalty: float,
) -> float:
    token = _select_with_lambda_risk(
        scene,
        risk_selector,
        risk_weight=lambda_value,
        proxy_score_weight=proxy_score_weight,
        progress_weight=progress_weight,
        unsafe_penalty=unsafe_penalty,
    )
    return _token_objective(
        scene,
        token,
        near_miss_threshold_m=near_miss_threshold_m,
        fail_penalty=fail_penalty,
        objective_progress_weight=objective_progress_weight,
        objective_ade_weight=objective_ade_weight,
    )


def _empty_value_policy(*, feature_names: tuple[str, ...], lambda_values: list[float]) -> dict[str, Any]:
    return {
        "model_type": "meta_lambda_value_ridge_v1",
        "lambda_values": [float(value) for value in lambda_values],
        "feature_names": list(feature_names),
        "feature_mean": [0.0 for _ in feature_names],
        "feature_scale": [1.0 for _ in feature_names],
        "weights": [[0.0 for _ in lambda_values] for _ in feature_names],
        "bias": [0.0 for _ in lambda_values],
        "train_label_histogram": {},
        "train_choice_accuracy": 0.0,
        "train_mse": 0.0,
    }


def predict_meta_lambda(scene: Mapping[str, Any], risk_selector: Any, policy: Mapping[str, Any]) -> float:
    features = meta_lambda_feature_row(scene, risk_selector)
    vector = np.asarray([float(features[name]) for name in policy["feature_names"]], dtype=np.float64)
    mean_vector = np.asarray(policy["feature_mean"], dtype=np.float64)
    scale_vector = np.asarray(policy["feature_scale"], dtype=np.float64)
    normalized = (vector - mean_vector) / scale_vector
    logits = normalized @ np.asarray(policy["weights"], dtype=np.float64) + np.asarray(policy["bias"], dtype=np.float64)
    index = int(np.argmax(logits))
    return float(policy["lambda_values"][index])


def meta_lambda_feature_row(scene: Mapping[str, Any], risk_selector: Any) -> dict[str, float]:
    candidates = list(scene.get("candidates", []))
    risks = []
    proxy_safe_count = 0
    selected_token = str(scene.get("selected_token", ""))
    proxy_selected_risk = 1.0
    for candidate in candidates:
        features = candidate["features"] if "features" in candidate else _features(scene, candidate)
        risk = float(risk_selector.predict_risk(features))
        risks.append(risk)
        proxy_safe_count += 1 if bool(candidate.get("proxy_safe")) else 0
        if str(candidate.get("token")) == selected_token:
            proxy_selected_risk = risk
    if not risks:
        risks = [1.0]
    risk_mean = sum(risks) / len(risks)
    risk_min = min(risks)
    risk_max = max(risks)
    risk_std = math.sqrt(sum((risk - risk_mean) ** 2 for risk in risks) / len(risks))
    risk_entropy = _probability_entropy(risks)
    six = scene["six_scalar_state"]
    route = scene["route_features"]
    actors = scene["actor_summary"]
    return {
        "obstacle_pressure": float(six["obstacle_pressure"]),
        "route_blockage": float(six["route_blockage"]),
        "corridor_blocked": 1.0 if bool(six["corridor_blocked"]) else 0.0,
        "heading_error_abs": abs(float(route["heading_error_rad"])),
        "lane_offset_abs": abs(float(route["lane_offset_m"])),
        "route_remaining_m": float(route["route_remaining_m"]),
        "nearest_actor_distance_m": float(actors["nearest_actor_distance_m"]),
        "leading_actor_distance_m": float(actors["leading_actor_distance_m"]),
        "rear_closing_actor_count": float(actors["rear_closing_actor_count"]),
        "crossing_actor_count": float(actors["crossing_actor_count"]),
        "candidate_count": float(len(candidates)),
        "proxy_safe_candidate_count": float(proxy_safe_count),
        "risk_min": risk_min,
        "risk_mean": risk_mean,
        "risk_max": risk_max,
        "risk_std": risk_std,
        "risk_entropy": risk_entropy,
        "proxy_selected_risk": proxy_selected_risk,
        "proxy_to_min_risk_margin": proxy_selected_risk - risk_min,
        "low_risk_count_0_2": float(sum(1 for risk in risks if risk <= 0.2)),
        "low_risk_count_0_5": float(sum(1 for risk in risks if risk <= 0.5)),
    }


def _best_lambda_index(
    scene: Mapping[str, Any],
    risk_selector: Any,
    *,
    lambda_values: list[float],
    near_miss_threshold_m: float,
    fail_penalty: float,
    objective_progress_weight: float,
    objective_ade_weight: float,
    proxy_score_weight: float,
    progress_weight: float,
    unsafe_penalty: float,
) -> int:
    best_index = 0
    best_score = float("-inf")
    for index, lambda_value in enumerate(lambda_values):
        token = _select_with_lambda_risk(
            scene,
            risk_selector,
            risk_weight=lambda_value,
            proxy_score_weight=proxy_score_weight,
            progress_weight=progress_weight,
            unsafe_penalty=unsafe_penalty,
        )
        score = _token_objective(
            scene,
            token,
            near_miss_threshold_m=near_miss_threshold_m,
            fail_penalty=fail_penalty,
            objective_progress_weight=objective_progress_weight,
            objective_ade_weight=objective_ade_weight,
        )
        if score > best_score:
            best_score = score
            best_index = index
    return best_index


def _select_with_lambda_risk(
    scene: Mapping[str, Any],
    risk_selector: Any,
    *,
    risk_weight: float,
    proxy_score_weight: float,
    progress_weight: float,
    unsafe_penalty: float,
) -> str:
    candidates = list(scene.get("candidates", []))
    if not candidates:
        return str(scene.get("selected_token", ""))
    selected = max(
        candidates,
        key=lambda candidate: _lambda_candidate_sort_key(
            scene,
            candidate,
            risk_selector,
            risk_weight=risk_weight,
            proxy_score_weight=proxy_score_weight,
            progress_weight=progress_weight,
            unsafe_penalty=unsafe_penalty,
        ),
    )
    return str(selected["token"])


def _lambda_candidate_sort_key(
    scene: Mapping[str, Any],
    candidate: Mapping[str, Any],
    risk_selector: Any,
    *,
    risk_weight: float,
    proxy_score_weight: float,
    progress_weight: float,
    unsafe_penalty: float,
) -> tuple[float, float, float]:
    features = candidate["features"] if "features" in candidate else _features(scene, candidate)
    risk = float(risk_selector.predict_risk(features))
    proxy_safe = bool(float(features["candidate_proxy_safe"]) >= 0.5)
    score = (
        proxy_score_weight * float(features["candidate_score_heuristic"])
        + progress_weight * float(features["candidate_final_progress_m"])
        - risk_weight * risk
        - (0.0 if proxy_safe else unsafe_penalty)
    )
    return score, float(candidate.get("min_proxy_clearance_m", 0.0)), float(candidate.get("final_progress_m", 0.0))


def _token_objective(
    scene: Mapping[str, Any],
    token: str,
    *,
    near_miss_threshold_m: float,
    fail_penalty: float,
    objective_progress_weight: float,
    objective_ade_weight: float,
) -> float:
    candidate = {str(row["token"]): row for row in scene.get("candidates", [])}.get(str(token))
    replay = {str(row["token"]): row for row in scene.get("candidate_replay_evaluations", [])}.get(str(token))
    if candidate is None:
        return -float(fail_penalty)
    failed = (
        replay is None
        or replay.get("realized_min_clearance_m") is None
        or float(replay["realized_min_clearance_m"]) < near_miss_threshold_m
    )
    utility = candidate_logged_ego_utility(candidate, scene)
    ade = 0.0 if utility is None or utility.get("ade_3s_m") is None else float(utility["ade_3s_m"])
    return (
        -(float(fail_penalty) if failed else 0.0)
        + float(objective_progress_weight) * float(candidate.get("final_progress_m", 0.0))
        - float(objective_ade_weight) * ade
    )


def _fit_softmax_policy(
    examples: list[Mapping[str, Any]],
    *,
    lambda_values: list[float],
    epochs: int,
    learning_rate: float,
    class_balanced: bool,
    seed: int,
) -> dict[str, Any]:
    feature_names = tuple(META_LAMBDA_FEATURE_NAMES)
    if not examples:
        return _empty_softmax_policy(feature_names=feature_names, lambda_values=lambda_values)
    x = np.asarray(
        [[float(example["features"][name]) for name in feature_names] for example in examples],
        dtype=np.float64,
    )
    y = np.asarray([int(example["label"]) for example in examples], dtype=np.int64)
    feature_mean = x.mean(axis=0)
    feature_scale = x.std(axis=0)
    feature_scale[feature_scale < 1.0e-8] = 1.0
    x_norm = (x - feature_mean) / feature_scale
    rng = np.random.default_rng(seed)
    weights = rng.normal(0.0, 0.05, size=(x_norm.shape[1], len(lambda_values)))
    bias = np.zeros((len(lambda_values),), dtype=np.float64)
    sample_weights = np.ones((len(y), 1), dtype=np.float64)
    if class_balanced:
        counts = np.bincount(y, minlength=len(lambda_values)).astype(np.float64)
        counts[counts <= 0.0] = 1.0
        class_weights = len(y) / (len(lambda_values) * counts)
        sample_weights = class_weights[y].reshape(-1, 1)
    for _ in range(max(1, int(epochs))):
        logits = x_norm @ weights + bias
        logits -= logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        target = np.zeros_like(probs)
        target[np.arange(len(y)), y] = 1.0
        grad_logits = ((probs - target) * sample_weights) / sample_weights.sum()
        weights -= learning_rate * (x_norm.T @ grad_logits)
        bias -= learning_rate * grad_logits.sum(axis=0)
    predictions = np.argmax(x_norm @ weights + bias, axis=1)
    return {
        "model_type": "meta_lambda_softmax_v1",
        "lambda_values": [float(value) for value in lambda_values],
        "feature_names": list(feature_names),
        "feature_mean": [float(value) for value in feature_mean],
        "feature_scale": [float(value) for value in feature_scale],
        "weights": [[float(value) for value in row] for row in weights],
        "bias": [float(value) for value in bias],
        "train_label_histogram": dict(Counter(str(lambda_values[int(index)]) for index in y)),
        "train_accuracy": round(float(np.mean(predictions == y)), 6),
        "class_balanced": bool(class_balanced),
    }


def _empty_softmax_policy(*, feature_names: tuple[str, ...], lambda_values: list[float]) -> dict[str, Any]:
    return {
        "model_type": "meta_lambda_softmax_v1",
        "lambda_values": [float(value) for value in lambda_values],
        "feature_names": list(feature_names),
        "feature_mean": [0.0 for _ in feature_names],
        "feature_scale": [1.0 for _ in feature_names],
        "weights": [[0.0 for _ in lambda_values] for _ in feature_names],
        "bias": [0.0 for _ in lambda_values],
        "train_label_histogram": {},
        "train_accuracy": 0.0,
    }


def _probability_entropy(values: list[float]) -> float:
    total = sum(max(0.0, float(value)) for value in values)
    if total <= 1.0e-12:
        return 0.0
    entropy = 0.0
    for value in values:
        probability = max(0.0, float(value)) / total
        if probability > 0.0:
            entropy -= probability * math.log(probability, 2)
    return entropy


def _summarize_sweep(split_results: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[float, list[Mapping[str, Any]]] = {}
    for split in split_results:
        for row in split["risk_weight_sweep"]:
            grouped.setdefault(float(row["risk_weight"]), []).append(row["holdout"])
    return [
        {"risk_weight": risk_weight, **_metric_summary(rows)}
        for risk_weight, rows in sorted(grouped.items())
    ]


def _summarize_meta_lambda(split_results: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = [split["meta_lambda"]["holdout"] for split in split_results if split.get("meta_lambda")]
    return [{"name": "meta_lambda", **_metric_summary(rows)}] if rows else []


def _summarize_meta_lambda_value(split_results: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = [split["meta_lambda_value"]["holdout"] for split in split_results if split.get("meta_lambda_value")]
    return [{"name": "meta_lambda_value", **_metric_summary(rows)}] if rows else []


def _summarize_constrained_thresholds(split_results: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[float, list[Mapping[str, Any]]] = {}
    for split in split_results:
        for row in split["constrained_threshold_sweep"]:
            grouped.setdefault(float(row["risk_threshold"]), []).append(row["holdout"])
    return [
        {"risk_threshold": risk_threshold, **_metric_summary(rows)}
        for risk_threshold, rows in sorted(grouped.items())
    ]


def _summarize_ablations(split_results: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for split in split_results:
        for row in split["ablations"]:
            grouped.setdefault(str(row["config"]["name"]), []).append(row["holdout"])
    order = [str(row["name"]) for row in _ablation_configs(40.0, 1.0, 0.02, 2.0)]
    return [
        {"name": name, **_metric_summary(grouped[name])}
        for name in order
        if name in grouped
    ]


def _token_shift_summary(split_results: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, Counter[str]] = {}
    for split in split_results:
        for row in split["ablations"]:
            name = str(row["config"]["name"])
            grouped.setdefault(name, Counter()).update(row["holdout"]["selected_token_histogram"])
    return [{"name": name, "token_histogram": dict(sorted(hist.items()))} for name, hist in sorted(grouped.items())]


def _metric_summary(rows: list[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    keys = (
        "selected_replay_infeasible_rate",
        "mean_selected_progress_m",
        "mean_ade_3s_m",
        "mean_fde_3s_m",
        "recoverable_proxy_optimism_recovery_rate",
        "selected_token_entropy",
    )
    return {
        key: _summary_values(
            [float(row[key]) for row in rows if row.get(key) is not None],
            available_count=sum(1 for row in rows if row.get(key) is not None),
        )
        for key in keys
    }


def _summary_values(values: list[float], *, available_count: int | None = None) -> dict[str, float]:
    if not values:
        return {"mean": None, "std": None, "available_count": 0}
    return {
        "mean": round(mean(values), 6),
        "std": round(pstdev(values), 6),
        "available_count": len(values) if available_count is None else available_count,
    }


def _mean_std(row: Mapping[str, float]) -> str:
    return f"{float(row['mean']):.3f} +/- {float(row['std']):.3f}"


def _mean_std_optional(row: Mapping[str, Any]) -> str:
    if row.get("mean") is None:
        return "n/a"
    return _mean_std(row)


def _csv_cell(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _expanded_range(
    minimum: float,
    maximum: float,
    *,
    pad_fraction: float,
    floor: float | None = None,
    ceiling: float | None = None,
) -> tuple[float, float]:
    if math.isclose(minimum, maximum):
        padding = max(abs(minimum) * pad_fraction, 1.0)
    else:
        padding = (maximum - minimum) * pad_fraction
    lower = minimum - padding
    upper = maximum + padding
    if floor is not None:
        lower = max(floor, lower)
    if ceiling is not None:
        upper = min(ceiling, upper)
    if math.isclose(lower, upper):
        upper = lower + 1.0
    return lower, upper


def _tick_values(minimum: float, maximum: float, count: int) -> list[float]:
    if count <= 1:
        return [minimum]
    step = (maximum - minimum) / float(count - 1)
    return [minimum + step * index for index in range(count)]


def _empty_svg(message: str) -> str:
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="760" height="240" viewBox="0 0 760 240">\n'
        '<rect x="0" y="0" width="760" height="240" fill="#fffdf7"/>\n'
        f'<text x="40" y="120" font-family="sans-serif" font-size="18">{message}</text>\n'
        "</svg>\n"
    )


def _select_with_score_model(scene: Mapping[str, Any], selector: Any) -> str:
    candidates = list(scene.get("candidates", []))
    if not candidates:
        return str(scene.get("selected_token", ""))
    selected = max(
        candidates,
        key=lambda candidate: (
            selector.predict_score(_features(scene, candidate)),
            float(candidate.get("min_proxy_clearance_m", 0.0)),
            float(candidate.get("final_progress_m", 0.0)),
        ),
    )
    return str(selected["token"])


def _features(scene: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, float]:
    from minimal_shot_av.model.nuplan_maneuver_token_selector import selector_feature_row

    return selector_feature_row(scene, candidate)


def _parse_float_list(text: str) -> list[float]:
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("expected at least one risk weight")
    return values


def _expert_distance_available(replay_report: Mapping[str, Any]) -> bool:
    return any(scene.get("expert_trajectory") for scene in replay_report.get("scenes", []))


def _expert_distance_note(replay_report: Mapping[str, Any]) -> str:
    if _expert_distance_available(replay_report):
        return "ADE_3s/FDE_3s are computed against logged ego future trajectory preserved in replay artifacts."
    return "The replay artifact does not include expert_trajectory, so expert deviation is n/a."


if __name__ == "__main__":
    raise SystemExit(main())
