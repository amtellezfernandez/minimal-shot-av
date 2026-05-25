#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any
from typing import Mapping

import numpy as np

from minimal_shot_av.cli.commands.train_nuplan_pairwise_border_selector import _baseline_ranked_candidates
from minimal_shot_av.cli.commands.train_nuplan_pairwise_border_selector import _candidate_objective
from minimal_shot_av.cli.commands.train_nuplan_replay_calibrated_selector import evaluate_selector_policy
from minimal_shot_av.cli.commands.train_nuplan_replay_calibrated_selector import split_replay_report_by_db
from minimal_shot_av.cli.commands.train_nuplan_replay_calibrated_selector import _candidate_by_token
from minimal_shot_av.cli.commands.train_nuplan_replay_calibrated_selector import _features
from minimal_shot_av.cli.commands.train_nuplan_replay_calibrated_selector import _replay_by_token
from minimal_shot_av.cli.commands.train_nuplan_replay_calibrated_selector import _select_replay_oracle_token
from minimal_shot_av.cli.commands.train_nuplan_replay_calibrated_selector import _select_with_score_model
from minimal_shot_av.model.nuplan_maneuver_token_selector import load_selector


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_INPUT = ROOT / "artifacts" / "corl2027" / "nuplan_public_replay_study_interaction250" / "replay.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "corl2027" / "nuplan_boundary_intervention_selector.json"
DEFAULT_REPORT_JSON = ROOT / "artifacts" / "corl2027" / "nuplan_boundary_intervention_selector_eval.json"
DEFAULT_REPORT_MARKDOWN = ROOT / "artifacts" / "corl2027" / "nuplan_boundary_intervention_selector_eval.md"
MODEL_TYPE = "nuplan_boundary_intervention_selector_v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a boundary-only replay-regret intervention selector anchored to a fixed baseline."
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
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--margin-threshold", type=float, default=0.75)
    parser.add_argument("--intervention-threshold", type=float, default=0.0)
    parser.add_argument("--fail-penalty", type=float, default=250.0)
    parser.add_argument("--objective-progress-weight", type=float, default=1.0)
    parser.add_argument("--objective-ade-weight", type=float, default=2.0)
    parser.add_argument("--positive-weight", type=float, default=4.0)
    parser.add_argument("--margin-temperature", type=float, default=0.5)
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
    examples = build_boundary_intervention_examples(
        train_report,
        baseline_selector=baseline_selector,
        near_miss_threshold_m=float(args.near_miss_threshold_m),
        fail_penalty=float(args.fail_penalty),
        objective_progress_weight=float(args.objective_progress_weight),
        objective_ade_weight=float(args.objective_ade_weight),
        top_k=int(args.top_k),
        margin_threshold=float(args.margin_threshold),
        positive_weight=float(args.positive_weight),
        margin_temperature=float(args.margin_temperature),
    )
    policy, fit_metrics = fit_boundary_intervention_policy(examples, ridge_alpha=float(args.ridge_alpha))
    policy["top_k"] = int(args.top_k)
    policy["margin_threshold"] = float(args.margin_threshold)
    policy["intervention_threshold"] = float(args.intervention_threshold)
    evaluation = {
        "schema": "nuplan_boundary_intervention_selector_eval_v1",
        "input_replay_json": str(args.input_replay_json),
        "baseline_selector_model": str(args.baseline_selector_model),
        "split": split,
        "near_miss_threshold_m": float(args.near_miss_threshold_m),
        "config": {
            "top_k": int(args.top_k),
            "margin_threshold": float(args.margin_threshold),
            "intervention_threshold": float(args.intervention_threshold),
            "fail_penalty": float(args.fail_penalty),
            "objective_progress_weight": float(args.objective_progress_weight),
            "objective_ade_weight": float(args.objective_ade_weight),
            "positive_weight": float(args.positive_weight),
            "margin_temperature": float(args.margin_temperature),
        },
        "train_fit_metrics": fit_metrics,
        "train_example_count": len(examples),
        "train": evaluate_selector_family(
            train_report,
            baseline_selector=baseline_selector,
            intervention_policy=policy,
            near_miss_threshold_m=float(args.near_miss_threshold_m),
        ),
        "holdout": evaluate_selector_family(
            holdout_report,
            baseline_selector=baseline_selector,
            intervention_policy=policy,
            near_miss_threshold_m=float(args.near_miss_threshold_m),
        ),
    }
    payload = {
        **policy,
        "training": {
            "input_replay_json": str(args.input_replay_json),
            "baseline_selector_model": str(args.baseline_selector_model),
            "split": split,
            "config": evaluation["config"],
            "train_fit_metrics": fit_metrics,
            "train_example_count": len(examples),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(evaluation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.report_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.report_markdown.write_text(markdown_evaluation(evaluation) + "\n", encoding="utf-8")
    print(markdown_evaluation(evaluation))
    return 0


def build_boundary_intervention_examples(
    replay_report: Mapping[str, Any],
    *,
    baseline_selector: Any,
    near_miss_threshold_m: float,
    fail_penalty: float,
    objective_progress_weight: float,
    objective_ade_weight: float,
    top_k: int,
    margin_threshold: float,
    positive_weight: float,
    margin_temperature: float,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for scene in replay_report.get("scenes", []):
        ranked = _baseline_ranked_candidates(scene, baseline_selector)
        if len(ranked) < 2:
            continue
        scene_margin = float(ranked[0]["score"]) - float(ranked[1]["score"])
        if scene_margin > float(margin_threshold):
            continue
        candidate_by_token = _candidate_by_token(scene)
        replay_by_token = _replay_by_token(scene)
        baseline_row = ranked[0]
        baseline_token = str(baseline_row["token"])
        baseline_features = _complete_features(scene, candidate_by_token[baseline_token])
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
        border_weight = 1.0 + math.exp(-scene_margin / max(float(margin_temperature), 1.0e-6))
        for alt_row in ranked[1 : max(2, int(top_k))]:
            alt_token = str(alt_row["token"])
            alt_features = _complete_features(scene, candidate_by_token[alt_token])
            alt_objective = _candidate_objective(
                scene,
                candidate_by_token[alt_token],
                replay_by_token.get(alt_token),
                near_miss_threshold_m=near_miss_threshold_m,
                fail_penalty=fail_penalty,
                objective_progress_weight=objective_progress_weight,
                objective_ade_weight=objective_ade_weight,
            )
            if alt_objective is None:
                continue
            delta = float(alt_objective - baseline_objective)
            features = _intervention_features(
                alt_features,
                baseline_features,
                alt_score=float(alt_row["score"]),
                baseline_score=float(baseline_row["score"]),
                scene_margin=scene_margin,
            )
            example_weight = border_weight * (float(positive_weight) if delta > 0.0 else 1.0)
            examples.append(
                {
                    "scene_id": str(scene["scene_id"]),
                    "baseline_token": baseline_token,
                    "alternative_token": alt_token,
                    "objective_value": delta,
                    "example_weight": float(example_weight),
                    "features": features,
                }
            )
    return examples


def fit_boundary_intervention_policy(
    examples: list[Mapping[str, Any]],
    *,
    ridge_alpha: float,
) -> tuple[dict[str, Any], dict[str, float]]:
    if not examples:
        raise ValueError("cannot fit boundary intervention selector on an empty example set")
    feature_names = tuple(_intervention_feature_names())
    x = np.asarray(
        [[float(example["features"][name]) for name in feature_names] for example in examples],
        dtype=np.float64,
    )
    y = np.asarray([float(example["objective_value"]) for example in examples], dtype=np.float64)
    weights = np.asarray([float(example.get("example_weight", 1.0)) for example in examples], dtype=np.float64)
    weights = np.clip(weights, 1.0e-6, None)
    feature_mean = x.mean(axis=0)
    feature_scale = x.std(axis=0)
    feature_scale[feature_scale < 1.0e-8] = 1.0
    x_norm = (x - feature_mean) / feature_scale
    design = np.concatenate([x_norm, np.ones((x_norm.shape[0], 1), dtype=np.float64)], axis=1)
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(ridge_alpha)
    penalty[-1, -1] = 0.0
    weighted_design = design * weights[:, None]
    coefficients = np.linalg.solve(design.T @ weighted_design + penalty, design.T @ (weights * y))
    predictions = design @ coefficients
    policy = {
        "model_type": MODEL_TYPE,
        "feature_names": list(feature_names),
        "feature_mean": [float(value) for value in feature_mean],
        "feature_scale": [float(value) for value in feature_scale],
        "weights": [float(value) for value in coefficients[:-1]],
        "bias": float(coefficients[-1]),
    }
    metrics = {
        "example_count": float(len(examples)),
        "mean_example_weight": float(weights.mean()),
        "objective_mean": float(y.mean()),
        "objective_mse": float(np.mean((predictions - y) ** 2)),
        "objective_mae": float(np.mean(np.abs(predictions - y))),
        "positive_delta_rate": float(np.mean(y > 0.0)),
    }
    return policy, metrics


def evaluate_selector_family(
    replay_report: Mapping[str, Any],
    *,
    baseline_selector: Any,
    intervention_policy: Mapping[str, Any],
    near_miss_threshold_m: float,
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
        "boundary_intervention": evaluate_selector_policy(
            replay_report,
            selector_name="boundary_intervention",
            selected_token_fn=lambda scene: select_with_boundary_intervention(
                scene,
                baseline_selector=baseline_selector,
                policy=intervention_policy,
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


def select_with_boundary_intervention(
    scene: Mapping[str, Any],
    *,
    baseline_selector: Any,
    policy: Mapping[str, Any],
) -> str:
    ranked = _baseline_ranked_candidates(scene, baseline_selector)
    if len(ranked) < 2:
        return str(scene.get("selected_token", ""))
    scene_margin = float(ranked[0]["score"]) - float(ranked[1]["score"])
    if scene_margin > float(policy["margin_threshold"]):
        return str(ranked[0]["token"])
    baseline_row = ranked[0]
    candidate_by_token = _candidate_by_token(scene)
    baseline_features = _complete_features(scene, candidate_by_token[str(baseline_row["token"])])
    best_token = str(baseline_row["token"])
    best_delta = float("-inf")
    for alt_row in ranked[1 : max(2, int(policy["top_k"]))]:
        alt_features = _complete_features(scene, candidate_by_token[str(alt_row["token"])])
        predicted_delta = predict_boundary_delta(
            _intervention_features(
                alt_features,
                baseline_features,
                alt_score=float(alt_row["score"]),
                baseline_score=float(baseline_row["score"]),
                scene_margin=scene_margin,
            ),
            policy,
        )
        if predicted_delta > best_delta:
            best_delta = predicted_delta
            best_token = str(alt_row["token"])
    if best_delta > float(policy["intervention_threshold"]):
        return best_token
    return str(baseline_row["token"])


def predict_boundary_delta(features: Mapping[str, float], policy: Mapping[str, Any]) -> float:
    feature_names = [str(name) for name in policy["feature_names"]]
    vector = np.asarray([float(features[name]) for name in feature_names], dtype=np.float64)
    mean = np.asarray(policy["feature_mean"], dtype=np.float64)
    scale = np.asarray(policy["feature_scale"], dtype=np.float64)
    normalized = (vector - mean) / scale
    return float(normalized @ np.asarray(policy["weights"], dtype=np.float64) + float(policy["bias"]))


def _intervention_feature_names() -> list[str]:
    return [
        "scene_margin",
        "score_delta",
        "progress_delta",
        "proxy_safe_delta",
        "clearance_delta",
        "speed_scale_delta",
        "lateral_offset_delta",
        "obstacle_pressure",
        "route_blockage",
        "nearest_actor_distance_m",
        "rear_closing_actor_count",
        "crossing_actor_count",
        "baseline_progress_m",
        "baseline_proxy_clearance_m",
    ]


def _complete_features(scene: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, float]:
    partial = candidate.get("features")
    if isinstance(partial, Mapping):
        base = {str(key): float(value) for key, value in partial.items()}
    else:
        base = {}
    if "six_scalar_state" in scene and "route_features" in scene and "actor_summary" in scene:
        derived = _features(scene, candidate)
        derived.update(base)
        base = derived
    base.setdefault("candidate_final_progress_m", float(candidate.get("final_progress_m", 0.0)))
    base.setdefault("candidate_proxy_safe", 1.0 if bool(candidate.get("proxy_safe")) else 0.0)
    base.setdefault("candidate_min_proxy_clearance_m", float(candidate.get("min_proxy_clearance_m", 0.0)))
    base.setdefault("candidate_score_heuristic", float(candidate.get("score", 0.0)))
    base.setdefault("candidate_speed_scale", float(candidate.get("speed_scale", 0.0)))
    base.setdefault("candidate_lateral_offset_m", float(candidate.get("lateral_offset_m", 0.0)))
    base.setdefault("obstacle_pressure", 0.0)
    base.setdefault("route_blockage", 0.0)
    base.setdefault("nearest_actor_distance_m", 0.0)
    base.setdefault("rear_closing_actor_count", 0.0)
    base.setdefault("crossing_actor_count", 0.0)
    return base


def _intervention_features(
    alternative_features: Mapping[str, float],
    baseline_features: Mapping[str, float],
    *,
    alt_score: float,
    baseline_score: float,
    scene_margin: float,
) -> dict[str, float]:
    return {
        "scene_margin": float(scene_margin),
        "score_delta": float(alt_score - baseline_score),
        "progress_delta": float(
            alternative_features["candidate_final_progress_m"] - baseline_features["candidate_final_progress_m"]
        ),
        "proxy_safe_delta": float(
            alternative_features["candidate_proxy_safe"] - baseline_features["candidate_proxy_safe"]
        ),
        "clearance_delta": float(
            alternative_features["candidate_min_proxy_clearance_m"]
            - baseline_features["candidate_min_proxy_clearance_m"]
        ),
        "speed_scale_delta": float(
            alternative_features["candidate_speed_scale"] - baseline_features["candidate_speed_scale"]
        ),
        "lateral_offset_delta": float(
            alternative_features["candidate_lateral_offset_m"] - baseline_features["candidate_lateral_offset_m"]
        ),
        "obstacle_pressure": float(baseline_features["obstacle_pressure"]),
        "route_blockage": float(baseline_features["route_blockage"]),
        "nearest_actor_distance_m": float(baseline_features["nearest_actor_distance_m"]),
        "rear_closing_actor_count": float(baseline_features["rear_closing_actor_count"]),
        "crossing_actor_count": float(baseline_features["crossing_actor_count"]),
        "baseline_progress_m": float(baseline_features["candidate_final_progress_m"]),
        "baseline_proxy_clearance_m": float(baseline_features["candidate_min_proxy_clearance_m"]),
    }


def markdown_evaluation(report: Mapping[str, Any]) -> str:
    lines = [
        "# nuPlan Boundary Intervention Selector Evaluation",
        "",
        f"- Input replay JSON: `{report['input_replay_json']}`",
        f"- Baseline selector: `{report['baseline_selector_model']}`",
        f"- Train scenes: `{report['split']['train_scene_count']}`",
        f"- Holdout scenes: `{report['split']['holdout_scene_count']}`",
        (
            f"- Boundary intervention: `top_k={report['config']['top_k']}, "
            f"margin_threshold={report['config']['margin_threshold']:.2f}, "
            f"intervention_threshold={report['config']['intervention_threshold']:.2f}`"
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


def load_boundary_intervention_policy(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if str(payload.get("model_type", "")) != MODEL_TYPE:
        raise ValueError(f"not a boundary intervention selector artifact: {path}")
    return dict(payload)
