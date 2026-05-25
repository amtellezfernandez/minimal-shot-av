#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any
from typing import Mapping

import numpy as np

from minimal_shot_av.cli.commands.train_nuplan_replay_calibrated_selector import candidate_logged_ego_utility
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
DEFAULT_OUTPUT = ROOT / "artifacts" / "corl2027" / "nuplan_pairwise_border_selector.json"
DEFAULT_REPORT_JSON = ROOT / "artifacts" / "corl2027" / "nuplan_pairwise_border_selector_eval.json"
DEFAULT_REPORT_MARKDOWN = ROOT / "artifacts" / "corl2027" / "nuplan_pairwise_border_selector_eval.md"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a pairwise border reranker over top baseline candidates on replay-disagreement scenes."
    )
    parser.add_argument("--input-replay-json", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--baseline-selector-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-markdown", type=Path, default=DEFAULT_REPORT_MARKDOWN)
    parser.add_argument("--holdout-fraction", type=float, default=0.3)
    parser.add_argument("--near-miss-threshold-m", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--l2", type=float, default=1.0e-3)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--margin-threshold", type=float, default=0.75)
    parser.add_argument("--max-pairs-per-scene", type=int, default=6)
    parser.add_argument("--fail-penalty", type=float, default=250.0)
    parser.add_argument("--objective-progress-weight", type=float, default=1.0)
    parser.add_argument("--objective-ade-weight", type=float, default=2.0)
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
    examples = build_pairwise_border_examples(
        train_report,
        baseline_selector=baseline_selector,
        near_miss_threshold_m=float(args.near_miss_threshold_m),
        fail_penalty=float(args.fail_penalty),
        objective_progress_weight=float(args.objective_progress_weight),
        objective_ade_weight=float(args.objective_ade_weight),
        top_k=int(args.top_k),
        margin_threshold=float(args.margin_threshold),
        max_pairs_per_scene=int(args.max_pairs_per_scene),
    )
    policy = fit_pairwise_border_policy(
        examples,
        iterations=int(args.iterations),
        learning_rate=float(args.learning_rate),
        l2=float(args.l2),
        top_k=int(args.top_k),
        margin_threshold=float(args.margin_threshold),
    )
    evaluation = {
        "schema": "nuplan_pairwise_border_selector_eval_v1",
        "input_replay_json": str(args.input_replay_json),
        "baseline_selector_model": str(args.baseline_selector_model),
        "split": split,
        "near_miss_threshold_m": float(args.near_miss_threshold_m),
        "training": {
            "iterations": int(args.iterations),
            "learning_rate": float(args.learning_rate),
            "l2": float(args.l2),
            "top_k": int(args.top_k),
            "margin_threshold": float(args.margin_threshold),
            "max_pairs_per_scene": int(args.max_pairs_per_scene),
            "pair_count": len(examples),
            "objective": {
                "fail_penalty": float(args.fail_penalty),
                "progress_weight": float(args.objective_progress_weight),
                "ade_weight": float(args.objective_ade_weight),
            },
        },
        "train": evaluate_selector_family(
            train_report,
            baseline_selector=baseline_selector,
            pairwise_policy=policy,
            near_miss_threshold_m=float(args.near_miss_threshold_m),
        ),
        "holdout": evaluate_selector_family(
            holdout_report,
            baseline_selector=baseline_selector,
            pairwise_policy=policy,
            near_miss_threshold_m=float(args.near_miss_threshold_m),
        ),
    }
    payload = {
        **policy,
        "training": evaluation["training"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(evaluation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.report_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.report_markdown.write_text(markdown_evaluation(evaluation) + "\n", encoding="utf-8")
    print(markdown_evaluation(evaluation))
    return 0


def build_pairwise_border_examples(
    replay_report: Mapping[str, Any],
    *,
    baseline_selector: Any,
    near_miss_threshold_m: float,
    fail_penalty: float,
    objective_progress_weight: float,
    objective_ade_weight: float,
    top_k: int,
    margin_threshold: float,
    max_pairs_per_scene: int,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for scene in replay_report.get("scenes", []):
        candidate_by_token = _candidate_by_token(scene)
        replay_by_token = _replay_by_token(scene)
        ranked = _baseline_ranked_candidates(scene, baseline_selector)
        if len(ranked) < 2:
            continue
        if float(ranked[0]["score"]) - float(ranked[1]["score"]) > float(margin_threshold):
            continue
        border = ranked[: max(2, int(top_k))]
        scored = []
        for row in border:
            token = str(row["token"])
            objective = _candidate_objective(
                scene,
                candidate_by_token[token],
                replay_by_token.get(token),
                near_miss_threshold_m=near_miss_threshold_m,
                fail_penalty=fail_penalty,
                objective_progress_weight=objective_progress_weight,
                objective_ade_weight=objective_ade_weight,
            )
            if objective is None:
                continue
            scored.append((float(objective), row))
        if len(scored) < 2:
            continue
        scored.sort(reverse=True, key=lambda item: item[0])
        pair_count = 0
        for better_index, (better_score, better_row) in enumerate(scored):
            for worse_score, worse_row in scored[better_index + 1 :]:
                if better_score <= worse_score + 1.0e-9:
                    continue
                diff = _pair_features(
                    better_row["features"],
                    worse_row["features"],
                    better_row["score"],
                    worse_row["score"],
                )
                examples.append(
                    {
                        "scene_id": str(scene["scene_id"]),
                        "better_token": str(better_row["token"]),
                        "worse_token": str(worse_row["token"]),
                        "feature_diff": diff,
                    }
                )
                pair_count += 1
                if pair_count >= int(max_pairs_per_scene):
                    break
            if pair_count >= int(max_pairs_per_scene):
                break
    return examples


def fit_pairwise_border_policy(
    examples: list[Mapping[str, Any]],
    *,
    iterations: int,
    learning_rate: float,
    l2: float,
    top_k: int,
    margin_threshold: float,
) -> dict[str, Any]:
    if not examples:
        raise ValueError("cannot fit pairwise border policy on an empty example set")
    feature_dim = len(examples[0]["feature_diff"])
    x = np.asarray([example["feature_diff"] for example in examples], dtype=np.float64)
    weights = np.zeros(feature_dim, dtype=np.float64)
    pair_count = float(x.shape[0])
    for _ in range(max(1, int(iterations))):
        margins = np.clip(x @ weights, -60.0, 60.0)
        loss_weights = 1.0 / (1.0 + np.exp(margins))
        gradient = -(x.T @ loss_weights) / pair_count + float(l2) * weights
        weights -= float(learning_rate) * gradient
    return {
        "model_type": "nuplan_pairwise_border_selector_v1",
        "weights": [float(value) for value in weights],
        "feature_names": list(_pair_feature_names()),
        "top_k": int(top_k),
        "margin_threshold": float(margin_threshold),
    }


def evaluate_selector_family(
    replay_report: Mapping[str, Any],
    *,
    baseline_selector: Any,
    pairwise_policy: Mapping[str, Any],
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
        "pairwise_border": evaluate_selector_policy(
            replay_report,
            selector_name="pairwise_border",
            selected_token_fn=lambda scene: select_with_pairwise_border(scene, baseline_selector, pairwise_policy),
            near_miss_threshold_m=near_miss_threshold_m,
        ),
        "replay_oracle": evaluate_selector_policy(
            replay_report,
            selector_name="replay_oracle",
            selected_token_fn=_select_replay_oracle_token,
            near_miss_threshold_m=near_miss_threshold_m,
        ),
    }


def select_with_pairwise_border(scene: Mapping[str, Any], baseline_selector: Any, policy: Mapping[str, Any]) -> str:
    ranked = _baseline_ranked_candidates(scene, baseline_selector)
    if len(ranked) < 2:
        return str(scene.get("selected_token", ""))
    if float(ranked[0]["score"]) - float(ranked[1]["score"]) > float(policy["margin_threshold"]):
        return str(ranked[0]["token"])
    border = ranked[: max(2, int(policy["top_k"]))]
    weights = np.asarray(policy["weights"], dtype=np.float64)
    tournament = Counter()
    for left_index, left_row in enumerate(border):
        for right_row in border[left_index + 1 :]:
            diff = _pair_features(left_row["features"], right_row["features"], left_row["score"], right_row["score"])
            margin = float(np.dot(diff, weights))
            winner = left_row if margin >= 0.0 else right_row
            tournament[str(winner["token"])] += 1
    best = max(
        border,
        key=lambda row: (
            int(tournament[str(row["token"])]),
            float(row["score"]),
            float(row["clearance"]),
            float(row["progress"]),
        ),
    )
    return str(best["token"])


def _baseline_ranked_candidates(scene: Mapping[str, Any], baseline_selector: Any) -> list[dict[str, Any]]:
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


def _pair_feature_names() -> list[str]:
    return [
        "score_delta",
        "progress_delta",
        "proxy_safe_delta",
        "clearance_delta",
        "speed_scale_delta",
        "lateral_offset_delta",
        "obstacle_pressure_delta",
        "route_blockage_delta",
        "nearest_actor_distance_delta",
        "rear_closing_delta",
        "crossing_delta",
    ]


def _pair_features(
    left_features: Mapping[str, float],
    right_features: Mapping[str, float],
    left_score: float,
    right_score: float,
) -> list[float]:
    return [
        float(left_score - right_score),
        float(left_features["candidate_final_progress_m"] - right_features["candidate_final_progress_m"]),
        float(left_features["candidate_proxy_safe"] - right_features["candidate_proxy_safe"]),
        float(left_features["candidate_min_proxy_clearance_m"] - right_features["candidate_min_proxy_clearance_m"]),
        float(left_features["candidate_speed_scale"] - right_features["candidate_speed_scale"]),
        float(left_features["candidate_lateral_offset_m"] - right_features["candidate_lateral_offset_m"]),
        float(left_features["obstacle_pressure"] - right_features["obstacle_pressure"]),
        float(left_features["route_blockage"] - right_features["route_blockage"]),
        float(left_features["nearest_actor_distance_m"] - right_features["nearest_actor_distance_m"]),
        float(left_features["rear_closing_actor_count"] - right_features["rear_closing_actor_count"]),
        float(left_features["crossing_actor_count"] - right_features["crossing_actor_count"]),
    ]


def markdown_evaluation(report: Mapping[str, Any]) -> str:
    lines = [
        "# nuPlan Pairwise Border Selector Evaluation",
        "",
        f"- Input replay JSON: `{report['input_replay_json']}`",
        f"- Baseline selector: `{report['baseline_selector_model']}`",
        f"- Train scenes: `{report['split']['train_scene_count']}`",
        f"- Holdout scenes: `{report['split']['holdout_scene_count']}`",
        (
            f"- Pairwise border: `top_k={report['training']['top_k']}, "
            f"margin_threshold={report['training']['margin_threshold']:.2f}, "
            f"pairs={report['training']['pair_count']}`"
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
