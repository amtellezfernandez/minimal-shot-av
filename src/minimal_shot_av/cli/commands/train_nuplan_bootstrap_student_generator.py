#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Mapping

import numpy as np

from minimal_shot_av.cli.commands.run_nuplan_bootstrap_candidate_loop import (
    load_teacher_targets,
)
from minimal_shot_av.cli.commands.run_nuplan_bootstrap_candidate_loop import _scene_key
from minimal_shot_av.cli.commands.run_nuplan_bootstrap_candidate_loop import _target_key
from minimal_shot_av.cli.commands.train_nuplan_replay_calibrated_selector import _select_replay_oracle_token
from minimal_shot_av.cli.commands.train_nuplan_replay_calibrated_selector import _select_with_score_model
from minimal_shot_av.cli.commands.train_nuplan_replay_calibrated_selector import _token_is_replay_safe
from minimal_shot_av.cli.commands.train_nuplan_replay_calibrated_selector import evaluate_selector_policy
from minimal_shot_av.cli.commands.train_nuplan_replay_calibrated_selector import split_replay_report_by_db
from minimal_shot_av.model.nuplan_maneuver_token_adapter import TOKEN_ORDER
from minimal_shot_av.model.nuplan_maneuver_token_selector import fit_selector_mlp
from minimal_shot_av.model.nuplan_maneuver_token_selector import selector_feature_row


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_INPUT = ROOT / "artifacts" / "corl2027" / "nuplan_public_replay_study_interaction250" / "replay.json"
DEFAULT_TARGETS = ROOT / "artifacts" / "corl2027" / "nuplan_bootstrap_teacher_targets.jsonl"
DEFAULT_OUTPUT = ROOT / "artifacts" / "corl2027" / "nuplan_bootstrap_student_generator.json"
DEFAULT_REPORT_JSON = ROOT / "artifacts" / "corl2027" / "nuplan_bootstrap_student_generator_eval.json"
DEFAULT_REPORT_MD = ROOT / "artifacts" / "corl2027" / "nuplan_bootstrap_student_generator_eval.md"
SCENE_TOKEN_STUDENT_MODEL_TYPE = "nuplan_bootstrap_scene_token_student_mlp_v1"

SCENE_BASE_FEATURE_NAMES = (
    "obstacle_pressure",
    "route_blockage",
    "corridor_blocked",
    "left_clearance_m",
    "right_clearance_m",
    "escape_side_float",
    "heading_error_rad",
    "lane_offset_m",
    "route_remaining_m",
    "nearest_actor_distance_m",
    "leading_actor_distance_m",
    "rear_closing_actor_count",
    "crossing_actor_count",
    "candidate_proxy_safe_count",
    "candidate_proxy_safe_rate",
    "max_candidate_clearance_m",
    "mean_candidate_clearance_m",
    "max_candidate_progress_m",
    "mean_candidate_progress_m",
    "max_candidate_score",
    "mean_candidate_score",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a G1 ManeuverToken student generator from bootstrap teacher targets."
    )
    parser.add_argument("--input-replay-json", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--teacher-targets-jsonl", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-markdown", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument("--near-miss-threshold-m", type=float, default=1.0)
    parser.add_argument("--holdout-fraction", type=float, default=0.3)
    parser.add_argument(
        "--student-objective",
        choices=("candidate_binary", "scene_token_mlp"),
        default="candidate_binary",
        help="candidate_binary reproduces the first G1 student; scene_token_mlp learns a scene-level token policy.",
    )
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    replay_report = json.loads(args.input_replay_json.read_text(encoding="utf-8"))
    targets = load_teacher_targets(args.teacher_targets_jsonl)
    train_report, holdout_report, split = split_replay_report_by_db(
        replay_report,
        holdout_fraction=float(args.holdout_fraction),
        seed=int(args.seed),
    )
    train_examples = build_bootstrap_student_examples(train_report, targets)
    holdout_examples = build_bootstrap_student_examples(holdout_report, targets)
    teacher_fn = teacher_target_selector(targets)
    if args.student_objective == "scene_token_mlp":
        selector, training_metrics = fit_scene_token_student(
            train_report,
            teacher_token_fn=teacher_fn,
            hidden_dim=int(args.hidden_dim),
            epochs=int(args.epochs),
            learning_rate=float(args.learning_rate),
            seed=int(args.seed),
        )
        student_fn = lambda scene, model=selector: predict_scene_token_student(model, scene)
        payload = {
            **scene_token_student_payload(selector),
            "training": {
                "input_replay_json": str(args.input_replay_json),
                "teacher_targets_jsonl": str(args.teacher_targets_jsonl),
                "student_objective": args.student_objective,
                "split": split,
                "training_metrics": training_metrics,
                "train_example_count": len(train_examples),
                "holdout_example_count": len(holdout_examples),
            },
        }
    else:
        selector, training_metrics = fit_selector_mlp(
            train_examples,
            hidden_dim=int(args.hidden_dim),
            epochs=int(args.epochs),
            learning_rate=float(args.learning_rate),
            seed=int(args.seed),
        )
        student_fn = lambda scene, model=selector: _select_with_score_model(scene, model)
        payload = {
            **selector.to_payload(),
            "training": {
                "input_replay_json": str(args.input_replay_json),
                "teacher_targets_jsonl": str(args.teacher_targets_jsonl),
                "student_objective": args.student_objective,
                "split": split,
                "training_metrics": training_metrics,
                "train_example_count": len(train_examples),
                "holdout_example_count": len(holdout_examples),
            },
        }
    report = {
        "schema": "nuplan_bootstrap_student_generator_eval_v1",
        "input_replay_json": str(args.input_replay_json),
        "teacher_targets_jsonl": str(args.teacher_targets_jsonl),
        "student_objective": args.student_objective,
        "near_miss_threshold_m": float(args.near_miss_threshold_m),
        "split": split,
        "training_metrics": training_metrics,
        "train_example_count": len(train_examples),
        "holdout_example_count": len(holdout_examples),
        "train": evaluate_student_family(
            train_report,
            student_token_fn=student_fn,
            targets=targets,
            near_miss_threshold_m=float(args.near_miss_threshold_m),
        ),
        "holdout": evaluate_student_family(
            holdout_report,
            student_token_fn=student_fn,
            targets=targets,
            near_miss_threshold_m=float(args.near_miss_threshold_m),
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.report_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.report_markdown.write_text(markdown_student_report(report) + "\n", encoding="utf-8")
    print(markdown_student_report(report))
    return 0


def build_bootstrap_student_examples(
    replay_report: Mapping[str, Any],
    targets: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    target_fn = teacher_target_selector(targets)
    examples = []
    for scene in replay_report.get("scenes", []):
        teacher_token = target_fn(scene)
        candidates = list(scene.get("candidates", []))
        positive_weight = max(1.0, float(len(candidates) - 1))
        for candidate in candidates:
            token = str(candidate.get("token", ""))
            is_target = token == teacher_token
            examples.append(
                {
                    "scene_id": str(scene.get("scene_id", "")),
                    "source_db_file": str(scene.get("source_db_file", "")),
                    "token": token,
                    "label": 1.0 if is_target else 0.0,
                    "example_weight": positive_weight if is_target else 1.0,
                    "supervision_score": 1.0 if is_target else 0.0,
                    "features": selector_feature_row(scene, candidate),
                }
            )
    if not examples:
        raise ValueError("cannot train bootstrap student on an empty replay report")
    return examples


def evaluate_student_family(
    replay_report: Mapping[str, Any],
    *,
    student_token_fn: Callable[[Mapping[str, Any]], str],
    targets: list[Mapping[str, Any]],
    near_miss_threshold_m: float,
) -> dict[str, Any]:
    teacher_fn = teacher_target_selector(targets)
    selectors = {
        "g0_proxy_top1": lambda scene: str(scene.get("selected_token", "")),
        "g1_student": student_token_fn,
        "teacher_promoted_upper_bound": teacher_fn,
        "replay_oracle": _select_replay_oracle_token,
    }
    rows = {}
    for name, fn in selectors.items():
        rows[name] = {
            **evaluate_selector_policy(
                replay_report,
                selector_name=name,
                selected_token_fn=fn,
                near_miss_threshold_m=near_miss_threshold_m,
            ),
            "gap_summary": top1_gap_summary(
                replay_report,
                selected_token_fn=fn,
                near_miss_threshold_m=near_miss_threshold_m,
            ),
            "teacher_agreement": teacher_agreement_summary(
                replay_report,
                selected_token_fn=fn,
                teacher_token_fn=teacher_fn,
            ),
        }
    return rows


def teacher_target_selector(targets: list[Mapping[str, Any]]) -> Callable[[Mapping[str, Any]], str]:
    target_by_key = {_target_key(target): target for target in targets}
    target_by_scene_id = {str(target.get("scene_id", "")): target for target in targets}

    def select(scene: Mapping[str, Any]) -> str:
        target = target_by_key.get(_scene_key(scene)) or target_by_scene_id.get(str(scene.get("scene_id", "")))
        if target is None:
            return str(scene.get("selected_token", ""))
        return str(target.get("teacher_token", scene.get("selected_token", "")))

    return select


def fit_scene_token_student(
    replay_report: Mapping[str, Any],
    *,
    teacher_token_fn: Callable[[Mapping[str, Any]], str],
    hidden_dim: int,
    epochs: int,
    learning_rate: float,
    seed: int,
    token_order: tuple[str, ...] | None = None,
) -> tuple[dict[str, Any], dict[str, float]]:
    scenes = list(replay_report.get("scenes", []))
    if not scenes:
        raise ValueError("cannot train scene-token student on an empty replay report")
    resolved_token_order = token_order or candidate_token_order_from_report(replay_report)
    feature_names = scene_feature_names(resolved_token_order)
    x = np.asarray(
        [[float(scene_feature_row(scene, token_order=resolved_token_order)[name]) for name in feature_names] for scene in scenes],
        dtype=np.float64,
    )
    y = np.asarray([resolved_token_order.index(str(teacher_token_fn(scene))) for scene in scenes], dtype=np.int64)
    class_counts = np.bincount(y, minlength=len(resolved_token_order)).astype(np.float64)
    class_weights = np.zeros_like(class_counts)
    nonzero = class_counts > 0.0
    class_weights[nonzero] = float(len(scenes)) / (float(nonzero.sum()) * class_counts[nonzero])
    sample_weights = class_weights[y]
    sample_weights = sample_weights / max(1.0e-6, float(sample_weights.mean()))
    fitted = _fit_multiclass_mlp(
        x=x,
        y=y,
        sample_weights=sample_weights,
        hidden_dim=hidden_dim,
        epochs=epochs,
        learning_rate=learning_rate,
        seed=seed,
        class_count=len(resolved_token_order),
    )
    model = {
        "model_type": SCENE_TOKEN_STUDENT_MODEL_TYPE,
        "feature_names": list(feature_names),
        "token_order": list(resolved_token_order),
        "hidden_dim": int(hidden_dim),
        **fitted,
    }
    predictions = [predict_scene_token_student(model, scene) for scene in scenes]
    accuracy = sum(1 for pred, label in zip(predictions, y) if pred == resolved_token_order[int(label)]) / len(scenes)
    probabilities = _predict_multiclass_probabilities(model, x)
    losses = -np.log(np.clip(probabilities[np.arange(len(y)), y], 1.0e-8, 1.0))
    return model, {
        "scene_count": float(len(scenes)),
        "example_count": float(len(scenes)),
        "scene_accuracy": float(accuracy),
        "example_log_loss": float(np.mean(losses)),
    }


def predict_scene_token_student(model: Mapping[str, Any], scene: Mapping[str, Any]) -> str:
    token_order = tuple(str(token) for token in model["token_order"])
    features = scene_feature_row(scene, token_order=token_order)
    feature_names = tuple(str(name) for name in model["feature_names"])
    vector = np.asarray([[float(features[name]) for name in feature_names]], dtype=np.float64)
    probabilities = _predict_multiclass_probabilities(model, vector)[0]
    candidate_tokens = {str(candidate.get("token", "")) for candidate in scene.get("candidates", [])}
    ranked = sorted(
        enumerate(probabilities),
        key=lambda item: (float(item[1]), -item[0]),
        reverse=True,
    )
    for index, _probability in ranked:
        token = token_order[int(index)]
        if token in candidate_tokens:
            return token
    return str(scene.get("selected_token", ""))


def scene_token_student_payload(model: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "model_type": SCENE_TOKEN_STUDENT_MODEL_TYPE,
        "feature_names": list(model["feature_names"]),
        "token_order": list(model["token_order"]),
        "hidden_dim": int(model["hidden_dim"]),
        "feature_mean": list(model["feature_mean"]),
        "feature_scale": list(model["feature_scale"]),
        "w1": [list(row) for row in model["w1"]],
        "b1": list(model["b1"]),
        "w2": [list(row) for row in model["w2"]],
        "b2": list(model["b2"]),
    }


def scene_feature_names(token_order: tuple[str, ...] | None = None) -> tuple[str, ...]:
    resolved_token_order = token_order or tuple(TOKEN_ORDER)
    names = list(SCENE_BASE_FEATURE_NAMES)
    for token in resolved_token_order:
        names.extend(
            [
                f"selected_token_is_{token}",
                f"{token}_available",
                f"{token}_proxy_safe",
                f"{token}_min_proxy_clearance_m",
                f"{token}_final_progress_m",
                f"{token}_score",
            ]
    )
    return tuple(names)


def scene_feature_row(
    scene: Mapping[str, Any],
    *,
    token_order: tuple[str, ...] | None = None,
) -> dict[str, float]:
    resolved_token_order = token_order or tuple(TOKEN_ORDER)
    six_scalar_state = scene["six_scalar_state"]
    route = scene["route_features"]
    actor_summary = scene["actor_summary"]
    candidates = list(scene.get("candidates", []))
    candidate_by_token = {str(candidate.get("token", "")): candidate for candidate in candidates}
    clearances = [float(candidate.get("min_proxy_clearance_m", 0.0)) for candidate in candidates]
    progress_values = [float(candidate.get("final_progress_m", 0.0)) for candidate in candidates]
    score_values = [float(candidate.get("score", 0.0)) for candidate in candidates]
    proxy_safe_count = sum(1 for candidate in candidates if bool(candidate.get("proxy_safe", False)))
    row = {
        "obstacle_pressure": float(six_scalar_state["obstacle_pressure"]),
        "route_blockage": float(six_scalar_state["route_blockage"]),
        "corridor_blocked": 1.0 if bool(six_scalar_state["corridor_blocked"]) else 0.0,
        "left_clearance_m": float(six_scalar_state["left_clearance_m"]),
        "right_clearance_m": float(six_scalar_state["right_clearance_m"]),
        "escape_side_float": float(six_scalar_state["vector"][5]),
        "heading_error_rad": float(route["heading_error_rad"]),
        "lane_offset_m": float(route["lane_offset_m"]),
        "route_remaining_m": float(route["route_remaining_m"]),
        "nearest_actor_distance_m": float(actor_summary["nearest_actor_distance_m"]),
        "leading_actor_distance_m": float(actor_summary["leading_actor_distance_m"]),
        "rear_closing_actor_count": float(actor_summary["rear_closing_actor_count"]),
        "crossing_actor_count": float(actor_summary["crossing_actor_count"]),
        "candidate_proxy_safe_count": float(proxy_safe_count),
        "candidate_proxy_safe_rate": float(proxy_safe_count) / max(1.0, float(len(candidates))),
        "max_candidate_clearance_m": max(clearances) if clearances else 0.0,
        "mean_candidate_clearance_m": _mean(clearances),
        "max_candidate_progress_m": max(progress_values) if progress_values else 0.0,
        "mean_candidate_progress_m": _mean(progress_values),
        "max_candidate_score": max(score_values) if score_values else 0.0,
        "mean_candidate_score": _mean(score_values),
    }
    selected_token = str(scene.get("selected_token", ""))
    for token in resolved_token_order:
        candidate = candidate_by_token.get(token)
        row[f"selected_token_is_{token}"] = 1.0 if selected_token == token else 0.0
        row[f"{token}_available"] = 1.0 if candidate is not None else 0.0
        row[f"{token}_proxy_safe"] = 1.0 if candidate is not None and bool(candidate.get("proxy_safe")) else 0.0
        row[f"{token}_min_proxy_clearance_m"] = (
            float(candidate.get("min_proxy_clearance_m", 0.0)) if candidate is not None else 0.0
        )
        row[f"{token}_final_progress_m"] = (
            float(candidate.get("final_progress_m", 0.0)) if candidate is not None else 0.0
        )
        row[f"{token}_score"] = float(candidate.get("score", 0.0)) if candidate is not None else 0.0
    return row


def _fit_multiclass_mlp(
    *,
    x: np.ndarray,
    y: np.ndarray,
    sample_weights: np.ndarray,
    hidden_dim: int,
    epochs: int,
    learning_rate: float,
    seed: int,
    class_count: int,
) -> dict[str, Any]:
    feature_mean = x.mean(axis=0)
    feature_scale = x.std(axis=0)
    feature_scale[feature_scale < 1.0e-8] = 1.0
    x_norm = (x - feature_mean) / feature_scale
    rng = np.random.default_rng(seed)
    w1 = rng.normal(0.0, 0.12, size=(x_norm.shape[1], hidden_dim))
    b1 = np.zeros((hidden_dim,), dtype=np.float64)
    w2 = rng.normal(0.0, 0.12, size=(hidden_dim, class_count))
    b2 = np.zeros((class_count,), dtype=np.float64)
    y_onehot = np.eye(class_count, dtype=np.float64)[y]
    weights = sample_weights.reshape(-1, 1)
    for _ in range(max(1, epochs)):
        z1 = x_norm @ w1 + b1
        h1 = np.maximum(z1, 0.0)
        logits = h1 @ w2 + b2
        probs = _softmax(logits)
        grad_logits = weights * (probs - y_onehot) / x_norm.shape[0]
        grad_w2 = h1.T @ grad_logits
        grad_b2 = grad_logits.sum(axis=0)
        grad_h1 = grad_logits @ w2.T
        grad_z1 = grad_h1 * (z1 > 0.0)
        grad_w1 = x_norm.T @ grad_z1
        grad_b1 = grad_z1.sum(axis=0)
        w1 -= learning_rate * grad_w1
        b1 -= learning_rate * grad_b1
        w2 -= learning_rate * grad_w2
        b2 -= learning_rate * grad_b2
    return {
        "feature_mean": tuple(float(value) for value in feature_mean),
        "feature_scale": tuple(float(value) for value in feature_scale),
        "w1": tuple(tuple(float(value) for value in row) for row in w1),
        "b1": tuple(float(value) for value in b1),
        "w2": tuple(tuple(float(value) for value in row) for row in w2),
        "b2": tuple(float(value) for value in b2),
    }


def _predict_multiclass_probabilities(model: Mapping[str, Any], x: np.ndarray) -> np.ndarray:
    mean = np.asarray(model["feature_mean"], dtype=np.float64)
    scale = np.asarray(model["feature_scale"], dtype=np.float64)
    normalized = (x - mean) / scale
    hidden = np.maximum(
        normalized @ np.asarray(model["w1"], dtype=np.float64) + np.asarray(model["b1"], dtype=np.float64),
        0.0,
    )
    logits = hidden @ np.asarray(model["w2"], dtype=np.float64) + np.asarray(model["b2"], dtype=np.float64)
    return _softmax(logits)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(np.clip(shifted, -40.0, 40.0))
    return exp_values / np.clip(exp_values.sum(axis=1, keepdims=True), 1.0e-8, None)


def candidate_token_order_from_report(replay_report: Mapping[str, Any]) -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set()
    for scene in replay_report.get("scenes", []):
        for candidate in scene.get("candidates", []):
            token = str(candidate.get("token", ""))
            if token and token not in seen:
                ordered.append(token)
                seen.add(token)
    for token in TOKEN_ORDER:
        if token not in seen:
            ordered.append(token)
            seen.add(token)
    return tuple(ordered)


def top1_gap_summary(
    replay_report: Mapping[str, Any],
    *,
    selected_token_fn: Callable[[Mapping[str, Any]], str],
    near_miss_threshold_m: float,
) -> dict[str, Any]:
    scenes = list(replay_report.get("scenes", []))
    generation_gap = 0
    top1_safe = 0
    selection_gap = 0
    for scene in scenes:
        oracle_token = scene.get("oracle_log_replay_safe_token")
        oracle_safe = oracle_token is not None and _token_is_replay_safe(
            scene,
            str(oracle_token),
            near_miss_threshold_m,
        )
        if not oracle_safe:
            generation_gap += 1
        selected_safe = _token_is_replay_safe(scene, str(selected_token_fn(scene)), near_miss_threshold_m)
        if selected_safe:
            top1_safe += 1
        if oracle_safe and not selected_safe:
            selection_gap += 1
    scene_count = len(scenes)
    return {
        "scene_count": scene_count,
        "generation_gap_count": generation_gap,
        "generation_gap_rate": _rate(generation_gap, scene_count),
        "top1_safe_count": top1_safe,
        "top1_safe_rate": _rate(top1_safe, scene_count),
        "selection_gap_count": selection_gap,
        "selection_gap_rate": _rate(selection_gap, scene_count),
    }


def teacher_agreement_summary(
    replay_report: Mapping[str, Any],
    *,
    selected_token_fn: Callable[[Mapping[str, Any]], str],
    teacher_token_fn: Callable[[Mapping[str, Any]], str],
) -> dict[str, Any]:
    scene_count = 0
    agreement = 0
    changed_target_count = 0
    changed_target_agreement = 0
    for scene in replay_report.get("scenes", []):
        scene_count += 1
        selected = str(selected_token_fn(scene))
        teacher = str(teacher_token_fn(scene))
        proxy = str(scene.get("selected_token", ""))
        if selected == teacher:
            agreement += 1
        if teacher != proxy:
            changed_target_count += 1
            if selected == teacher:
                changed_target_agreement += 1
    return {
        "scene_count": scene_count,
        "teacher_agreement_count": agreement,
        "teacher_agreement_rate": _rate(agreement, scene_count),
        "changed_target_count": changed_target_count,
        "changed_target_agreement_count": changed_target_agreement,
        "changed_target_agreement_rate": _rate(changed_target_agreement, changed_target_count),
    }


def markdown_student_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# nuPlan Bootstrap Student Generator",
        "",
        f"- Input replay JSON: `{report['input_replay_json']}`",
        f"- Teacher targets: `{report['teacher_targets_jsonl']}`",
        f"- Split unit: `{report['split']['split_unit']}`",
        f"- Train scenes: `{report['split']['train_scene_count']}`",
        f"- Holdout scenes: `{report['split']['holdout_scene_count']}`",
        f"- Training scene accuracy: `{report['training_metrics']['scene_accuracy']:.3f}`",
        "",
    ]
    for split_name in ("train", "holdout"):
        lines.extend(
            [
                f"## {split_name.title()}",
                "",
                "| Selector | Replay-infeasible | Rate | Selection Gap | Teacher Agree | Progress | ADE3 | Entropy |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for selector_name, metrics in report[split_name].items():
            gap = metrics["gap_summary"]
            agreement = metrics["teacher_agreement"]
            lines.append(
                f"| {selector_name} | {metrics['selected_replay_infeasible_count']} | "
                f"{metrics['selected_replay_infeasible_rate']:.3f} | "
                f"{gap['selection_gap_count']} | "
                f"{agreement['teacher_agreement_rate']:.3f} | "
                f"{metrics['mean_selected_progress_m']:.3f} | "
                f"{_optional(metrics['mean_ade_3s_m'])} | "
                f"{metrics['selected_token_entropy']:.3f} |"
            )
        lines.extend(["", "### Token Histograms", ""])
        for selector_name, metrics in report[split_name].items():
            histogram = json.dumps(metrics["selected_token_histogram"], sort_keys=True)
            lines.append(f"- `{selector_name}`: `{histogram}`")
        lines.append("")
    return "\n".join(lines)


def _rate(numerator: int, denominator: int) -> float:
    return round(float(numerator) / float(denominator), 6) if denominator else 0.0


def _mean(values: list[float]) -> float:
    return round(sum(float(value) for value in values) / len(values), 6) if values else 0.0


def _optional(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
