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

from minimal_shot_av.cli.commands.analyze_nuplan_bootstrap_candidate_calibration import candidate_distribution
from minimal_shot_av.cli.commands.train_nuplan_bootstrap_student_generator import _fit_multiclass_mlp
from minimal_shot_av.cli.commands.train_nuplan_bootstrap_student_generator import candidate_token_order_from_report
from minimal_shot_av.cli.commands.train_nuplan_bootstrap_student_generator import predict_scene_token_student
from minimal_shot_av.cli.commands.train_nuplan_bootstrap_student_generator import scene_feature_names
from minimal_shot_av.cli.commands.train_nuplan_bootstrap_student_generator import scene_feature_row
from minimal_shot_av.cli.commands.train_nuplan_replay_calibrated_selector import _candidate_by_token
from minimal_shot_av.cli.commands.train_nuplan_replay_calibrated_selector import _replay_by_token
from minimal_shot_av.cli.commands.train_nuplan_replay_calibrated_selector import _select_replay_oracle_token
from minimal_shot_av.cli.commands.train_nuplan_replay_calibrated_selector import _select_with_score_model
from minimal_shot_av.cli.commands.train_nuplan_replay_calibrated_selector import _token_is_replay_safe
from minimal_shot_av.cli.commands.train_nuplan_replay_calibrated_selector import candidate_logged_ego_utility
from minimal_shot_av.cli.commands.train_nuplan_replay_calibrated_selector import evaluate_selector_policy
from minimal_shot_av.cli.commands.train_nuplan_replay_calibrated_selector import split_replay_report_by_db
from minimal_shot_av.cli.commands.train_nuplan_replay_value_selector import build_replay_value_examples
from minimal_shot_av.model.nuplan_maneuver_token_selector import fit_replay_value_mlp_selector
import numpy as np


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_INPUT = ROOT / "artifacts" / "corl2027" / "nuplan_public_replay_study_interaction1000_expert" / "replay.json"
DEFAULT_OUTPUT_JSON = ROOT / "artifacts" / "corl2027" / "nuplan_recoverable_regret_bootstrap.json"
DEFAULT_OUTPUT_MD = ROOT / "artifacts" / "corl2027" / "nuplan_recoverable_regret_bootstrap.md"
DEFAULT_TARGETS_JSONL = ROOT / "artifacts" / "corl2027" / "nuplan_recoverable_regret_targets.jsonl"
DEFAULT_STUDENT_JSON = ROOT / "artifacts" / "corl2027" / "nuplan_recoverable_regret_student.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run recoverable-regret bootstrap analysis and train a G1 scene-token student."
    )
    parser.add_argument("--input-replay-json", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--targets-jsonl", type=Path, default=DEFAULT_TARGETS_JSONL)
    parser.add_argument("--student-output", type=Path, default=DEFAULT_STUDENT_JSON)
    parser.add_argument("--near-miss-threshold-m", type=float, default=1.0)
    parser.add_argument("--holdout-fraction", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fail-penalty", type=float, default=250.0)
    parser.add_argument("--progress-weight", type=float, default=1.0)
    parser.add_argument("--ade-weight", type=float, default=2.0)
    parser.add_argument("--clearance-weight", type=float, default=0.0)
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=700)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--regret-sample-weight", type=float, default=4.0)
    parser.add_argument("--changed-target-weight", type=float, default=2.0)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument(
        "--student-kind",
        choices=("scene_token_mlp", "replay_value_mlp"),
        default="scene_token_mlp",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    replay_report = json.loads(args.input_replay_json.read_text(encoding="utf-8"))
    train_report, holdout_report, split = split_replay_report_by_db(
        replay_report,
        holdout_fraction=float(args.holdout_fraction),
        seed=int(args.seed),
    )
    score_config = {
        "near_miss_threshold_m": float(args.near_miss_threshold_m),
        "fail_penalty": float(args.fail_penalty),
        "progress_weight": float(args.progress_weight),
        "ade_weight": float(args.ade_weight),
        "clearance_weight": float(args.clearance_weight),
    }
    all_targets = recoverable_regret_targets(replay_report, teacher_name="oracle_at_k", **score_config)
    iteration_reports, student_model, student_fit = run_recoverable_regret_iterations(
        train_report,
        holdout_report,
        iterations=max(1, int(args.iterations)),
        score_config=score_config,
        hidden_dim=int(args.hidden_dim),
        epochs=int(args.epochs),
        learning_rate=float(args.learning_rate),
        seed=int(args.seed),
        regret_sample_weight=float(args.regret_sample_weight),
        changed_target_weight=float(args.changed_target_weight),
        student_kind=str(args.student_kind),
    )
    final_iteration = iteration_reports[-1]
    report = {
        "schema": "nuplan_recoverable_regret_bootstrap_v1",
        "input_replay_json": str(args.input_replay_json),
        "split": split,
        "score_config": score_config,
        "student_training": {
            "student_kind": str(args.student_kind),
            "regret_sample_weight": float(args.regret_sample_weight),
            "changed_target_weight": float(args.changed_target_weight),
            "iterations": max(1, int(args.iterations)),
        },
        "claim_boundary": (
            "This is a ManeuverToken surrogate for VLA candidate bootstrapping: G1 changes top-1 selection "
            "from recovered oracle@K targets, but does not yet regenerate new trajectory candidates."
        ),
        "candidate_distribution": candidate_distribution(replay_report),
        "student_fit_metrics": student_fit,
        "target_summary": target_summary(all_targets),
        "iterations": iteration_reports,
        "train": final_iteration["train"],
        "holdout": final_iteration["holdout"],
    }
    student_payload = student_model if isinstance(student_model, dict) else student_model.to_payload()
    payload = {
        **student_payload,
        "training": {
            "input_replay_json": str(args.input_replay_json),
            "split": split,
            "score_config": score_config,
            "student_fit_metrics": student_fit,
            "train_target_count": int(final_iteration["train_target_summary"]["target_count"]),
            "iterations": max(1, int(args.iterations)),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text(markdown_recoverable_regret_report(report) + "\n", encoding="utf-8")
    args.targets_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.targets_jsonl.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in all_targets),
        encoding="utf-8",
    )
    args.student_output.parent.mkdir(parents=True, exist_ok=True)
    args.student_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(markdown_recoverable_regret_report(report))
    return 0


def run_recoverable_regret_iterations(
    train_report: Mapping[str, Any],
    holdout_report: Mapping[str, Any],
    *,
    iterations: int,
    score_config: Mapping[str, float],
    hidden_dim: int,
    epochs: int,
    learning_rate: float,
    seed: int,
    regret_sample_weight: float,
    changed_target_weight: float,
    student_kind: str,
) -> tuple[list[dict[str, Any]], Any, dict[str, float]]:
    current_train = json.loads(json.dumps(train_report))
    current_holdout = json.loads(json.dumps(holdout_report))
    token_order = candidate_token_order_from_report(train_report)
    reports = []
    last_model: Any | None = None
    last_fit: dict[str, float] | None = None
    for iteration_index in range(1, max(1, int(iterations)) + 1):
        train_targets = recoverable_regret_targets(
            current_train,
            teacher_name=f"oracle_at_k_iter{iteration_index}",
            **score_config,
        )
        holdout_targets = recoverable_regret_targets(
            current_holdout,
            teacher_name=f"oracle_at_k_iter{iteration_index}",
            **score_config,
        )
        if student_kind == "replay_value_mlp":
            student_model, student_fit = fit_regret_weighted_replay_value_student(
                current_train,
                targets=train_targets,
                score_config=score_config,
                hidden_dim=hidden_dim,
                epochs=epochs,
                learning_rate=learning_rate,
                seed=seed + 1009 * (iteration_index - 1),
                regret_sample_weight=regret_sample_weight,
                changed_target_weight=changed_target_weight,
            )
            student_fn = lambda scene, model=student_model: _select_with_score_model(scene, model)
        else:
            student_model, student_fit = fit_regret_weighted_scene_token_student(
                current_train,
                targets=train_targets,
                token_order=token_order,
                hidden_dim=hidden_dim,
                epochs=epochs,
                learning_rate=learning_rate,
                seed=seed + 1009 * (iteration_index - 1),
                regret_sample_weight=regret_sample_weight,
                changed_target_weight=changed_target_weight,
            )
            student_fn = lambda scene, model=student_model: predict_scene_token_student(model, scene)
        reports.append(
            {
                "iteration": iteration_index,
                "student_fit_metrics": student_fit,
                "student_kind": student_kind,
                "train_target_summary": target_summary(train_targets),
                "holdout_target_summary": target_summary(holdout_targets),
                "train": evaluate_bootstrap_family(
                    current_train,
                    student_token_fn=student_fn,
                    teacher_token_fn=target_token_selector(train_targets),
                    score_config=score_config,
                ),
                "holdout": evaluate_bootstrap_family(
                    current_holdout,
                    student_token_fn=student_fn,
                    teacher_token_fn=target_token_selector(holdout_targets),
                    score_config=score_config,
                ),
            }
        )
        current_train = promote_student_top1_report(current_train, student_fn)
        current_holdout = promote_student_top1_report(current_holdout, student_fn)
        last_model = student_model
        last_fit = student_fit
    if last_model is None or last_fit is None:
        raise ValueError("recoverable-regret bootstrap did not run any iterations")
    return reports, last_model, last_fit


def promote_student_top1_report(
    replay_report: Mapping[str, Any],
    selected_token_fn: Callable[[Mapping[str, Any]], str],
) -> dict[str, Any]:
    copied = json.loads(json.dumps(replay_report))
    changed = 0
    applied = 0
    for scene in copied.get("scenes", []):
        token = str(selected_token_fn(scene))
        candidate = _candidate_by_token(scene).get(token)
        replay = _replay_by_token(scene).get(token)
        if candidate is None or replay is None:
            continue
        previous = str(scene.get("selected_token", ""))
        scene["selected_token"] = token
        scene["selected_token_source"] = "recoverable_regret_student_promoted"
        scene["selected_token_realized_min_clearance_m"] = replay.get("realized_min_clearance_m")
        scene["selected_token_realized_near_miss"] = replay.get("realized_near_miss")
        scene["selected_token_proxy_safe"] = bool(candidate.get("proxy_safe", False))
        scene["selected_token_min_proxy_clearance_m"] = candidate.get("min_proxy_clearance_m")
        scene["selected_token_score"] = candidate.get("score")
        scene["recoverable_regret_bootstrap"] = {
            "previous_selected_token": previous,
            "student_token": token,
        }
        applied += 1
        changed += 1 if token != previous else 0
    copied["scene_count"] = len(list(copied.get("scenes", [])))
    copied["recoverable_regret_bootstrap"] = {
        "applied_count": applied,
        "changed_token_count": changed,
        "changed_token_rate": rate(changed, copied["scene_count"]),
    }
    return copied


def recoverable_regret_targets(
    replay_report: Mapping[str, Any],
    *,
    teacher_name: str,
    near_miss_threshold_m: float,
    fail_penalty: float,
    progress_weight: float,
    ade_weight: float,
    clearance_weight: float,
) -> list[dict[str, Any]]:
    targets = []
    for scene in replay_report.get("scenes", []):
        proxy_token = str(scene.get("selected_token", ""))
        oracle_token, oracle_score = oracle_at_k_token_and_score(
            scene,
            near_miss_threshold_m=near_miss_threshold_m,
            fail_penalty=fail_penalty,
            progress_weight=progress_weight,
            ade_weight=ade_weight,
            clearance_weight=clearance_weight,
        )
        proxy_score = token_score(
            scene,
            proxy_token,
            near_miss_threshold_m=near_miss_threshold_m,
            fail_penalty=fail_penalty,
            progress_weight=progress_weight,
            ade_weight=ade_weight,
            clearance_weight=clearance_weight,
        )
        if oracle_token is None or proxy_score is None:
            continue
        targets.append(
            {
                "scene_id": str(scene.get("scene_id", "")),
                "source_db_file": str(scene.get("source_db_file", "")),
                "scenario_type": str(scene.get("scenario_type", "")),
                "teacher": teacher_name,
                "teacher_token": oracle_token,
                "proxy_token": proxy_token,
                "oracle_token": oracle_token,
                "proxy_score": float(proxy_score),
                "oracle_score": float(oracle_score),
                "recoverable_regret": float(oracle_score - proxy_score),
                "teacher_changes_proxy": oracle_token != proxy_token,
                "proxy_replay_safe": _token_is_replay_safe(scene, proxy_token, near_miss_threshold_m),
                "teacher_replay_safe": _token_is_replay_safe(scene, oracle_token, near_miss_threshold_m),
            }
        )
    return targets


def evaluate_bootstrap_family(
    replay_report: Mapping[str, Any],
    *,
    student_token_fn: Callable[[Mapping[str, Any]], str],
    teacher_token_fn: Callable[[Mapping[str, Any]], str],
    score_config: Mapping[str, float],
) -> dict[str, Any]:
    near_miss_threshold_m = float(score_config["near_miss_threshold_m"])
    selectors = {
        "g0_proxy_top1": lambda scene: str(scene.get("selected_token", "")),
        "g1_student_top1": student_token_fn,
        "teacher_promoted_upper_bound": teacher_token_fn,
        "oracle_at_k": lambda scene: oracle_at_k_token_and_score(scene, **score_config)[0]
        or str(scene.get("selected_token", "")),
        "replay_oracle_safe_only": _select_replay_oracle_token,
    }
    rows = {}
    for name, fn in selectors.items():
        metrics = evaluate_selector_policy(
            replay_report,
            selector_name=name,
            selected_token_fn=fn,
            near_miss_threshold_m=near_miss_threshold_m,
        )
        rows[name] = {
            **metrics,
            "score_summary": score_summary(
                replay_report,
                selected_token_fn=fn,
                score_config=score_config,
            ),
            "gap_summary": regret_gap_summary(
                replay_report,
                selected_token_fn=fn,
                score_config=score_config,
            ),
        }
    return rows


def fit_regret_weighted_scene_token_student(
    replay_report: Mapping[str, Any],
    *,
    targets: list[Mapping[str, Any]],
    token_order: tuple[str, ...],
    hidden_dim: int,
    epochs: int,
    learning_rate: float,
    seed: int,
    regret_sample_weight: float,
    changed_target_weight: float,
) -> tuple[Any, dict[str, float]]:
    scenes = list(replay_report.get("scenes", []))
    if not scenes:
        raise ValueError("cannot train G1 student on an empty replay report")
    target_by_key = {
        f"{target.get('source_db_file', '')}::{target.get('scene_id', '')}": target
        for target in targets
    }
    target_by_scene_id = {str(target.get("scene_id", "")): target for target in targets}
    feature_names = scene_feature_names(token_order)
    labels = []
    sample_weights = []
    max_regret = max([float(target.get("recoverable_regret", 0.0)) for target in targets] or [1.0])
    max_regret = max(max_regret, 1.0e-6)
    for scene in scenes:
        target = target_by_key.get(scene_key(scene)) or target_by_scene_id.get(str(scene.get("scene_id", "")))
        teacher_token = str(target.get("teacher_token", scene.get("selected_token", ""))) if target else str(
            scene.get("selected_token", "")
        )
        labels.append(token_order.index(teacher_token) if teacher_token in token_order else 0)
        regret = float(target.get("recoverable_regret", 0.0)) if target else 0.0
        changed = bool(target.get("teacher_changes_proxy", False)) if target else False
        sample_weights.append(
            1.0
            + float(regret_sample_weight) * min(1.0, regret / max_regret)
            + (float(changed_target_weight) if changed else 0.0)
        )
    x = np.asarray(
        [
            [float(scene_feature_row(scene, token_order=token_order)[name]) for name in feature_names]
            for scene in scenes
        ],
        dtype=np.float64,
    )
    y = np.asarray(labels, dtype=np.int64)
    raw_weights = np.asarray(sample_weights, dtype=np.float64)
    class_counts = np.bincount(y, minlength=len(token_order)).astype(np.float64)
    class_weights = np.zeros_like(class_counts)
    nonzero = class_counts > 0.0
    class_weights[nonzero] = float(len(scenes)) / (float(nonzero.sum()) * class_counts[nonzero])
    weights = raw_weights * class_weights[y]
    weights = weights / max(1.0e-6, float(weights.mean()))
    fitted = _fit_multiclass_mlp(
        x=x,
        y=y,
        sample_weights=weights,
        hidden_dim=hidden_dim,
        epochs=epochs,
        learning_rate=learning_rate,
        seed=seed,
        class_count=len(token_order),
    )
    model = {
        "model_type": "nuplan_bootstrap_scene_token_student_mlp_v1",
        "feature_names": list(feature_names),
        "token_order": list(token_order),
        "hidden_dim": int(hidden_dim),
        **fitted,
    }
    predictions = [predict_scene_token_student(model, scene) for scene in scenes]
    accuracy = sum(1 for pred, label in zip(predictions, y) if pred == token_order[int(label)]) / len(scenes)
    changed_indices = [
        index
        for index, scene in enumerate(scenes)
        if bool(
            (
                target_by_key.get(scene_key(scene))
                or target_by_scene_id.get(str(scene.get("scene_id", "")))
                or {}
            ).get("teacher_changes_proxy", False)
        )
    ]
    changed_accuracy = (
        sum(1 for index in changed_indices if predictions[index] == token_order[int(y[index])]) / len(changed_indices)
        if changed_indices
        else 0.0
    )
    return model, {
        "scene_count": float(len(scenes)),
        "scene_accuracy": float(accuracy),
        "changed_target_accuracy": float(changed_accuracy),
        "mean_sample_weight_raw": float(raw_weights.mean()),
        "max_sample_weight_raw": float(raw_weights.max()),
        "regret_sample_weight": float(regret_sample_weight),
        "changed_target_weight": float(changed_target_weight),
    }


def fit_regret_weighted_replay_value_student(
    replay_report: Mapping[str, Any],
    *,
    targets: list[Mapping[str, Any]],
    score_config: Mapping[str, float],
    hidden_dim: int,
    epochs: int,
    learning_rate: float,
    seed: int,
    regret_sample_weight: float,
    changed_target_weight: float,
) -> tuple[dict[str, Any], dict[str, float]]:
    examples = build_replay_value_examples(
        replay_report,
        near_miss_threshold_m=float(score_config["near_miss_threshold_m"]),
        fail_penalty=float(score_config["fail_penalty"]),
        objective_progress_weight=float(score_config["progress_weight"]),
        objective_ade_weight=float(score_config["ade_weight"]),
    )
    target_by_key = {
        f"{target.get('source_db_file', '')}::{target.get('scene_id', '')}": target
        for target in targets
    }
    target_by_scene_id = {str(target.get("scene_id", "")): target for target in targets}
    max_regret = max([float(target.get("recoverable_regret", 0.0)) for target in targets] or [1.0])
    max_regret = max(max_regret, 1.0e-6)
    weighted_examples = []
    for example in examples:
        target = target_by_key.get(f"{example.get('source_db_file', '')}::{example.get('scene_id', '')}") or target_by_scene_id.get(
            str(example.get("scene_id", ""))
        )
        regret = float(target.get("recoverable_regret", 0.0)) if target else 0.0
        changed = bool(target.get("teacher_changes_proxy", False)) if target else False
        weighted_examples.append(
            {
                **example,
                "example_weight": (
                    1.0
                    + float(regret_sample_weight) * min(1.0, regret / max_regret)
                    + (float(changed_target_weight) if changed else 0.0)
                ),
            }
        )
    selector, fit_metrics = fit_replay_value_mlp_selector(
        weighted_examples,
        hidden_dim=hidden_dim,
        epochs=epochs,
        learning_rate=learning_rate,
        seed=seed,
    )
    return selector, fit_metrics


def oracle_at_k_token_and_score(
    scene: Mapping[str, Any],
    *,
    near_miss_threshold_m: float,
    fail_penalty: float,
    progress_weight: float,
    ade_weight: float,
    clearance_weight: float,
) -> tuple[str | None, float]:
    token_order = tuple(str(candidate.get("token", "")) for candidate in scene.get("candidates", []))
    scored = []
    for candidate in scene.get("candidates", []):
        score = token_score(
            scene,
            str(candidate.get("token", "")),
            near_miss_threshold_m=near_miss_threshold_m,
            fail_penalty=fail_penalty,
            progress_weight=progress_weight,
            ade_weight=ade_weight,
            clearance_weight=clearance_weight,
        )
        if score is not None:
            scored.append((str(candidate.get("token", "")), float(score)))
    if not scored:
        return None, 0.0
    return max(scored, key=lambda row: (row[1], -token_order_index(row[0], token_order)))


def token_score(
    scene: Mapping[str, Any],
    token: str,
    *,
    near_miss_threshold_m: float,
    fail_penalty: float,
    progress_weight: float,
    ade_weight: float,
    clearance_weight: float,
) -> float | None:
    candidate = _candidate_by_token(scene).get(str(token))
    replay = _replay_by_token(scene).get(str(token))
    if candidate is None or replay is None or replay.get("realized_min_clearance_m") is None:
        return None
    utility = candidate_logged_ego_utility(candidate, scene)
    if utility is None:
        return None
    clearance = float(replay["realized_min_clearance_m"])
    replay_fail = clearance < float(near_miss_threshold_m)
    return (
        -float(fail_penalty) * (1.0 if replay_fail else 0.0)
        + float(progress_weight) * float(candidate.get("final_progress_m", 0.0))
        - float(ade_weight) * float(utility["ade_3s_m"])
        + float(clearance_weight) * clearance
    )


def score_summary(
    replay_report: Mapping[str, Any],
    *,
    selected_token_fn: Callable[[Mapping[str, Any]], str],
    score_config: Mapping[str, float],
) -> dict[str, float]:
    scores = []
    for scene in replay_report.get("scenes", []):
        score = token_score(scene, str(selected_token_fn(scene)), **score_config)
        if score is not None:
            scores.append(float(score))
    return {
        "scored_scene_count": len(scores),
        "mean_score": mean(scores),
        "median_score": median(scores),
    }


def regret_gap_summary(
    replay_report: Mapping[str, Any],
    *,
    selected_token_fn: Callable[[Mapping[str, Any]], str],
    score_config: Mapping[str, float],
) -> dict[str, float]:
    regrets = []
    positive_regrets = 0
    generation_gap = 0
    scene_count = 0
    for scene in replay_report.get("scenes", []):
        selected_score = token_score(scene, str(selected_token_fn(scene)), **score_config)
        oracle_token, oracle_score = oracle_at_k_token_and_score(scene, **score_config)
        if selected_score is None or oracle_token is None:
            continue
        scene_count += 1
        regret = max(0.0, float(oracle_score) - float(selected_score))
        regrets.append(regret)
        if regret > 1.0e-6:
            positive_regrets += 1
        if not _token_is_replay_safe(scene, oracle_token, float(score_config["near_miss_threshold_m"])):
            generation_gap += 1
    return {
        "scene_count": scene_count,
        "mean_recoverable_regret": mean(regrets),
        "median_recoverable_regret": median(regrets),
        "positive_regret_count": positive_regrets,
        "positive_regret_rate": rate(positive_regrets, scene_count),
        "generation_gap_count": generation_gap,
        "generation_gap_rate": rate(generation_gap, scene_count),
    }


def target_token_selector(targets: list[Mapping[str, Any]]) -> Callable[[Mapping[str, Any]], str]:
    target_by_key = {
        f"{target.get('source_db_file', '')}::{target.get('scene_id', '')}": str(target.get("teacher_token", ""))
        for target in targets
    }
    target_by_scene_id = {
        str(target.get("scene_id", "")): str(target.get("teacher_token", ""))
        for target in targets
    }

    def select(scene: Mapping[str, Any]) -> str:
        key = f"{scene.get('source_db_file', '')}::{scene.get('scene_id', '')}"
        return target_by_key.get(key) or target_by_scene_id.get(str(scene.get("scene_id", ""))) or str(
            scene.get("selected_token", "")
        )

    return select


def scene_key(scene: Mapping[str, Any]) -> str:
    return f"{scene.get('source_db_file', '')}::{scene.get('scene_id', '')}"


def target_summary(targets: list[Mapping[str, Any]]) -> dict[str, Any]:
    regrets = [float(row["recoverable_regret"]) for row in targets]
    changed = sum(1 for row in targets if bool(row.get("teacher_changes_proxy")))
    return {
        "target_count": len(targets),
        "teacher_changes_proxy_count": changed,
        "teacher_changes_proxy_rate": rate(changed, len(targets)),
        "mean_recoverable_regret": mean(regrets),
        "median_recoverable_regret": median(regrets),
        "token_histogram": dict(sorted(Counter(str(row["teacher_token"]) for row in targets).items())),
    }


def token_order_index(token: str, token_order: tuple[str, ...]) -> int:
    return token_order.index(token) if token in token_order else len(token_order)


def markdown_recoverable_regret_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# nuPlan Recoverable-Regret Bootstrap",
        "",
        f"- Input replay JSON: `{report['input_replay_json']}`",
        f"- Split unit: `{report['split']['split_unit']}`",
        f"- Train scenes: `{report['split']['train_scene_count']}`",
        f"- Holdout scenes: `{report['split']['holdout_scene_count']}`",
        f"- Claim boundary: {report['claim_boundary']}",
        "",
        "## Target Summary",
        "",
        f"- Target count: `{report['target_summary']['target_count']}`",
        f"- Teacher changes proxy: `{report['target_summary']['teacher_changes_proxy_count']}`",
        f"- Mean recoverable regret: `{report['target_summary']['mean_recoverable_regret']:.3f}`",
        f"- Target tokens: `{json.dumps(report['target_summary']['token_histogram'], sort_keys=True)}`",
        "",
    ]
    if len(report.get("iterations", [])) > 1:
        lines.extend(["## Iteration Summary", ""])
        for iteration in report["iterations"]:
            holdout = iteration["holdout"]
            proxy = holdout["g0_proxy_top1"]
            student = holdout["g1_student_top1"]
            teacher = holdout["teacher_promoted_upper_bound"]
            lines.extend(
                [
                    f"### Iteration {iteration['iteration']}",
                    "",
                    "| Policy | Score | Regret | Replay Fail | Progress | ADE3 |",
                    "|---|---:|---:|---:|---:|---:|",
                    iteration_policy_row("current_proxy", proxy),
                    iteration_policy_row("student_top1", student),
                    iteration_policy_row("teacher_upper_bound", teacher),
                    "",
                ]
            )
    for split_name in ("train", "holdout"):
        lines.extend(
            [
                f"## {split_name.title()}",
                "",
                "| Iteration/Policy | Score | Regret | Positive Regret | Replay Fail | Progress | ADE3 | Entropy |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for name, metrics in report[split_name].items():
            score = metrics["score_summary"]
            gap = metrics["gap_summary"]
            lines.append(
                f"| {name} | {score['mean_score']:.3f} | "
                f"{gap['mean_recoverable_regret']:.3f} | "
                f"{gap['positive_regret_count']} | "
                f"{metrics['selected_replay_infeasible_rate']:.3f} | "
                f"{metrics['mean_selected_progress_m']:.3f} | "
                f"{optional_metric(metrics['mean_ade_3s_m'])} | "
                f"{metrics['selected_token_entropy']:.3f} |"
            )
        lines.append("")
    return "\n".join(lines)


def iteration_policy_row(label: str, metrics: Mapping[str, Any]) -> str:
    return (
        f"| {label} | {metrics['score_summary']['mean_score']:.3f} | "
        f"{metrics['gap_summary']['mean_recoverable_regret']:.3f} | "
        f"{metrics['selected_replay_infeasible_rate']:.3f} | "
        f"{metrics['mean_selected_progress_m']:.3f} | "
        f"{optional_metric(metrics['mean_ade_3s_m'])} |"
    )


def mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0


def median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return round(ordered[middle], 6)
    return round(0.5 * (ordered[middle - 1] + ordered[middle]), 6)


def rate(numerator: int, denominator: int) -> float:
    return round(float(numerator) / float(denominator), 6) if denominator else 0.0


def optional_metric(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
