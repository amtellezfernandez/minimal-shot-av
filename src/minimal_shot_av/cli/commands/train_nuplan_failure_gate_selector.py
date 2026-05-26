#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Mapping

import numpy as np

from minimal_shot_av.cli.commands.train_nuplan_bootstrap_student_generator import scene_feature_names
from minimal_shot_av.cli.commands.train_nuplan_bootstrap_student_generator import scene_feature_row
from minimal_shot_av.cli.commands.train_nuplan_replay_value_selector import build_replay_value_examples
from minimal_shot_av.cli.commands.train_nuplan_replay_calibrated_selector import _select_replay_oracle_token
from minimal_shot_av.cli.commands.train_nuplan_replay_calibrated_selector import _select_with_score_model
from minimal_shot_av.cli.commands.train_nuplan_replay_calibrated_selector import _candidate_by_token
from minimal_shot_av.cli.commands.train_nuplan_replay_calibrated_selector import _token_is_replay_safe
from minimal_shot_av.cli.commands.train_nuplan_replay_calibrated_selector import build_replay_calibration_examples
from minimal_shot_av.cli.commands.train_nuplan_replay_calibrated_selector import evaluate_selector_policy
from minimal_shot_av.cli.commands.train_nuplan_replay_calibrated_selector import select_with_constrained_replay_risk
from minimal_shot_av.cli.commands.train_nuplan_replay_calibrated_selector import split_replay_report_by_db
from minimal_shot_av.model.nuplan_maneuver_token_selector import fit_replay_calibrated_selector
from minimal_shot_av.model.nuplan_maneuver_token_selector import fit_replay_value_mlp_selector
from minimal_shot_av.model.nuplan_maneuver_token_selector import fit_replay_value_selector
from minimal_shot_av.model.nuplan_maneuver_token_selector import fit_selector_mlp
from minimal_shot_av.model.nuplan_maneuver_token_selector import load_selector
from minimal_shot_av.model.nuplan_maneuver_token_selector import selector_from_payload
from minimal_shot_av.model.nuplan_maneuver_token_selector import selector_feature_row


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_INPUT = ROOT / "artifacts" / "corl2027" / "nuplan_public_replay_study_interaction250" / "replay.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "corl2027" / "nuplan_failure_gate_selector.json"
DEFAULT_REPORT_JSON = ROOT / "artifacts" / "corl2027" / "nuplan_failure_gate_selector_eval.json"
DEFAULT_REPORT_MD = ROOT / "artifacts" / "corl2027" / "nuplan_failure_gate_selector_eval.md"
MODEL_TYPE = "nuplan_failure_gate_selector_v1"
FALLBACK_FEATURE_NAMES = (
    "fallback_differs_from_proxy",
    "fallback_proxy_safe",
    "fallback_min_proxy_clearance_m",
    "fallback_final_progress_m",
    "fallback_score",
    "fallback_minus_proxy_clearance_m",
    "fallback_minus_proxy_progress_m",
    "fallback_minus_proxy_score",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a scene-level failure gate that invokes a calibrated selector only when proxy top-1 fails."
    )
    parser.add_argument("--input-replay-json", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--safe-selector-model", type=Path)
    parser.add_argument(
        "--fit-safe-selector",
        action="store_true",
        help="Fit the replay-calibrated fallback on the train DB split instead of loading an artifact.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-markdown", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument("--near-miss-threshold-m", type=float, default=1.0)
    parser.add_argument("--holdout-fraction", type=float, default=0.3)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gate-threshold", type=float, default=0.5)
    parser.add_argument(
        "--min-fallback-progress-retention",
        type=float,
        default=0.0,
        help="Only allow gate switches whose fallback keeps this fraction of proxy-token progress.",
    )
    parser.add_argument(
        "--threshold-sweep",
        default="0.2,0.3,0.4,0.5,0.6,0.7,0.8",
        help="Comma-separated gate thresholds to report as a safety-utility frontier.",
    )
    parser.add_argument("--positive-weight", type=float, default=4.0)
    parser.add_argument("--safe-hidden-dim", type=int, default=24)
    parser.add_argument("--safe-epochs", type=int, default=400)
    parser.add_argument("--safe-learning-rate", type=float, default=0.05)
    parser.add_argument("--safe-risk-weight", type=float, default=80.0)
    parser.add_argument("--safe-proxy-score-weight", type=float, default=1.0)
    parser.add_argument("--safe-progress-weight", type=float, default=0.02)
    parser.add_argument("--safe-unsafe-penalty", type=float, default=2.0)
    parser.add_argument(
        "--safe-selector-kind",
        choices=("replay_calibrated", "replay_value", "replay_value_mlp", "progress_oracle_student"),
        default="replay_calibrated",
        help="Fallback model to fit when --fit-safe-selector is used.",
    )
    parser.add_argument("--safe-value-fail-penalty", type=float, default=100.0)
    parser.add_argument("--safe-value-progress-weight", type=float, default=1.0)
    parser.add_argument("--safe-value-ade-weight", type=float, default=0.0)
    parser.add_argument("--safe-value-ridge-alpha", type=float, default=1.0e-3)
    parser.add_argument("--safe-student-positive-weight", type=float, default=8.0)
    parser.add_argument(
        "--fallback-mode",
        choices=("score", "risk_constrained", "risk_constrained_progress"),
        default="score",
        help="How to turn the calibrated safe model into a fallback action.",
    )
    parser.add_argument("--fallback-risk-threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    replay_report = json.loads(args.input_replay_json.read_text(encoding="utf-8"))
    train_report, holdout_report, split = split_replay_report_by_db(
        replay_report,
        holdout_fraction=float(args.holdout_fraction),
        seed=int(args.seed),
    )
    safe_selector, safe_selector_source, safe_fit_metrics = load_or_fit_safe_selector(
        args,
        train_report=train_report,
        near_miss_threshold_m=float(args.near_miss_threshold_m),
    )
    safe_token_fn = build_safe_token_fn(
        safe_selector,
        fallback_mode=str(args.fallback_mode),
        fallback_risk_threshold=float(args.fallback_risk_threshold),
    )
    train_examples = build_failure_gate_examples(
        train_report,
        safe_token_fn=safe_token_fn,
        near_miss_threshold_m=float(args.near_miss_threshold_m),
        positive_weight=float(args.positive_weight),
        min_fallback_progress_retention=float(args.min_fallback_progress_retention),
    )
    gate_model, fit_metrics = fit_failure_gate_mlp(
        train_examples,
        hidden_dim=int(args.hidden_dim),
        epochs=int(args.epochs),
        learning_rate=float(args.learning_rate),
        seed=int(args.seed),
    )
    report = {
        "schema": "nuplan_failure_gate_selector_eval_v1",
        "input_replay_json": str(args.input_replay_json),
        "safe_selector_source": safe_selector_source,
        "safe_selector_fit_metrics": safe_fit_metrics,
        "fallback_mode": str(args.fallback_mode),
        "fallback_risk_threshold": float(args.fallback_risk_threshold),
        "split": split,
        "near_miss_threshold_m": float(args.near_miss_threshold_m),
        "gate_threshold": float(args.gate_threshold),
        "min_fallback_progress_retention": float(args.min_fallback_progress_retention),
        "threshold_sweep": [float(value) for value in str(args.threshold_sweep).split(",") if value.strip()],
        "train_fit_metrics": fit_metrics,
        "train_example_count": len(train_examples),
        "train": evaluate_failure_gate_family(
            train_report,
            gate_model=gate_model,
            safe_token_fn=safe_token_fn,
            gate_threshold=float(args.gate_threshold),
            near_miss_threshold_m=float(args.near_miss_threshold_m),
            min_fallback_progress_retention=float(args.min_fallback_progress_retention),
        ),
        "holdout": evaluate_failure_gate_family(
            holdout_report,
            gate_model=gate_model,
            safe_token_fn=safe_token_fn,
            gate_threshold=float(args.gate_threshold),
            near_miss_threshold_m=float(args.near_miss_threshold_m),
            min_fallback_progress_retention=float(args.min_fallback_progress_retention),
        ),
    }
    report["threshold_sweep_results"] = {
        "train": threshold_sweep_results(
            train_report,
            gate_model=gate_model,
            safe_token_fn=safe_token_fn,
            thresholds=report["threshold_sweep"],
            near_miss_threshold_m=float(args.near_miss_threshold_m),
            min_fallback_progress_retention=float(args.min_fallback_progress_retention),
        ),
        "holdout": threshold_sweep_results(
            holdout_report,
            gate_model=gate_model,
            safe_token_fn=safe_token_fn,
            thresholds=report["threshold_sweep"],
            near_miss_threshold_m=float(args.near_miss_threshold_m),
            min_fallback_progress_retention=float(args.min_fallback_progress_retention),
        ),
    }
    report["threshold_recommendations"] = threshold_recommendations(
        report["threshold_sweep_results"]["train"],
        proxy_progress_m=report["train"]["proxy_selector"]["mean_selected_progress_m"],
    )
    report["threshold_recommendation_holdout"] = threshold_recommendation_holdout(
        report["threshold_recommendations"],
        report["threshold_sweep_results"]["holdout"],
    )
    payload = {
        **failure_gate_payload(gate_model),
        "training": {
            "input_replay_json": str(args.input_replay_json),
            "safe_selector_source": safe_selector_source,
            "safe_selector_fit_metrics": safe_fit_metrics,
            "safe_selector_payload": safe_selector.to_payload() if hasattr(safe_selector, "to_payload") else None,
            "fallback_mode": str(args.fallback_mode),
            "fallback_risk_threshold": float(args.fallback_risk_threshold),
            "min_fallback_progress_retention": float(args.min_fallback_progress_retention),
            "split": split,
            "gate_threshold": float(args.gate_threshold),
            "positive_weight": float(args.positive_weight),
            "min_fallback_progress_retention": float(args.min_fallback_progress_retention),
            "train_fit_metrics": fit_metrics,
            "train_example_count": len(train_examples),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.report_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.report_markdown.write_text(markdown_gate_report(report) + "\n", encoding="utf-8")
    print(markdown_gate_report(report))
    return 0


def load_failure_gate_policy(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    training = payload.get("training", {}) if isinstance(payload.get("training"), dict) else {}
    safe_payload = training.get("safe_selector_payload")
    if not isinstance(safe_payload, dict):
        raise ValueError(f"failure gate policy {path} does not contain an embedded safe selector payload")
    return {
        "gate_model": payload,
        "safe_selector": selector_from_payload(safe_payload),
        "fallback_mode": str(training.get("fallback_mode", "score")),
        "fallback_risk_threshold": float(training.get("fallback_risk_threshold", 0.5)),
        "gate_threshold": float(training.get("gate_threshold", 0.5)),
        "min_fallback_progress_retention": float(training.get("min_fallback_progress_retention", 0.0)),
    }


def select_with_failure_gate_policy(scene: Mapping[str, Any], policy: Mapping[str, Any]) -> str:
    safe_token_fn = build_safe_token_fn(
        policy["safe_selector"],
        fallback_mode=str(policy.get("fallback_mode", "score")),
        fallback_risk_threshold=float(policy.get("fallback_risk_threshold", 0.5)),
    )
    return select_with_failure_gate(
        scene,
        gate_model=policy["gate_model"],
        safe_token_fn=safe_token_fn,
        gate_threshold=float(policy.get("gate_threshold", 0.5)),
        min_fallback_progress_retention=float(policy.get("min_fallback_progress_retention", 0.0)),
    )


def build_safe_token_fn(
    safe_selector: Any,
    *,
    fallback_mode: str,
    fallback_risk_threshold: float,
) -> Callable[[Mapping[str, Any]], str]:
    if fallback_mode == "risk_constrained":
        if not hasattr(safe_selector, "predict_risk"):
            raise ValueError("risk_constrained fallback requires a replay-calibrated selector with predict_risk")
        return lambda scene: select_with_constrained_replay_risk(
            scene,
            safe_selector,
            risk_threshold=fallback_risk_threshold,
        )
    if fallback_mode == "risk_constrained_progress":
        if not hasattr(safe_selector, "predict_risk"):
            raise ValueError("risk_constrained_progress fallback requires a replay-calibrated selector")
        return lambda scene: select_highest_progress_below_risk(
            scene,
            safe_selector,
            risk_threshold=fallback_risk_threshold,
        )
    return lambda scene: _select_with_score_model(scene, safe_selector)


def load_or_fit_safe_selector(
    args: argparse.Namespace,
    *,
    train_report: Mapping[str, Any],
    near_miss_threshold_m: float,
) -> tuple[Any, str, dict[str, Any] | None]:
    if args.fit_safe_selector:
        if str(args.safe_selector_kind) == "progress_oracle_student":
            examples = build_progress_oracle_student_examples(
                train_report,
                near_miss_threshold_m=near_miss_threshold_m,
                positive_weight=float(args.safe_student_positive_weight),
            )
            selector, metrics = fit_selector_mlp(
                examples,
                hidden_dim=int(args.safe_hidden_dim),
                epochs=int(args.safe_epochs),
                learning_rate=float(args.safe_learning_rate),
                seed=int(args.seed),
            )
            metrics = {
                **metrics,
                "safe_selector_kind": "progress_oracle_student",
                "safe_student_positive_weight": float(args.safe_student_positive_weight),
            }
            return selector, "fit_progress_oracle_student_on_train_split", metrics
        if str(args.safe_selector_kind) in {"replay_value", "replay_value_mlp"}:
            examples = build_replay_value_examples(
                train_report,
                near_miss_threshold_m=near_miss_threshold_m,
                fail_penalty=float(args.safe_value_fail_penalty),
                objective_progress_weight=float(args.safe_value_progress_weight),
                objective_ade_weight=float(args.safe_value_ade_weight),
            )
            if str(args.safe_selector_kind) == "replay_value_mlp":
                selector, metrics = fit_replay_value_mlp_selector(
                    examples,
                    hidden_dim=int(args.safe_hidden_dim),
                    epochs=int(args.safe_epochs),
                    learning_rate=float(args.safe_learning_rate),
                    seed=int(args.seed),
                )
                selector_source = "fit_replay_value_mlp_on_train_split"
            else:
                selector, metrics = fit_replay_value_selector(
                    examples,
                    ridge_alpha=float(args.safe_value_ridge_alpha),
                )
                selector_source = "fit_replay_value_on_train_split"
            metrics = {
                **metrics,
                "safe_selector_kind": str(args.safe_selector_kind),
                "safe_value_fail_penalty": float(args.safe_value_fail_penalty),
                "safe_value_progress_weight": float(args.safe_value_progress_weight),
                "safe_value_ade_weight": float(args.safe_value_ade_weight),
            }
            return selector, selector_source, metrics
        examples = build_replay_calibration_examples(
            train_report,
            near_miss_threshold_m=near_miss_threshold_m,
        )
        selector, metrics = fit_replay_calibrated_selector(
            examples,
            hidden_dim=int(args.safe_hidden_dim),
            epochs=int(args.safe_epochs),
            learning_rate=float(args.safe_learning_rate),
            seed=int(args.seed),
            risk_weight=float(args.safe_risk_weight),
            proxy_score_weight=float(args.safe_proxy_score_weight),
            progress_weight=float(args.safe_progress_weight),
            unsafe_penalty=float(args.safe_unsafe_penalty),
            near_miss_threshold_m=near_miss_threshold_m,
        )
        metrics = {**metrics, "safe_selector_kind": "replay_calibrated"}
        return selector, "fit_replay_calibrated_on_train_split", metrics
    if args.safe_selector_model is None:
        raise ValueError("pass --safe-selector-model or --fit-safe-selector")
    return load_selector(args.safe_selector_model), str(args.safe_selector_model), None


def build_progress_oracle_student_examples(
    replay_report: Mapping[str, Any],
    *,
    near_miss_threshold_m: float,
    positive_weight: float,
) -> list[dict[str, Any]]:
    examples = []
    for scene in replay_report.get("scenes", []):
        target_token = select_progress_oracle_fallback_token(
            scene,
            near_miss_threshold_m=near_miss_threshold_m,
        )
        for candidate in scene.get("candidates", []):
            token = str(candidate.get("token", ""))
            label = 1.0 if token == target_token else 0.0
            examples.append(
                {
                    "scene_id": str(scene.get("scene_id", "")),
                    "source_db_file": str(scene.get("source_db_file", "")),
                    "token": token,
                    "label": label,
                    "example_weight": float(positive_weight) if label > 0.5 else 1.0,
                    "supervision_score": float(candidate.get("final_progress_m", 0.0)) if label > 0.5 else 0.0,
                    "features": selector_feature_row(scene, candidate),
                }
            )
    if not examples:
        raise ValueError("cannot fit progress-oracle fallback on an empty replay report")
    return examples


def select_progress_oracle_fallback_token(
    scene: Mapping[str, Any],
    *,
    near_miss_threshold_m: float,
) -> str:
    safe_candidates = [
        candidate
        for candidate in scene.get("candidates", [])
        if _token_is_replay_safe(scene, str(candidate.get("token", "")), near_miss_threshold_m)
    ]
    if not safe_candidates:
        return str(scene.get("selected_token", ""))
    selected = max(
        safe_candidates,
        key=lambda candidate: (
            float(candidate.get("final_progress_m", 0.0)),
            float(candidate.get("score", 0.0)),
            float(candidate.get("min_proxy_clearance_m", 0.0)),
        ),
    )
    return str(selected.get("token", scene.get("selected_token", "")))


def select_highest_progress_below_risk(
    scene: Mapping[str, Any],
    selector: Any,
    *,
    risk_threshold: float,
) -> str:
    candidates = list(scene.get("candidates", []))
    if not candidates:
        return str(scene.get("selected_token", ""))
    scored = []
    for candidate in candidates:
        features = candidate.get("features") if "features" in candidate else selector_feature_row(scene, candidate)
        scored.append(
            {
                "token": str(candidate.get("token", "")),
                "risk": float(selector.predict_risk(features)),
                "progress": float(candidate.get("final_progress_m", 0.0)),
                "score": float(candidate.get("score", 0.0)),
                "clearance": float(candidate.get("min_proxy_clearance_m", 0.0)),
            }
        )
    feasible = [row for row in scored if row["risk"] <= float(risk_threshold)]
    if feasible:
        selected = max(feasible, key=lambda row: (row["progress"], row["score"], row["clearance"], -row["risk"]))
    else:
        selected = min(scored, key=lambda row: (row["risk"], -row["progress"], -row["score"]))
    return str(selected["token"])


def build_failure_gate_examples(
    replay_report: Mapping[str, Any],
    *,
    safe_token_fn: Callable[[Mapping[str, Any]], str],
    near_miss_threshold_m: float,
    positive_weight: float,
    min_fallback_progress_retention: float = 0.0,
) -> list[dict[str, Any]]:
    examples = []
    for scene in replay_report.get("scenes", []):
        proxy_token = str(scene.get("selected_token", ""))
        safe_token = str(safe_token_fn(scene))
        proxy_safe = _token_is_replay_safe(scene, proxy_token, near_miss_threshold_m)
        safe_selector_safe = _token_is_replay_safe(scene, safe_token, near_miss_threshold_m)
        progress_allowed = fallback_meets_progress_retention(
            scene,
            fallback_token=safe_token,
            min_fallback_progress_retention=min_fallback_progress_retention,
        )
        should_switch = (not proxy_safe) and safe_selector_safe and progress_allowed
        examples.append(
            {
                "scene_id": str(scene.get("scene_id", "")),
                "source_db_file": str(scene.get("source_db_file", "")),
                "label": 1.0 if should_switch else 0.0,
                "example_weight": float(positive_weight) if should_switch else 1.0,
                "proxy_token": proxy_token,
                "safe_token": safe_token,
                "features": failure_gate_feature_row(scene, safe_token),
            }
        )
    if not examples:
        raise ValueError("cannot train failure gate on an empty replay report")
    return examples


def fit_failure_gate_mlp(
    examples: list[Mapping[str, Any]],
    *,
    hidden_dim: int,
    epochs: int,
    learning_rate: float,
    seed: int,
) -> tuple[dict[str, Any], dict[str, float]]:
    feature_names = failure_gate_feature_names()
    x = np.asarray(
        [[float(example["features"][name]) for name in feature_names] for example in examples],
        dtype=np.float64,
    )
    y = np.asarray([float(example["label"]) for example in examples], dtype=np.float64).reshape(-1, 1)
    weights = np.asarray([float(example.get("example_weight", 1.0)) for example in examples], dtype=np.float64)
    weights = np.clip(weights, 1.0e-6, None).reshape(-1, 1)
    weights = weights / float(weights.mean())
    feature_mean = x.mean(axis=0)
    feature_scale = x.std(axis=0)
    feature_scale[feature_scale < 1.0e-8] = 1.0
    x_norm = (x - feature_mean) / feature_scale
    rng = np.random.default_rng(seed)
    w1 = rng.normal(0.0, 0.12, size=(x_norm.shape[1], hidden_dim))
    b1 = np.zeros((hidden_dim,), dtype=np.float64)
    w2 = rng.normal(0.0, 0.12, size=(hidden_dim, 1))
    b2 = np.zeros((1,), dtype=np.float64)
    for _ in range(max(1, epochs)):
        z1 = x_norm @ w1 + b1
        h1 = np.maximum(z1, 0.0)
        logits = h1 @ w2 + b2
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
        grad_logits = weights * (probs - y) / x_norm.shape[0]
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
    model = {
        "model_type": MODEL_TYPE,
        "feature_names": list(feature_names),
        "hidden_dim": int(hidden_dim),
        "feature_mean": tuple(float(value) for value in feature_mean),
        "feature_scale": tuple(float(value) for value in feature_scale),
        "w1": tuple(tuple(float(value) for value in row) for row in w1),
        "b1": tuple(float(value) for value in b1),
        "w2": tuple(float(value) for value in w2[:, 0]),
        "b2": float(b2[0]),
    }
    probabilities = np.asarray([predict_failure_probability(model, example["features"]) for example in examples])
    labels = y[:, 0]
    loss = -np.mean(
        labels * np.log(np.clip(probabilities, 1.0e-8, 1.0))
        + (1.0 - labels) * np.log(np.clip(1.0 - probabilities, 1.0e-8, 1.0))
    )
    return model, {
        "example_count": float(len(examples)),
        "positive_rate": float(labels.mean()),
        "gate_accuracy": float(np.mean((probabilities >= 0.5) == (labels >= 0.5))),
        "gate_log_loss": float(loss),
    }


def evaluate_failure_gate_family(
    replay_report: Mapping[str, Any],
    *,
    gate_model: Mapping[str, Any],
    safe_token_fn: Callable[[Mapping[str, Any]], str],
    gate_threshold: float,
    near_miss_threshold_m: float,
    min_fallback_progress_retention: float = 0.0,
) -> dict[str, Any]:
    gated_fn = lambda scene: select_with_failure_gate(
        scene,
        gate_model=gate_model,
        safe_token_fn=safe_token_fn,
        gate_threshold=gate_threshold,
        min_fallback_progress_retention=min_fallback_progress_retention,
    )
    rows = {
        "proxy_selector": evaluate_selector_policy(
            replay_report,
            selector_name="proxy_selector",
            selected_token_fn=lambda scene: str(scene.get("selected_token", "")),
            near_miss_threshold_m=near_miss_threshold_m,
        ),
        "safe_selector": evaluate_selector_policy(
            replay_report,
            selector_name="safe_selector",
            selected_token_fn=safe_token_fn,
            near_miss_threshold_m=near_miss_threshold_m,
        ),
        "oracle_failure_gate": evaluate_selector_policy(
            replay_report,
            selector_name="oracle_failure_gate",
            selected_token_fn=lambda scene: select_with_oracle_failure_gate(
                scene,
                safe_token_fn=safe_token_fn,
                near_miss_threshold_m=near_miss_threshold_m,
                min_fallback_progress_retention=min_fallback_progress_retention,
            ),
            near_miss_threshold_m=near_miss_threshold_m,
        ),
        "progress_oracle_failure_gate": evaluate_selector_policy(
            replay_report,
            selector_name="progress_oracle_failure_gate",
            selected_token_fn=lambda scene: select_with_progress_oracle_failure_gate(
                scene,
                near_miss_threshold_m=near_miss_threshold_m,
            ),
            near_miss_threshold_m=near_miss_threshold_m,
        ),
        "failure_gate": evaluate_selector_policy(
            replay_report,
            selector_name="failure_gate",
            selected_token_fn=gated_fn,
            near_miss_threshold_m=near_miss_threshold_m,
        ),
        "replay_oracle": evaluate_selector_policy(
            replay_report,
            selector_name="replay_oracle",
            selected_token_fn=_select_replay_oracle_token,
            near_miss_threshold_m=near_miss_threshold_m,
        ),
    }
    rows["failure_gate"]["gate_diagnostics"] = failure_gate_diagnostics(
        replay_report,
        gate_model=gate_model,
        safe_token_fn=safe_token_fn,
        gate_threshold=gate_threshold,
        near_miss_threshold_m=near_miss_threshold_m,
        min_fallback_progress_retention=min_fallback_progress_retention,
    )
    rows["oracle_failure_gate"]["gate_diagnostics"] = oracle_failure_gate_diagnostics(
        replay_report,
        safe_token_fn=safe_token_fn,
        near_miss_threshold_m=near_miss_threshold_m,
        min_fallback_progress_retention=min_fallback_progress_retention,
    )
    return rows


def threshold_sweep_results(
    replay_report: Mapping[str, Any],
    *,
    gate_model: Mapping[str, Any],
    safe_token_fn: Callable[[Mapping[str, Any]], str],
    thresholds: list[float],
    near_miss_threshold_m: float,
    min_fallback_progress_retention: float = 0.0,
) -> list[dict[str, Any]]:
    rows = []
    for threshold in thresholds:
        token_fn = lambda scene, t=threshold: select_with_failure_gate(
            scene,
            gate_model=gate_model,
            safe_token_fn=safe_token_fn,
            gate_threshold=t,
            min_fallback_progress_retention=min_fallback_progress_retention,
        )
        metrics = evaluate_selector_policy(
            replay_report,
            selector_name=f"failure_gate_t{threshold:.2f}",
            selected_token_fn=token_fn,
            near_miss_threshold_m=near_miss_threshold_m,
        )
        diagnostics = failure_gate_diagnostics(
            replay_report,
            gate_model=gate_model,
            safe_token_fn=safe_token_fn,
            gate_threshold=float(threshold),
            near_miss_threshold_m=near_miss_threshold_m,
            min_fallback_progress_retention=min_fallback_progress_retention,
        )
        rows.append(
            {
                "threshold": float(threshold),
                "selected_replay_infeasible_rate": metrics["selected_replay_infeasible_rate"],
                "selected_replay_infeasible_count": metrics["selected_replay_infeasible_count"],
                "mean_selected_progress_m": metrics["mean_selected_progress_m"],
                "mean_ade_3s_m": metrics["mean_ade_3s_m"],
                "selected_token_entropy": metrics["selected_token_entropy"],
                "switch_rate": diagnostics["switch_rate"],
                "true_positive_switch_count": diagnostics["true_positive_switch_count"],
                "missed_avoidable_failure_count": diagnostics["missed_avoidable_failure_count"],
                "false_positive_switch_count": diagnostics["false_positive_switch_count"],
            }
        )
    return rows


def threshold_recommendations(
    train_rows: list[Mapping[str, Any]],
    *,
    proxy_progress_m: float,
) -> dict[str, Any]:
    return {
        "safety_first": _select_threshold_row(train_rows, min_progress_m=None),
        "balanced_80pct_proxy_progress": _select_threshold_row(
            train_rows,
            min_progress_m=0.8 * float(proxy_progress_m),
        ),
        "utility_90pct_proxy_progress": _select_threshold_row(
            train_rows,
            min_progress_m=0.9 * float(proxy_progress_m),
        ),
    }


def threshold_recommendation_holdout(
    recommendations: Mapping[str, Mapping[str, Any]],
    holdout_rows: list[Mapping[str, Any]],
) -> dict[str, Any]:
    by_threshold = {round(float(row["threshold"]), 6): dict(row) for row in holdout_rows}
    return {
        name: by_threshold.get(round(float(row["threshold"]), 6), {})
        for name, row in recommendations.items()
    }


def _select_threshold_row(
    rows: list[Mapping[str, Any]],
    *,
    min_progress_m: float | None,
) -> dict[str, Any]:
    eligible = [
        row for row in rows if min_progress_m is None or float(row["mean_selected_progress_m"]) >= min_progress_m
    ]
    if not eligible:
        eligible = list(rows)
    selected = min(
        eligible,
        key=lambda row: (
            float(row["selected_replay_infeasible_rate"]),
            -float(row["mean_selected_progress_m"]),
            float(row["threshold"]),
        ),
    )
    return dict(selected)


def select_with_failure_gate(
    scene: Mapping[str, Any],
    *,
    gate_model: Mapping[str, Any],
    safe_token_fn: Callable[[Mapping[str, Any]], str],
    gate_threshold: float,
    min_fallback_progress_retention: float = 0.0,
) -> str:
    safe_token = str(safe_token_fn(scene))
    probability = predict_failure_probability(gate_model, failure_gate_feature_row(scene, safe_token))
    progress_allowed = fallback_meets_progress_retention(
        scene,
        fallback_token=safe_token,
        min_fallback_progress_retention=min_fallback_progress_retention,
    )
    if probability >= float(gate_threshold) and progress_allowed:
        return safe_token
    return str(scene.get("selected_token", ""))


def select_with_oracle_failure_gate(
    scene: Mapping[str, Any],
    *,
    safe_token_fn: Callable[[Mapping[str, Any]], str],
    near_miss_threshold_m: float,
    min_fallback_progress_retention: float = 0.0,
) -> str:
    proxy_token = str(scene.get("selected_token", ""))
    safe_token = str(safe_token_fn(scene))
    proxy_safe = _token_is_replay_safe(scene, proxy_token, near_miss_threshold_m)
    fallback_safe = _token_is_replay_safe(scene, safe_token, near_miss_threshold_m)
    progress_allowed = fallback_meets_progress_retention(
        scene,
        fallback_token=safe_token,
        min_fallback_progress_retention=min_fallback_progress_retention,
    )
    if (not proxy_safe) and fallback_safe and progress_allowed:
        return safe_token
    return proxy_token


def select_with_progress_oracle_failure_gate(
    scene: Mapping[str, Any],
    *,
    near_miss_threshold_m: float,
) -> str:
    proxy_token = str(scene.get("selected_token", ""))
    if _token_is_replay_safe(scene, proxy_token, near_miss_threshold_m):
        return proxy_token
    return select_progress_oracle_fallback_token(
        scene,
        near_miss_threshold_m=near_miss_threshold_m,
    )


def fallback_meets_progress_retention(
    scene: Mapping[str, Any],
    *,
    fallback_token: str,
    min_fallback_progress_retention: float,
) -> bool:
    if float(min_fallback_progress_retention) <= 0.0:
        return True
    candidate_by_token = _candidate_by_token(scene)
    proxy_candidate = candidate_by_token.get(str(scene.get("selected_token", "")))
    fallback_candidate = candidate_by_token.get(str(fallback_token))
    if fallback_candidate is None:
        return False
    fallback_progress = float(fallback_candidate.get("final_progress_m", 0.0))
    if proxy_candidate is None:
        return fallback_progress >= 0.0
    proxy_progress = float(proxy_candidate.get("final_progress_m", 0.0))
    if proxy_progress <= 1.0e-6:
        return fallback_progress >= proxy_progress
    return fallback_progress >= float(min_fallback_progress_retention) * proxy_progress


def failure_gate_diagnostics(
    replay_report: Mapping[str, Any],
    *,
    gate_model: Mapping[str, Any],
    safe_token_fn: Callable[[Mapping[str, Any]], str],
    gate_threshold: float,
    near_miss_threshold_m: float,
    min_fallback_progress_retention: float = 0.0,
) -> dict[str, Any]:
    diagnostics = Counter()
    probabilities = []
    for scene in replay_report.get("scenes", []):
        proxy_token = str(scene.get("selected_token", ""))
        safe_token = str(safe_token_fn(scene))
        proxy_safe = _token_is_replay_safe(scene, proxy_token, near_miss_threshold_m)
        safe_selector_safe = _token_is_replay_safe(scene, safe_token, near_miss_threshold_m)
        progress_allowed = fallback_meets_progress_retention(
            scene,
            fallback_token=safe_token,
            min_fallback_progress_retention=min_fallback_progress_retention,
        )
        should_switch = (not proxy_safe) and safe_selector_safe and progress_allowed
        probability = predict_failure_probability(gate_model, failure_gate_feature_row(scene, safe_token))
        switch = probability >= float(gate_threshold) and progress_allowed
        probabilities.append(probability)
        diagnostics["switch_count" if switch else "keep_count"] += 1
        if should_switch and switch:
            diagnostics["true_positive_switch_count"] += 1
        if should_switch and not switch:
            diagnostics["missed_avoidable_failure_count"] += 1
        if (not should_switch) and switch:
            diagnostics["false_positive_switch_count"] += 1
        if proxy_safe and switch and not safe_selector_safe:
            diagnostics["introduced_failure_count"] += 1
    scene_count = len(list(replay_report.get("scenes", [])))
    return {
        "scene_count": scene_count,
        "switch_count": int(diagnostics["switch_count"]),
        "switch_rate": _rate(int(diagnostics["switch_count"]), scene_count),
        "true_positive_switch_count": int(diagnostics["true_positive_switch_count"]),
        "missed_avoidable_failure_count": int(diagnostics["missed_avoidable_failure_count"]),
        "false_positive_switch_count": int(diagnostics["false_positive_switch_count"]),
        "introduced_failure_count": int(diagnostics["introduced_failure_count"]),
        "switch_precision": _rate(
            int(diagnostics["true_positive_switch_count"]),
            int(diagnostics["switch_count"]),
        ),
        "switch_recall": _rate(
            int(diagnostics["true_positive_switch_count"]),
            int(diagnostics["true_positive_switch_count"]) + int(diagnostics["missed_avoidable_failure_count"]),
        ),
        "mean_gate_probability": _mean(probabilities),
    }


def oracle_failure_gate_diagnostics(
    replay_report: Mapping[str, Any],
    *,
    safe_token_fn: Callable[[Mapping[str, Any]], str],
    near_miss_threshold_m: float,
    min_fallback_progress_retention: float = 0.0,
) -> dict[str, Any]:
    diagnostics = Counter()
    for scene in replay_report.get("scenes", []):
        proxy_token = str(scene.get("selected_token", ""))
        safe_token = str(safe_token_fn(scene))
        proxy_safe = _token_is_replay_safe(scene, proxy_token, near_miss_threshold_m)
        fallback_safe = _token_is_replay_safe(scene, safe_token, near_miss_threshold_m)
        progress_allowed = fallback_meets_progress_retention(
            scene,
            fallback_token=safe_token,
            min_fallback_progress_retention=min_fallback_progress_retention,
        )
        should_switch = (not proxy_safe) and fallback_safe and progress_allowed
        if should_switch:
            diagnostics["switch_count"] += 1
            diagnostics["true_positive_switch_count"] += 1
        else:
            diagnostics["keep_count"] += 1
    scene_count = len(list(replay_report.get("scenes", [])))
    return {
        "scene_count": scene_count,
        "switch_count": int(diagnostics["switch_count"]),
        "switch_rate": _rate(int(diagnostics["switch_count"]), scene_count),
        "true_positive_switch_count": int(diagnostics["true_positive_switch_count"]),
        "missed_avoidable_failure_count": 0,
        "false_positive_switch_count": 0,
        "introduced_failure_count": 0,
        "switch_precision": 1.0 if diagnostics["switch_count"] else 0.0,
        "switch_recall": 1.0 if diagnostics["switch_count"] else 0.0,
    }


def predict_failure_probability(model: Mapping[str, Any], features: Mapping[str, float]) -> float:
    feature_names = tuple(str(name) for name in model["feature_names"])
    vector = np.asarray([float(features[name]) for name in feature_names], dtype=np.float64)
    mean = np.asarray(model["feature_mean"], dtype=np.float64)
    scale = np.asarray(model["feature_scale"], dtype=np.float64)
    normalized = (vector - mean) / scale
    hidden = np.maximum(
        normalized @ np.asarray(model["w1"], dtype=np.float64) + np.asarray(model["b1"], dtype=np.float64),
        0.0,
    )
    logit = float(hidden @ np.asarray(model["w2"], dtype=np.float64) + float(model["b2"]))
    return float(1.0 / (1.0 + np.exp(-max(-40.0, min(40.0, logit)))))


def failure_gate_feature_names() -> tuple[str, ...]:
    return tuple(scene_feature_names()) + FALLBACK_FEATURE_NAMES


def failure_gate_feature_row(scene: Mapping[str, Any], fallback_token: str) -> dict[str, float]:
    row = dict(scene_feature_row(scene))
    candidate_by_token = _candidate_by_token(scene)
    proxy_token = str(scene.get("selected_token", ""))
    proxy_candidate = candidate_by_token.get(proxy_token)
    fallback_candidate = candidate_by_token.get(str(fallback_token))
    row["fallback_differs_from_proxy"] = 1.0 if str(fallback_token) != proxy_token else 0.0
    row["fallback_proxy_safe"] = (
        1.0 if fallback_candidate is not None and bool(fallback_candidate.get("proxy_safe", False)) else 0.0
    )
    row["fallback_min_proxy_clearance_m"] = (
        float(fallback_candidate.get("min_proxy_clearance_m", 0.0)) if fallback_candidate is not None else 0.0
    )
    row["fallback_final_progress_m"] = (
        float(fallback_candidate.get("final_progress_m", 0.0)) if fallback_candidate is not None else 0.0
    )
    row["fallback_score"] = float(fallback_candidate.get("score", 0.0)) if fallback_candidate is not None else 0.0
    proxy_clearance = float(proxy_candidate.get("min_proxy_clearance_m", 0.0)) if proxy_candidate is not None else 0.0
    proxy_progress = float(proxy_candidate.get("final_progress_m", 0.0)) if proxy_candidate is not None else 0.0
    proxy_score = float(proxy_candidate.get("score", 0.0)) if proxy_candidate is not None else 0.0
    row["fallback_minus_proxy_clearance_m"] = row["fallback_min_proxy_clearance_m"] - proxy_clearance
    row["fallback_minus_proxy_progress_m"] = row["fallback_final_progress_m"] - proxy_progress
    row["fallback_minus_proxy_score"] = row["fallback_score"] - proxy_score
    return row


def failure_gate_payload(model: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "model_type": MODEL_TYPE,
        "feature_names": list(model["feature_names"]),
        "hidden_dim": int(model["hidden_dim"]),
        "feature_mean": list(model["feature_mean"]),
        "feature_scale": list(model["feature_scale"]),
        "w1": [list(row) for row in model["w1"]],
        "b1": list(model["b1"]),
        "w2": list(model["w2"]),
        "b2": float(model["b2"]),
    }


def markdown_gate_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# nuPlan Failure-Gated Selector",
        "",
        f"- Input replay JSON: `{report['input_replay_json']}`",
        f"- Safe selector: `{report['safe_selector_source']}`",
        f"- Fallback mode: `{report['fallback_mode']}`",
        f"- Fallback risk threshold: `{report['fallback_risk_threshold']:.3f}`",
        f"- Split unit: `{report['split']['split_unit']}`",
        f"- Train scenes: `{report['split']['train_scene_count']}`",
        f"- Holdout scenes: `{report['split']['holdout_scene_count']}`",
        f"- Gate threshold: `{report['gate_threshold']:.3f}`",
        f"- Train positive rate: `{report['train_fit_metrics']['positive_rate']:.3f}`",
        f"- Train gate accuracy: `{report['train_fit_metrics']['gate_accuracy']:.3f}`",
        "",
    ]
    for split_name in ("train", "holdout"):
        lines.extend(
            [
                f"## {split_name.title()}",
                "",
                "| Selector | Replay-infeasible | Rate | Progress | ADE3 | Entropy |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for name, metrics in report[split_name].items():
            lines.append(
                f"| {name} | {metrics['selected_replay_infeasible_count']} | "
                f"{metrics['selected_replay_infeasible_rate']:.3f} | "
                f"{metrics['mean_selected_progress_m']:.3f} | "
                f"{_optional(metrics['mean_ade_3s_m'])} | "
                f"{metrics['selected_token_entropy']:.3f} |"
            )
        gate = report[split_name]["failure_gate"]["gate_diagnostics"]
        oracle_gate = report[split_name]["oracle_failure_gate"]["gate_diagnostics"]
        lines.extend(
            [
                "",
                "### Gate Diagnostics",
                "",
                "| Metric | Value |",
                "|---|---:|",
                f"| switch count | {gate['switch_count']} |",
                f"| switch rate | {gate['switch_rate']:.3f} |",
                f"| true positive switches | {gate['true_positive_switch_count']} |",
                f"| missed avoidable failures | {gate['missed_avoidable_failure_count']} |",
                f"| false positive switches | {gate['false_positive_switch_count']} |",
                f"| introduced failures | {gate['introduced_failure_count']} |",
                f"| switch precision | {gate['switch_precision']:.3f} |",
                f"| switch recall | {gate['switch_recall']:.3f} |",
                f"| oracle switch count | {oracle_gate['switch_count']} |",
                "",
            ]
        )
        lines.extend(
            [
                "### Threshold Sweep",
                "",
                "| Threshold | Replay-infeasible Rate | Progress | ADE3 | Switch Rate | Missed Avoidable |",
                "|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in report["threshold_sweep_results"][split_name]:
            lines.append(
                f"| {row['threshold']:.2f} | {row['selected_replay_infeasible_rate']:.3f} | "
                f"{row['mean_selected_progress_m']:.3f} | "
                f"{_optional(row['mean_ade_3s_m'])} | "
                f"{row['switch_rate']:.3f} | {row['missed_avoidable_failure_count']} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Train-Selected Operating Points",
            "",
            "| Policy | Threshold | Holdout Replay-infeasible Rate | Holdout Progress | Holdout ADE3 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, train_row in report["threshold_recommendations"].items():
        holdout_row = report["threshold_recommendation_holdout"].get(name, {})
        lines.append(
            f"| {name} | {train_row['threshold']:.2f} | "
            f"{holdout_row.get('selected_replay_infeasible_rate', 0.0):.3f} | "
            f"{holdout_row.get('mean_selected_progress_m', 0.0):.3f} | "
            f"{_optional(holdout_row.get('mean_ade_3s_m'))} |"
        )
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
